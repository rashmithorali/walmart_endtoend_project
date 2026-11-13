#------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------INITIAL SETUP------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Notebook 1: External Volume Setup
# MAGIC Maps ADLS Gen2 path `adfraw/orders` to a Unity Catalog External Volume.

# COMMAND ----------
# Parameters
catalog_name = "powerinsight"
schema_name = "raw"
volume_name = "raw_volume"

# ADLS Gen2 storage path (using abfss protocol)
storage_account = "powerinsightsa" # replace with your ADLS storage account name
container_name = "adflanding"
base_adls_path = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/orders"

# COMMAND ----------
# Ensure Catalog and Schema exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# COMMAND ----------
# Create the external volume pointing directly to the ADLS Gen2 landing folder
spark.sql(f"""
CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}
LOCATION '{base_adls_path}'
COMMENT 'External volume pointing to ADLS Gen2 container adfraw/orders'
""")

# COMMAND ----------
# Verify volume creation
display(spark.sql(f"DESCRIBE VOLUME {catalog_name}.{schema_name}.{volume_name}"))


#------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------RAW-------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Notebook 2: Multi-Part Parquet Ingestion to Raw Catalog
# MAGIC Reads partitioned Parquet folder via Unity Catalog Volume and writes to `raw_catalog.orders.orders_raw`.

# COMMAND ----------
# Dynamic parameters (ADF / Databricks Workflows)
dbutils.widgets.text("year", "2026")
dbutils.widgets.text("month", "10")
dbutils.widgets.text("day", "17")
dbutils.widgets.text("target_catalog", "raw_catalog")
dbutils.widgets.text("target_schema", "orders")

year_val = dbutils.widgets.get("year").zfill(4)
month_val = dbutils.widgets.get("month").zfill(2)
day_val = dbutils.widgets.get("day").zfill(2)
target_catalog = dbutils.widgets.get("target_catalog")
target_schema = dbutils.widgets.get("target_schema")

# COMMAND ----------
from pyspark.sql import functions as F

# Target Table
target_table = f"{target_catalog}.{target_schema}.orders_raw"

# Unity Catalog volume directory path pointing to the partition folder
# Reading the folder path loads all part-*.parquet files inside it automatically
partition_volume_dir = (
    f"/Volumes/{target_catalog}/landing/orders_landing/"
    f"year={year_val}/month={month_val}/day={day_val}"
)

# COMMAND ----------
# Read the partitioned directory of Parquet files
raw_source_df = (
    spark.read.format("parquet")
    .load(partition_volume_dir)
)

# Enforce partition values and add audit/lineage metadata
raw_ingested_df = (
    raw_source_df
    .withColumn("year", F.lit(int(year_val)))
    .withColumn("month", F.lit(int(month_val)))
    .withColumn("day", F.lit(int(day_val)))
    .withColumn("_raw_ingested_at", F.current_timestamp())
    .withColumn("_source_file_path", F.input_file_name())
)

# COMMAND ----------
# Ensure Target Schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_catalog}.{target_schema}")

# Idempotent write: replaceWhere replaces ONLY the matched incoming partition.
# If re-run for 2026/10/17, it cleanly overwrites only this slice of data.
(
    raw_ingested_df.write
    .format("delta")
    .mode("overwrite")
    .option("replaceWhere", f"year = {int(year_val)} AND month = {int(month_val)} AND day = {int(day_val)}")
    .partitionBy("year", "month", "day")
    .saveAsTable(target_table)
)

print(f"Successfully loaded partition year={year_val}/month={month_val}/day={day_val} into {target_table}")

#------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------BRONZE----------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------

# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Notebook 3: Raw to Bronze Transformation
# MAGIC Cleans, deduplicates, and upserts Parquet-derived records into `bronze_catalog.orders.orders_bronze`.

# COMMAND ----------
dbutils.widgets.text("year", "2026")
dbutils.widgets.text("month", "10")
dbutils.widgets.text("day", "17")
dbutils.widgets.text("source_catalog", "raw_catalog")
dbutils.widgets.text("target_catalog", "bronze_catalog")

year_val = int(dbutils.widgets.get("year"))
month_val = int(dbutils.widgets.get("month"))
day_val = int(dbutils.widgets.get("day"))
source_catalog = dbutils.widgets.get("source_catalog")
target_catalog = dbutils.widgets.get("target_catalog")

source_table = f"{source_catalog}.orders.orders_raw"
target_table = f"{target_catalog}.orders.orders_bronze"

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# COMMAND ----------
# Read only the current partition slice from the Raw catalog table
raw_partition_df = (
    spark.read.table(source_table)
    .filter(
        (F.col("year") == year_val) & 
        (F.col("month") == month_val) & 
        (F.col("day") == day_val)
    )
)

# COMMAND ----------
# 1. Null handling on critical business keys
non_null_df = raw_partition_df.filter(
    F.col("order_id").isNotNull() & F.col("customer_id").isNotNull()
)

# 2. In-batch deduplication across multiple Parquet files
# Parquet batch slices often have duplicate records across part-files; keep the latest update
dedup_window = (
    Window.partitionBy("order_id")
    .orderBy(F.col("updated_at").desc_nulls_last(), F.col("_raw_ingested_at").desc())
)

bronze_prepared_df = (
    non_null_df
    .withColumn("row_rank", F.row_number().over(dedup_window))
    .filter(F.col("row_rank") == 1)
    .drop("row_rank")
    .withColumn("order_id", F.trim(F.col("order_id").cast("string")))
    .withColumn("customer_id", F.trim(F.col("customer_id").cast("string")))
    .withColumn("order_amount", F.col("order_amount").cast("decimal(18,2)"))
    .withColumn("order_status", F.upper(F.trim(F.col("order_status"))))
    .withColumn("_bronze_processed_at", F.current_timestamp())
)

# COMMAND ----------
# Ensure Bronze target catalog & schema exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {target_catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_catalog}.orders")

# Create target Delta table if it does not already exist
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {target_table} (
    order_id STRING,
    customer_id STRING,
    order_amount DECIMAL(18,2),
    order_status STRING,
    updated_at TIMESTAMP,
    year INT,
    month INT,
    day INT,
    _raw_ingested_at TIMESTAMP,
    _source_file_path STRING,
    _bronze_processed_at TIMESTAMP
)
USING DELTA
PARTITIONED BY (year, month, day)
""")

# COMMAND ----------
# Idempotent Upsert via Delta MERGE INTO
target_delta = DeltaTable.forName(spark, target_table)

(
    target_delta.alias("target")
    .merge(
        bronze_prepared_df.alias("source"),
        """
        target.order_id = source.order_id 
        AND target.year = source.year 
        AND target.month = source.month 
        AND target.day = source.day
        """
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)

print(f"Bronze processing complete for {year_val}-{month_val:02d}-{day_val:02d}. Target updated: {target_table}")