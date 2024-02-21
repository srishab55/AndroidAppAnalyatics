from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import col, expr
from pyspark.sql.types import DateType

def bin_date_field_by_year_with_bucketizer(df, field, bin_ranges):
    # Convert date string to date type
    df = df.withColumn(field, col(field).cast(DateType()))
    # Extract year from date
    df = df.withColumn(f"{field}_year", expr(f"YEAR(`{field}`)"))

    # Define splits for bucketizer
    splits = [start_year for (start_year, _) in bin_ranges] + [float('inf')]
    # Apply bucketizer
    bucketizer = Bucketizer(splits=splits, inputCol=f"{field}_year", outputCol=f"{field}_bin")
    bucketed_df = bucketizer.transform(df)

    # Map bin indices to bin ranges
    for i, (start_year, end_year) in enumerate(bin_ranges):
        bucketed_df = bucketed_df.withColumn(f"{field}_bin",
                                             expr(f"CASE WHEN `{field}_bin` = {i} THEN '{start_year}-{end_year}' ELSE NULL END"))

    return bucketed_df

# Example usage:
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Bin Date Fields") \
        .getOrCreate()

    # Sample DataFrame (replace this with your DataFrame)
    data = [("2022-01-01",), ("2022-03-15",), ("2023-06-20",), ("2024-09-10",), ("2025-12-31",)]
    df = spark.createDataFrame(data, ["date"])

    # Define bin ranges for years
    bin_ranges = [(2010, 2019), (2020, 2029), (2030, 2039)]  # Define your bin ranges here

    # Convert date field into bins by year using Bucketizer
    df = bin_date_field_by_year_with_bucketizer(df, "date", bin_ranges)

    # Show the results
    df.show()
