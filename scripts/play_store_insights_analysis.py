from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import col


def bin_numerical_field_with_bucketizer(df, field, bin_ranges, bin_labels):
    # Define splits for bucketizer
    splits = [float('-inf')] + [end for (start, end) in bin_ranges] + [float('inf')]
    # Apply bucketizer
    bucketizer = Bucketizer(splits=splits, inputCol=field, outputCol=f"{field}_bin")
    bucketed_df = bucketizer.transform(df)
    # Add bin labels
    return bucketed_df


# Example usage:
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Bin Numerical Fields") \
        .getOrCreate()

    # Sample DataFrame (replace this with your DataFrame)
    data = [(1, 20), (2, 15000), (3, 50000), (4, 80000), (5, 100000)]
    df = spark.createDataFrame(data, ["id", "value"])

    # Define bin ranges and bin labels
    bin_ranges = [(10, 100), (100, 1000), (1000, 10000), (10000, 100000)]
    bin_labels = ["10-100", "101-1000", "1001-10000", "10001-100000"]

    # Convert numerical field "value" into bins with Bucketizer
    df = bin_numerical_field_with_bucketizer(df, "value", bin_ranges, bin_labels)

    # Show the results
    df.show()
