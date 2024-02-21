from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import *


def group_by_and_aggregate(df, group_by_cols, agg_funcs):
    """
    Group by specified columns and aggregate using the specified aggregation functions.

    Parameters:
        df (DataFrame): Input DataFrame.
        group_by_cols (list): List of column names to group by.
        agg_funcs (dict): Dictionary where keys are column names and values are aggregation functions.

    Returns:
        DataFrame: DataFrame with grouped and aggregated results.
    """
    # Apply group by and aggregation
    result_df = df.groupBy(*group_by_cols).agg(
        *[agg_func(col(col_name)).alias(col_name) for col_name, agg_func in agg_funcs.items()])

    # for col_name,item in agg_funcs.items():
    #     printf(f"checking for {col_name}")
    #     res = result_df.agg(sum(f"{col_name})")).collect()[0][0]
    #     result_df.filter(f"item > 0.02*{res}")

    return result_df


class PlayStoreInsights:
    def __init__(self, data_path):
        self.data_path = data_path
        self.spark = SparkSession.builder \
            .appName("Play Store Insights") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.sql.debug.maxToStringFields", 1000) \
            .getOrCreate()

    def getBasicAnalysis(self):
        (self.data
         .withColumn("mean_num_of_installs", round(avg(col("minInstalls"), 2)))
         .withColumn("standard_deviation_num_of_installs", stddev(col("minInstalls")))
         .withColumn("mean_inAppProductPrice", round(avg(col("inAppProductPrice"), 2)))
         .withColumn("standard_deviation_inAppProductPrice", stddev(col("inAppProductPrice")))
         .withColumn("free_count", count(when(col("free") == "true", 1)))
         .withColumn("paid_count", count(when(col("free") == "false", 1)))
         .withColumn("ad_supported_count", count(when(col("adSupported") == "true", 1)))
         .withColumn("contains_ads_count", count(when(col("containsAds") == "true", 1)))

         )

    def load_data(self):
        '''
        This function will load the data from the given path

        '''
        self.data = self.spark.read.option("header", "true").csv(self.data_path)

    def useful_columns(self):
        '''
        This function will select the useful columns from the data
        '''
        self.data = self.data.selectExpr(
            "appId",
            "developer",
            "developerWebsite",
            "free",
            "genre",
            "cast(IFNULL(REGEXP_EXTRACT(inAppProductPrice, r'(\d+)'), '0') as float) AS inAppProductPrice",
            "cast(IFNULL(REGEXP_EXTRACT(minInstalls, r'(\d+)'), '0') as int) AS minInstalls",
            "cast(ratings as int) AS ratings",
            "adSupported",
            "containsAds",
            "reviews",
            "cast(score as float) AS score",
            "summary",
            "title",
            "ParseReleasedDayYear AS releasedDayYear",
            "dateUpdated",
            "cast(ROUND(price) as int) AS price",
            "cast(ROUND(maxprice) as int) AS maxprice",
            "cast(IFNULL(REGEXP_EXTRACT(minprice, r'(\d+)'), '0') as int) AS minprice",
            "cast(ParseReleasedDayYear as date) AS newParseReleasedDayYear",

        ).withColumn("year", year(col("newParseReleasedDayYear")))

    def print_max_min_numeric_columns(self, df):
        '''

        :param self:
        :param df:
        :return:
        '''
        # Iterate over each column in the DataFrame
        for col_name in df.columns:
            # Check if the column is numeric
            if df.schema[col_name].dataType in [ByteType(), ShortType(), IntegerType(), LongType(), FloatType(),
                                                DoubleType()]:
                print(f"checking for {col_name}")
                # Calculate the maximum and minimum values for the column
                max_val = df.agg(max(col(col_name))).collect()[0][0]
                min_val = df.agg(min(col(col_name))).collect()[0][0]
                avg_val = df.agg(avg(col(col_name))).collect()[0][0]
                # Print the results
                print(f"Column: {col_name}, Max: {max_val}, Min: {min_val}", f" Avg: {avg_val:.2f}")

    def write_csv_with_column_names_as_pairs(self, df, file_path):

        # Open file for writing
        with open("../output/"+file_path, "w") as f:
            for row in df.collect():
                # Extract column names and values and pair them together
                pairs = [f"{column}={row[column]}" for column in df.columns]

                f.write(",".join(pairs) + "\n")

    def clean_data(self):
        self.data = self.data.dropna().dropDuplicates()

    def bin_numerical_field(self, df, field, bin_ranges, bin_labels):
        expr = [
            when((col(field) >= start) & (col(field) <= end), label)
            for (start, end), label in zip(bin_ranges, bin_labels)
        ]
        df = df.withColumn(f"{field}_bin", *expr)
        return df

    def bin_numerical_field_with_bucketizer(self, df, field, bin_ranges, bin_labels):
        # Define splits for bucketizer
        splits = [float('-inf')] + [end for (start, end) in bin_ranges] + [float('inf')]
        # Apply bucketizer
        bucketizer = Bucketizer(splits=splits, inputCol=field, outputCol=f"{field}_bin")
        bucketed_df = bucketizer.transform(df)
        # Map bin indices to labels
        for i, label in enumerate(bin_labels):
            bucketed_df = bucketed_df.withColumn(f"{field}_bin",
                                                 expr(
                                                     f"CASE WHEN `{field}_bin` = {i} THEN '{label}' ELSE `{field}_bin` END"))
        return bucketed_df

    def map_bins(self):
        in_app_Product_Price_bin_ranges = [(0, 1), (2, 10), (11, 50), (51, 200), (201, 1000)]
        in_app_Product_Price_bin_labels = ["0-1", "2-10", "11-50", "51-200", "201-1000"]
        self.data = self.bin_numerical_field_with_bucketizer(self.data, "inAppProductPrice",
                                                             in_app_Product_Price_bin_ranges,
                                                             in_app_Product_Price_bin_labels)

        min_installs_bin_ranges = [(0, 10000), (10001, 100000), (100001, 1000000), (1000001, 10000000),
                                   (10000001, 1000000000)]
        min_installs_bin_labels = ["0-10000", "10001-100000", "100001-1000000", "1000001-10000000", ]
        self.data = self.bin_numerical_field_with_bucketizer(self.data, "minInstalls",
                                                             min_installs_bin_ranges,
                                                             min_installs_bin_labels)

        No_of_ratings_bin_ranges = [(0, 100), (101, 10000), (10001, 100000), (100001, 10000000)]
        No_of_ratings_bin_labels = ["0-100", "101-10000", "10001-100000", "100001-10000000"]
        self.data = self.bin_numerical_field_with_bucketizer(self.data, "ratings",
                                                             No_of_ratings_bin_ranges,
                                                             No_of_ratings_bin_labels)

        releaseDate_bin_ranges = [(1900, 1999), (2000, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025)]
        releaseDate_bin_labels = ["1900-1999", "2000-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
        self.data = self.bin_numerical_field_with_bucketizer(self.data, "year",
                                                             releaseDate_bin_ranges,
                                                             releaseDate_bin_labels)

    def get_agg_data(self, df):
        # Group by inAppProductPrice
        # 1. Free vs Paid apps installation distribution
        group_by_cols = ["free", "minInstalls_bin"]

        # Define aggregation functions
        agg_funcs = {
            "appId": count
        }

        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "free_vs_paid_installation_distribution.csv")

        # 2. average price of in-app products for both free and paid apps.

        group_by_cols = ["free"]
        agg_funcs = {
            "inAppProductPrice": avg
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "average_price_of_in_app_products.csv")

        # 3. Number of apps in each genre
        group_by_cols = ["genre"]
        agg_funcs = {
            "appId": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "number_of_apps_in_each_genre.csv")

        # 4. Genre VS Total number of app downloads
        group_by_cols = ["genre", "minInstalls_bin"]
        agg_funcs = {
            "appId": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "genre_vs_total_number_of_app_downloads.csv")

        # 5. Genre VS Average Review Score

        group_by_cols = ["genre"]
        agg_funcs = {
            "score": avg
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "genre_vs_average_review_score.csv")

        # 6. Identify the most prevalent genre in each year
        group_by_cols = ["year_bin", "genre"]
        agg_funcs = {
            "appId": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "most_prevalent_genre_in_each_year.csv")
        # 7. Identify the top 5 developers with the highest number of apps

        group_by_cols = ["developer"]
        agg_funcs = {
            "appId": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "top_5_developers_with_highest_number_of_apps.csv")

        # 8. These are the most consistent developers, they have achived over 10,000 downloads while maintaing perfect 5 star reviews:

        group_by_cols = ["developer"]
        agg_funcs = {
            "minInstalls": sum,
            "ratings": sum
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs).filter(col("minInstalls") > 10000).filter(col("ratings") > 10000),
                                                  "most_consistent_developers.csv")


        #9. Effect of In - App advertisment On Number of Installs
        group_by_cols = ["adSupported"]
        agg_funcs = {
            "minInstalls": sum
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "effect_of_in_app_advertisment_on_number_of_installs.csv")

        #10. Effect on the Price of in-app products of free apps
        group_by_cols = ["adSupported", "free"]
        agg_funcs = {
            "inAppProductPrice_bin": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "effect_on_the_price_of_in_app_products_of_free_apps.csv")
        #11. Top 5 genres by average app installs for ad-supported apps
        group_by_cols = ["adSupported", "genre"]
        agg_funcs = {
            "minInstalls": avg
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs).filter(col("adSupported") == "true"),
                                                  "top_5_genres_by_average_app_installs_for_ad_supported_apps.csv")
        #12. Disturbution of Ad - supported Apps across Genres
        group_by_cols = ["adSupported", "genre"]
        agg_funcs = {
            "appId": count
        }
        self.write_csv_with_column_names_as_pairs(group_by_and_aggregate(df, group_by_cols, agg_funcs),
                                                  "disturbution_of_ad_supported_apps_across_genres.csv")

    def basic_analysis(self):
        self.total_apps = self.data.count()
        # self.avg_rating = self.data.agg({"ratings": "avg"}).collect()[0][0]
        self.max_reviews = self.data.agg({"reviews": "max"}).collect()[0][0]
        self.most_downloaded_app = self.data.orderBy(col("minInstalls").desc()).first()["appId"]

    def display_insights(self):
        print(f"Total number of apps: {self.total_apps}")
        # print(f"Average rating of apps: {self.avg_rating:.2f}")
        print(f"Maximum number of reviews: {self.max_reviews}")
        print(f"The most downloaded app: {self.most_downloaded_app}")

    def display_schema_and_sample_data(self):
        print("Schema:")
        self.data.printSchema()
        print("Sample Data:")
        self.data.show(5)

    def run_analysis(self):
        self.load_data()
        self.useful_columns()
        self.clean_data()
        #self.print_max_min_numeric_columns(self.data)
        self.map_bins()
        self.get_agg_data(self.data)
        # self.display_schema_and_sample_data()
        # self.basic_analysis()
        # self.display_insights()
        self.spark.stop()


if __name__ == "__main__":
    data_path = "../data/google-play-dataset-by-tapivedotcom.csv"
    insights_analyzer = PlayStoreInsights(data_path)
    insights_analyzer.run_analysis()

