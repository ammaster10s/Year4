#@title Creation Time ALIGN
from pyspark.sql import functions as F

# find min creation time per device
device_min_df = (
    phone_ac_df.groupBy("Device")
    .agg(F.min("Creation_Time").alias("min_ct"))
)

# find the Nexus4 reference (min creation time of nexus4)
nexus_ref = (
    device_min_df
    .filter(F.col("Device").like("nexus4%"))
    .select(F.min("min_ct"))
    .collect()[0][0]
)

print(" Nexus4 baseline Creation_Time:", nexus_ref)

# Step 3: shift all devices so they start at the same baseline
aligned_df = (
    phone_ac_df.join(device_min_df, on="Device", how="left")
    .withColumn("Creation_Time_Aligned",
                F.col("Creation_Time") - F.col("min_ct") + F.lit(nexus_ref))
    .drop("min_ct")
)

summary_df = (
    aligned_df.groupBy("Model")
    .agg(F.min("Creation_Time_Aligned").alias("min_ct"),
         F.max("Creation_Time_Aligned").alias("max_ct"))
)

@F.udf("string")
def ns_to_datetime_str(ns):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(ns)/1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

summary_df = (
    summary_df
    .withColumn("min_time", ns_to_datetime_str("min_ct"))
    .withColumn("max_time", ns_to_datetime_str("max_ct"))
)
summary_df.show(truncate=False)

#@title Data Exploration
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sort by aligned time
sorted_df = aligned_df.orderBy("Creation_Time_Aligned")

# Get model list
models = [r["Model"] for r in sorted_df.select("Model").distinct().collect()]
print(f"📦 Found models: {models}")

# === MAIN LOOP over models ===
for model in models:
    print(f"\n🧩 ===== Model: {model} =====")

    df_model = sorted_df.filter(F.col("Model") == model)

    # ---- 2.1 List Unit ----
    unit_list = [r["Device"] for r in df_model.select("Device").distinct().collect()]
    print(f"📱 Devices under {model}: {unit_list}")

    # ---- 2.2 Min, Max Creation Time ----
    time_summary = (
        df_model.groupBy("User", "Device")
        .agg(
            F.min("Creation_Time_Aligned").alias("min_ct"),
            F.max("Creation_Time_Aligned").alias("max_ct")
        )
    )
    time_summary.show(truncate=False)

    # ---- 2.3 Count (Total Rows per User, gt) ----
    count_summary = (
        df_model.groupBy("User", "gt")
        .count()
        .orderBy("User", "gt")
    )
    print("Total rows per User × gt:")
    count_summary.show(100, truncate=False)

    # ---- 2.4 Min, Max, Avg, Std per User × gt ----
    xyz_stats = (
        df_model.groupBy("User", "gt")
        .agg(
            F.min("x").alias("min_x"),
            F.max("x").alias("max_x"),
            F.avg("x").alias("avg_x"),
            F.stddev("x").alias("std_x"),
            F.min("y").alias("min_y"),
            F.max("y").alias("max_y"),
            F.avg("y").alias("avg_y"),
            F.stddev("y").alias("std_y"),
            F.min("z").alias("min_z"),
            F.max("z").alias("max_z"),
            F.avg("z").alias("avg_z"),
            F.stddev("z").alias("std_z")
        )
        .orderBy("User", "gt")
    )
    xyz_stats.show(truncate=False)

    # ---- 2.5 Plot per model × device × user ----
    users = [r["User"] for r in df_model.select("User").distinct().collect()]
    devices = [r["Device"] for r in df_model.select("Device").distinct().collect()]

    for user in users:
        for device in devices:
            print(f"\n📈 Plotting: {model} | {device} | User {user}")

            user_device_df = (
                df_model
                .filter((F.col("User") == user) & (F.col("Device") == device))
                .orderBy("Creation_Time_Aligned")
            )

            if user_device_df.count() == 0:
                print(f"⚠️ No data for {model}, {device}, user {user}. Skipping.")
                continue

            # Convert to Pandas
            pd_df = (
                user_device_df
                .withColumn("Magnitude", (F.col("x")**2 + F.col("y")**2 + F.col("z")**2)**0.5)
                .select("Creation_Time_Aligned", "Magnitude", "gt")
                .toPandas()
            )

            # Convert aligned nanoseconds → datetime
            pd_df["Datetime"] = pd.to_datetime(
                pd_df["Creation_Time_Aligned"].astype(float) / 1_000_000_000, unit="s"
            )

            if len(pd_df) < 5:
                print(f"⚠️ Too few samples for {model}, {device}, user {user}")
                continue

            # Plot
            plt.figure(figsize=(12, 6))
            sns.scatterplot(
                x="Datetime", y="Magnitude",
                hue="gt", data=pd_df,
                s=3, alpha=0.4, palette="tab10"
            )
            plt.title(f"{model} | {device} | User {user} — Magnitude vs Time")
            plt.xlabel("Creation Time (Aligned)")
            plt.ylabel("Magnitude")
            plt.ylim(0, 25)
            plt.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

print("\n✅ All models processed and sorted by Creation_Time.")

from pyspark.sql import functions as F

# find min creation time per device
device_min_df = (
    phone_ac_df.groupBy("Device")
    .agg(F.min("Creation_Time").alias("min_ct"))
)

# find the Nexus4 reference (min creation time of nexus4)
nexus_ref = (
    device_min_df
    .filter(F.col("Device").like("nexus4%"))
    .select(F.min("min_ct"))
    .collect()[0][0]
)

print(" Nexus4 baseline Creation_Time:", nexus_ref)

# Step 3: shift all devices so they start at the same baseline
aligned_df = (
    phone_ac_df.join(device_min_df, on="Device", how="left")
    .withColumn("Creation_Time_Aligned",
                F.col("Creation_Time") - F.col("min_ct") + F.lit(nexus_ref))
    .drop("min_ct")
)

summary_df = (
    aligned_df.groupBy("Model")
    .agg(F.min("Creation_Time_Aligned").alias("min_ct"),
         F.max("Creation_Time_Aligned").alias("max_ct"))
)

@F.udf("string")
def ns_to_datetime_str(ns):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(ns)/1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

summary_df = (
    summary_df
    .withColumn("min_time", ns_to_datetime_str("min_ct"))
    .withColumn("max_time", ns_to_datetime_str("max_ct"))
)
summary_df.show(truncate=False)

def find_null_intervals_df(df):
    is_null_string = (df['gt'] == "null")
    start_of_null_block = is_null_string & (~is_null_string.shift(1).fillna(False))
    end_of_null_block = is_null_string & (~is_null_string.shift(-1).fillna(False))

    start_times = df.loc[start_of_null_block, 'Datetime']
    end_times = df.loc[end_of_null_block, 'Datetime']

    null_intervals = pd.DataFrame({
        'Start_Time': start_times.reset_index(drop=True),
        'End_Time': end_times.reset_index(drop=True)
    })

    timedelta_duration = null_intervals['End_Time'] - null_intervals['Start_Time']
    null_intervals['Duration_Minutes'] = timedelta_duration.dt.total_seconds() / 60

    return null_intervals


def perform_spark_analysis(phone_ac_df, target_model="s3", main_date_ns=1424649600000000000):
    """
    Performs initial filtering, caching, and summary aggregations using Spark-like operations.

    Args:
        phone_ac_df (Spark DataFrame): The raw input data.
        target_model (str): Model to filter (e.g., "nexus4").
        main_date_ns (int): Nanosecond timestamp for filtering (e.g., start of 2015-02-23).

    Returns:
        Spark DataFrame: The filtered and prepared DataFrame for further analysis.
    """

    # 1. Filter, Cache, and Check Dates (Optimization)
    nexus_phone_ac_df = phone_ac_df.filter(phone_ac_df["Model"] == target_model)
    nexus_phone_ac_df.cache() # Cache is essential for repeated use


    nexus_phone_ac_df.groupBy("Device","User","gt").count().orderBy("User","gt","Device").show(100,truncate=False)
    print("Main Date:", ns_to_datetime(main_date_ns))
    nexus_phone_ac_df.filter(col("Creation_Time").cast(LongType()) <= main_date_ns).count()
    filtered_ac_df = nexus_phone_ac_df.filter(col("Creation_Time").cast(LongType()) > main_date_ns)


    summary_df = (
        filtered_ac_df
        .groupBy("User", "Device")
        .agg(
            F.min("Creation_Time").alias("min_time_ns"),
            F.max("Creation_Time").alias("max_time_ns")
        )
    )

    ns_to_datetime_udf = F.udf(lambda ns: ns_to_datetime(ns).strftime("%Y-%m-%d %H:%M:%S"))
    summary_df = (
        summary_df
        .withColumn("min_time", ns_to_datetime_udf("min_time_ns"))
        .withColumn("max_time", ns_to_datetime_udf("max_time_ns"))
    )


    total_duration_df = summary_df.withColumn("Duration", round((F.unix_timestamp("max_time") - F.unix_timestamp("min_time")) / 60, 1)).select("User","Duration").groupBy("User").agg(sum("Duration"))

    overlap_df = summary_df.groupBy("User").agg(
        F.max("min_time").alias("nexus_2"),
        F.min("max_time").alias("nexus_1"))\
        .withColumn("Overlapped", round( (F.unix_timestamp("nexus_1") - F.unix_timestamp("nexus_2"))/60,1))\
        .withColumn("Flag", F.when(F.col("Overlapped")> 0, F.lit(1)).otherwise(F.lit(0)))


    return filtered_ac_df


def analyze_user_gt_transitions(filtered_ac_df, window_spec):
    """
    Performs window function analysis (lag/lead) to determine activity transitions
    around 'null' values for every user.
    """

    # Find all unique users in the filtered dataset
    users = [row['User'] for row in filtered_ac_df.select("User").distinct().collect()]

    transition_summary = {}

    for user in users:
        user_df = filtered_ac_df.filter(col("User") == user).orderBy("Creation_Time")

        # Calculate lag (activity BEFORE null)
        lag_df = (
            user_df
            .withColumn("lag_1_gt", F.lag("gt").over(window_spec))
            .filter(col("gt") == "null")
            .groupBy("lag_1_gt").count().orderBy(F.desc("count"))
        )

        # Calculate lead (activity AFTER null)
        lead_df = (
            user_df
            .withColumn("lead_1_gt", F.lead("gt").over(window_spec))
            .filter(col("gt") == "null")
            .groupBy("lead_1_gt").count().orderBy(F.desc("count"))
        )

        transition_summary[user] = {
            "before_null": lag_df.toPandas(),
            "after_null": lead_df.toPandas()
        }

    return transition_summary


def prepare_data_for_pandas_analysis(spark_user_df):
    """
    Converts a Spark DataFrame for a single user to a prepared Pandas DataFrame.
    Calculates Magnitude and Datetime. Optimized for the final plotting step.
    """

    user_df = spark_user_df.withColumn("Magnitude", (F.col("x")**2 + F.col("y")**2 + F.col("z")**2)**0.5)
    pd_a_df = user_df.select("Creation_Time", "Magnitude", "gt").toPandas()

    pd_a_df['Creation_Time'] = pd_a_df['Creation_Time'].astype(float)
    pd_a_df['Creation_Time_us'] = pd_a_df['Creation_Time'] / 1000
    pd_a_df['Datetime'] = pd.to_datetime(pd_a_df['Creation_Time_us'], unit='us')

    return pd_a_df

def find_null_intervals_df(df):
    """
    Finds and calculates the duration in minutes for all contiguous "null" intervals.

    Now returns only the count of intervals.
    """
    is_null_string = (df['gt'] == "null")
    start_of_null_block = is_null_string & (~is_null_string.shift(1).fillna(False))
    end_of_null_block = is_null_string & (~is_null_string.shift(-1).fillna(False))

    # The number of start times equals the total number of intervals
    total_interval_count = start_of_null_block.sum()

    # Returning the count instead of the DataFrame
    return total_interval_count


def generate_time_series_plot(pd_df, color_map, title_suffix=""):
    """
    Generates the final optimized scatter plot for the entire time series data,
    with a fixed Y-axis range (0-25).

    Args:
        pd_df (pd.DataFrame): The single user's data.
        color_map (dict): The predefined map for consistent coloring across all plots.
        title_suffix (str): Suffix for the plot title.
    """

    # 1. Define Fixed Color Palette (Now passed in as an argument: color_map)
    # This ensures global consistency across all users.

    # 2. Plotting
    plt.figure(figsize=(14, 7))

    sns.scatterplot(
        x='Datetime',
        y='Magnitude',
        hue='gt',
        data=pd_df,
        s=1,         # Small dot size for high density
        alpha=0.3,   # Transparency to show density
        palette=color_map # Use the externally defined, fixed color map
    )

    # Set the Y-axis limit between 0 and 25
    plt.ylim(0, 25)

    plt.title(f'Overall Magnitude Trend vs. Time, Colored by Ground Truth (gt) {title_suffix}')
    plt.xlabel('Creation Time (Datetime)')
    plt.ylabel('Magnitude')
    plt.legend(title='Ground Truth (gt)', markerscale=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    from pyspark.sql import functions as F

pairs = (
    filtered_ac_df
    .select("User", "Device")
    .distinct()
    .orderBy("User", "Device")
    .collect()
)

ac_feature_dfs = {}

for row in pairs:
    user, device = row["User"], row["Device"]

    # filter accelerometer data for this user-device
    ac_sub = filtered_ac_df.filter(
        (F.col("User") == user) & (F.col("Device") == device)
    )

    # skip if no rows
    if ac_sub.isEmpty():
        continue

    # compute feature aggregates
    features = (
        ac_sub.groupBy("User", "gt")
        .agg(
            F.min("x").alias("min_x"), F.max("x").alias("max_x"),
            F.min("y").alias("min_y"), F.max("y").alias("max_y"),
            F.min("z").alias("min_z"), F.max("z").alias("max_z"),
            F.mean("x").alias("mean_x"), F.mean("y").alias("mean_y"), F.mean("z").alias("mean_z"),
            F.stddev("x").alias("std_x"), F.stddev("y").alias("std_y"), F.stddev("z").alias("std_z")
        )
        .withColumn("Device", F.lit(device))
    )

    # store result in dictionary
    ac_feature_dfs[(user, device)] = features

    print(f"Aggregated features for User={user}, Device={device}")


#@title Helper Func
@F.udf(T.DoubleType())
def energy_udf(values):
    if not values:
        return None
    arr = np.array([float(v) for v in values if v is not None])
    return float(np.mean(arr ** 2)) if len(arr) > 0 else None

@F.udf(T.DoubleType())
def rms_udf(values):
    if not values:
        return None
    arr = np.array([float(v) for v in values if v is not None])
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else None

@F.udf(T.DoubleType())
def magnitude_mean_udf(xs, ys, zs):
    arr_x = np.array([float(v) for v in xs if v is not None])
    arr_y = np.array([float(v) for v in ys if v is not None])
    arr_z = np.array([float(v) for v in zs if v is not None])
    arr = np.sqrt(arr_x**2 + arr_y**2 + arr_z**2)
    return float(np.mean(arr)) if len(arr) > 0 else None

@F.udf(T.DoubleType())
def magnitude_std_udf(xs, ys, zs):
    arr_x = np.array([float(v) for v in xs if v is not None])
    arr_y = np.array([float(v) for v in ys if v is not None])
    arr_z = np.array([float(v) for v in zs if v is not None])
    arr = np.sqrt(arr_x**2 + arr_y**2 + arr_z**2)
    return float(np.std(arr)) if len(arr) > 0 else None

@F.udf(T.StringType())
def majority_vote_udf(gt_list):
    if not gt_list:
        return None
    vals, counts = np.unique([str(v) for v in gt_list if v is not None], return_counts=True)
    return str(vals[np.argmax(counts)])

# ====== 2. Extract features per window ======
def extract_features(df, prefix):
    df = df.cache()  # cache for repeated computations
    agg_exprs = []
    for axis in ["x", "y", "z"]:
        col = F.col(axis)
        agg_exprs += [
            F.mean(col).alias(f"{prefix}_{axis}_mean"),
            F.stddev(col).alias(f"{prefix}_{axis}_std"),
            F.min(col).alias(f"{prefix}_{axis}_min"),
            F.max(col).alias(f"{prefix}_{axis}_max"),
            F.expr(f"percentile({axis}, 0.5)").alias(f"{prefix}_{axis}_median"),
            F.expr(f"percentile({axis}, 0.75) - percentile({axis}, 0.25)").alias(f"{prefix}_{axis}_iqr"),
            F.skewness(col).alias(f"{prefix}_{axis}_skew"),
            F.kurtosis(col).alias(f"{prefix}_{axis}_kurt"),
            F.collect_list(col).alias(f"{axis}_list")
        ]
    grouped = df.groupBy("User", "window_id").agg(*agg_exprs)

    result = (
        grouped
        .withColumn(f"{prefix}_x_energy", energy_udf("x_list"))
        .withColumn(f"{prefix}_y_energy", energy_udf("y_list"))
        .withColumn(f"{prefix}_z_energy", energy_udf("z_list"))
        .withColumn(f"{prefix}_x_rms", rms_udf("x_list"))
        .withColumn(f"{prefix}_y_rms", rms_udf("y_list"))
        .withColumn(f"{prefix}_z_rms", rms_udf("z_list"))
        .withColumn(f"{prefix}_mag_mean", magnitude_mean_udf("x_list", "y_list", "z_list"))
        .withColumn(f"{prefix}_mag_std", magnitude_std_udf("x_list", "y_list", "z_list"))
        .drop("x_list", "y_list", "z_list")
    )
    df.unpersist()  # release memory
    return result

# ====== 3. Aggregate GT per window ======
def aggregate_gt(df):
    df = df.cache()
    result = df.groupBy("User", "window_id").agg(
        F.collect_list("gt").alias("gt_list")
    ).withColumn("gt_majority", majority_vote_udf("gt_list")).drop("gt_list")
    df.unpersist()
    return result

# ====== 4. Build feature DF using AC time-based windows ======
def build_feature_df_ac_time_window(ac_df, gr_df, prefix_ac="ac", prefix_gr="gr", window_size_sec=3):
    # Cast numeric columns
    for col in ["x", "y", "z"]:
        ac_df = ac_df.withColumn(col, F.col(col).cast("double"))
        gr_df = gr_df.withColumn(col, F.col(col).cast("double"))

    # Sort AC
    ac_df = ac_df.orderBy("User", "Creation_Time").cache()

    # Compute time-based windows
    window_size_ns = window_size_sec * 1_000_000_000
    w_spec = Window.partitionBy("User").orderBy("Creation_Time")
    ac_df = ac_df.withColumn(
        "time_offset", F.col("Creation_Time") - F.first("Creation_Time").over(w_spec)
    ).withColumn(
        "window_id", (F.col("time_offset") / window_size_ns).cast("int")
    ).drop("time_offset")

    # AC features
    ac_feat = extract_features(ac_df, prefix_ac)

    # AC window start/end
    ac_time = ac_df.groupBy("User", "window_id").agg(
        F.min("Creation_Time").alias("window_start_time"),
        F.max("Creation_Time").alias("window_end_time")
    )
    ac_feat = ac_feat.join(ac_time, ["User", "window_id"], "left")

    # GT
    gt_df = aggregate_gt(ac_df)
    ac_df.unpersist()

    # GR join using AC time
    ac_time_alias = ac_time.alias("ac")
    gr_df_alias = gr_df.alias("gr")
    gr_w = gr_df_alias.join(
        ac_time_alias,
        on=[
            (gr_df_alias.User == ac_time_alias.User) &
            (gr_df_alias.Creation_Time >= ac_time_alias.window_start_time) &
            (gr_df_alias.Creation_Time <= ac_time_alias.window_end_time)
        ],
        how="inner"
    ).select(gr_df_alias["*"], ac_time_alias.window_id)
    gr_feat = extract_features(gr_w, prefix_gr)
    gr_w.unpersist()

    # Combine AC + GR + GT
    feature_df = (
        ac_feat.join(gr_feat, ["User", "window_id"], "left")
               .join(gt_df, ["User", "window_id"], "left")
    )
    return feature_df

