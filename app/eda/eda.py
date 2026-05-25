from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


base_dir = Path(__file__).resolve().parent
input_file_path = base_dir.parent / "database"
output_dir = base_dir / "outputs"
output_dir.mkdir(exist_ok=True)

bts_october_2015_url = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2015_10.zip"
)
bts_zip_path = input_file_path / "external_bts_2015_10.zip"
airport_id_mapping_path = input_file_path / "airport_id_to_iata_2015_10.csv"

if not bts_zip_path.exists():
    urlretrieve(bts_october_2015_url, bts_zip_path)

with ZipFile(bts_zip_path) as bts_zip:
    bts_csv_name = [name for name in bts_zip.namelist() if name.endswith(".csv")][0]
    with bts_zip.open(bts_csv_name) as bts_csv:
        df_bts_airports = pd.read_csv(
            bts_csv,
            usecols=["OriginAirportID", "Origin", "DestAirportID", "Dest"],
            dtype={
                "OriginAirportID": "string",
                "Origin": "string",
                "DestAirportID": "string",
                "Dest": "string",
            },
        )

origin_airport_id_mapping = df_bts_airports[["OriginAirportID", "Origin"]].rename(
    columns={"OriginAirportID": "AIRPORT_ID", "Origin": "IATA_CODE"}
)
destination_airport_id_mapping = df_bts_airports[["DestAirportID", "Dest"]].rename(
    columns={"DestAirportID": "AIRPORT_ID", "Dest": "IATA_CODE"}
)
airport_id_mapping = (
    pd.concat([origin_airport_id_mapping, destination_airport_id_mapping])
    .drop_duplicates()
    .sort_values("AIRPORT_ID")
    .reset_index(drop=True)
)
airport_id_mapping.to_csv(airport_id_mapping_path, index=False)

airport_id_to_iata = dict(
    zip(airport_id_mapping["AIRPORT_ID"], airport_id_mapping["IATA_CODE"])
)

df_airlines = pd.read_csv(input_file_path / "airlines.csv")
df_airports = pd.read_csv(input_file_path / "airports.csv")
df_flights = pd.read_csv(
    input_file_path / "flights.csv",
    dtype={
        "AIRLINE": "string",
        "ORIGIN_AIRPORT": "string",
        "DESTINATION_AIRPORT": "string",
    },
    low_memory=False,
)

df_flights["ORIGIN_AIRPORT_ORIGINAL"] = df_flights["ORIGIN_AIRPORT"]
df_flights["DESTINATION_AIRPORT_ORIGINAL"] = df_flights["DESTINATION_AIRPORT"]
df_flights["ORIGIN_AIRPORT_WAS_NUMERIC"] = df_flights["ORIGIN_AIRPORT"].str.fullmatch(
    r"\d+"
)
df_flights["DESTINATION_AIRPORT_WAS_NUMERIC"] = df_flights[
    "DESTINATION_AIRPORT"
].str.fullmatch(r"\d+")

df_flights["ORIGIN_AIRPORT"] = df_flights["ORIGIN_AIRPORT"].replace(airport_id_to_iata)
df_flights["DESTINATION_AIRPORT"] = df_flights["DESTINATION_AIRPORT"].replace(
    airport_id_to_iata
)

airlines_lookup = df_airlines.rename(
    columns={
        "IATA_CODE": "AIRLINE",
        "AIRLINE": "AIRLINE_NAME",
    }
)

origin_airports_lookup = df_airports.rename(
    columns={
        "IATA_CODE": "ORIGIN_AIRPORT",
        "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "CITY": "ORIGIN_CITY",
        "STATE": "ORIGIN_STATE",
        "COUNTRY": "ORIGIN_COUNTRY",
        "LATITUDE": "ORIGIN_LATITUDE",
        "LONGITUDE": "ORIGIN_LONGITUDE",
    }
)

destination_airports_lookup = df_airports.rename(
    columns={
        "IATA_CODE": "DESTINATION_AIRPORT",
        "AIRPORT": "DESTINATION_AIRPORT_NAME",
        "CITY": "DESTINATION_CITY",
        "STATE": "DESTINATION_STATE",
        "COUNTRY": "DESTINATION_COUNTRY",
        "LATITUDE": "DESTINATION_LATITUDE",
        "LONGITUDE": "DESTINATION_LONGITUDE",
    }
)

df_join = (
    df_flights.merge(airlines_lookup, on="AIRLINE", how="left", validate="many_to_one")
    .merge(origin_airports_lookup, on="ORIGIN_AIRPORT", how="left", validate="many_to_one")
    .merge(destination_airports_lookup, on="DESTINATION_AIRPORT", how="left", validate="many_to_one")
)

df_join["FLIGHT_DATE"] = pd.to_datetime(df_join[["YEAR", "MONTH", "DAY"]])
df_join["SCHEDULED_DEPARTURE_HOUR"] = df_join["SCHEDULED_DEPARTURE"] // 100
df_join["SCHEDULED_ARRIVAL_HOUR"] = df_join["SCHEDULED_ARRIVAL"] // 100
df_join["ROUTE"] = df_join["ORIGIN_AIRPORT"] + "-" + df_join["DESTINATION_AIRPORT"]
df_join["IS_WEEKEND"] = df_join["DAY_OF_WEEK"].isin([6, 7])

df_completed_raw = df_join[
    (df_join["CANCELLED"] == 0)
    & (df_join["DIVERTED"] == 0)
    & df_join["ARRIVAL_DELAY"].notna()
].copy()

df_completed = df_completed_raw.drop(
    columns=["CANCELLED", "DIVERTED", "CANCELLATION_REASON"]
).copy()

delay_reason_cols = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
]
df_completed[delay_reason_cols] = df_completed[delay_reason_cols].fillna(0)
df_completed["TAIL_NUMBER"] = df_completed["TAIL_NUMBER"].fillna("UNKNOWN")
df_completed["IS_DELAYED_15"] = df_completed["ARRIVAL_DELAY"] >= 15

column_profile = pd.DataFrame(
    {
        "column": df_completed_raw.columns,
        "dtype": df_completed_raw.dtypes.astype(str).values,
        "missing_count": df_completed_raw.isna().sum().values,
        "missing_pct": (df_completed_raw.isna().mean() * 100).round(2).values,
        "unique_count": df_completed_raw.nunique(dropna=True).values,
        "unique_pct": (
            df_completed_raw.nunique(dropna=True) / len(df_completed_raw) * 100
        ).round(4).values,
    }
).sort_values(["missing_pct", "unique_count"], ascending=[False, False])
column_profile.to_csv(output_dir / "completed_column_profile_before_treatment.csv", index=False)

clean_column_profile = pd.DataFrame(
    {
        "column": df_completed.columns,
        "dtype": df_completed.dtypes.astype(str).values,
        "missing_count": df_completed.isna().sum().values,
        "missing_pct": (df_completed.isna().mean() * 100).round(2).values,
        "unique_count": df_completed.nunique(dropna=True).values,
        "unique_pct": (df_completed.nunique(dropna=True) / len(df_completed) * 100)
        .round(4)
        .values,
    }
).sort_values(["missing_pct", "unique_count"], ascending=[False, False])
clean_column_profile.to_csv(output_dir / "completed_column_profile_after_treatment.csv", index=False)

numeric_cols = df_completed.select_dtypes(include="number").columns
numeric_profile = df_completed[numeric_cols].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
).T
numeric_profile["iqr"] = numeric_profile["75%"] - numeric_profile["25%"]
numeric_profile["lower_iqr_limit"] = numeric_profile["25%"] - 1.5 * numeric_profile["iqr"]
numeric_profile["upper_iqr_limit"] = numeric_profile["75%"] + 1.5 * numeric_profile["iqr"]
numeric_profile["outlier_count_iqr"] = [
    (
        (df_completed[col] < numeric_profile.loc[col, "lower_iqr_limit"])
        | (df_completed[col] > numeric_profile.loc[col, "upper_iqr_limit"])
    ).sum()
    for col in numeric_profile.index
]
numeric_profile["outlier_pct_iqr"] = (
    numeric_profile["outlier_count_iqr"] / len(df_completed) * 100
).round(2)
numeric_profile.to_csv(output_dir / "completed_numeric_profile.csv", index_label="column")

categorical_cols = df_completed.select_dtypes(include=["object", "string", "boolean"]).columns
categorical_profile = pd.DataFrame(
    {
        "column": categorical_cols,
        "unique_count": df_completed[categorical_cols].nunique(dropna=True).values,
        "top_value": [
            df_completed[col].value_counts(dropna=False).index[0] for col in categorical_cols
        ],
        "top_value_count": [
            df_completed[col].value_counts(dropna=False).iloc[0] for col in categorical_cols
        ],
    }
)
categorical_profile["top_value_pct"] = (
    categorical_profile["top_value_count"] / len(df_completed) * 100
).round(2)
categorical_profile.to_csv(output_dir / "completed_categorical_profile.csv", index=False)

treatment_plan = pd.DataFrame(
    [
        {
            "column_group": "Rows",
            "columns": "CANCELLED, DIVERTED, ARRIVAL_DELAY",
            "diagnosis": "Project focuses on flight delay, so use completed flights only.",
            "suggested_treatment": "Keep rows with CANCELLED = 0, DIVERTED = 0 and ARRIVAL_DELAY not null.",
        },
        {
            "column_group": "Cancellation",
            "columns": "CANCELLED, DIVERTED, CANCELLATION_REASON",
            "diagnosis": "Constant or not relevant after filtering completed flights.",
            "suggested_treatment": "Drop from the completed analytical dataset.",
        },
        {
            "column_group": "Delay causes",
            "columns": ", ".join(delay_reason_cols),
            "diagnosis": "Null means no reported delay cause in completed flights.",
            "suggested_treatment": "Fill null with 0 for EDA. Avoid as pre-flight model features.",
        },
        {
            "column_group": "Aircraft",
            "columns": "TAIL_NUMBER",
            "diagnosis": "Small amount of missing aircraft identifiers.",
            "suggested_treatment": "Fill null with UNKNOWN if used as category.",
        },
        {
            "column_group": "Airport enrichments",
            "columns": "ORIGIN_*, DESTINATION_*",
            "diagnosis": "Missing joins indicate airport codes not present in airports.csv.",
            "suggested_treatment": "Use 3-letter IATA records for airport/city/state/geographic analyses.",
        },
        {
            "column_group": "Extreme delays",
            "columns": "DEPARTURE_DELAY, ARRIVAL_DELAY",
            "diagnosis": "Extreme values are operationally meaningful, not necessarily errors.",
            "suggested_treatment": "Keep for EDA, report median and percentiles, optionally clip for regression.",
        },
        {
            "column_group": "Operational actuals",
            "columns": "DEPARTURE_TIME, TAXI_OUT, WHEELS_OFF, WHEELS_ON, TAXI_IN, ARRIVAL_TIME",
            "diagnosis": "Known only after operation starts or ends.",
            "suggested_treatment": "Useful for explanatory EDA. Avoid in pre-flight prediction models.",
        },
    ]
)
treatment_plan.to_csv(output_dir / "completed_treatment_plan.csv", index=False)

missing_to_plot = column_profile[column_profile["missing_pct"] > 0].copy()
plt.figure(figsize=(12, 7))
plt.barh(missing_to_plot["column"], missing_to_plot["missing_pct"])
plt.xlabel("Missing values (%)")
plt.title("Completed flights: missing values before treatment")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / "completed_missing_values.png", dpi=150)
plt.close()

plot_numeric_cols = [
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",
    "TAXI_OUT",
    "TAXI_IN",
    "AIR_TIME",
    "ELAPSED_TIME",
    "DISTANCE",
]
sample = df_completed.sample(n=min(100_000, len(df_completed)), random_state=42)

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.ravel()
for idx, col in enumerate(plot_numeric_cols):
    values = sample[col].dropna()
    p01, p99 = values.quantile([0.01, 0.99])
    axes[idx].hist(values.clip(p01, p99), bins=50)
    axes[idx].set_title(f"{col} clipped to P01-P99")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Frequency")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(output_dir / "completed_numeric_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 7))
df_completed.boxplot(column=plot_numeric_cols, rot=45, showfliers=False)
plt.title("Completed flights: numeric variables without outlier points")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(output_dir / "completed_numeric_boxplots.png", dpi=150)
plt.close()

airport_delay = (
    df_completed.dropna(subset=["ORIGIN_LATITUDE", "ORIGIN_LONGITUDE"])
    .groupby(
        [
            "ORIGIN_AIRPORT",
            "ORIGIN_AIRPORT_NAME",
            "ORIGIN_LATITUDE",
            "ORIGIN_LONGITUDE",
        ],
        as_index=False,
    )
    .agg(
        flights=("ARRIVAL_DELAY", "size"),
        avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        delayed_15_pct=("IS_DELAYED_15", "mean"),
    )
)
airport_delay["delayed_15_pct"] = (airport_delay["delayed_15_pct"] * 100).round(2)
airport_delay.to_csv(output_dir / "completed_origin_airport_delay_profile.csv", index=False)

airport_delay_plot = airport_delay[airport_delay["flights"] >= 500].copy()
plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    airport_delay_plot["ORIGIN_LONGITUDE"],
    airport_delay_plot["ORIGIN_LATITUDE"],
    c=airport_delay_plot["avg_arrival_delay"],
    s=np.sqrt(airport_delay_plot["flights"]) * 2,
    cmap="coolwarm",
    alpha=0.75,
)
plt.colorbar(scatter, label="Average arrival delay (min)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Origin airports by average arrival delay")
plt.tight_layout()
plt.savefig(output_dir / "completed_origin_airport_delay_map.png", dpi=150)
plt.close()

print(f"Joined dataset: {df_join.shape[0]:,} rows x {df_join.shape[1]:,} columns")
print(
    f"Completed dataset: {df_completed.shape[0]:,} rows x "
    f"{df_completed.shape[1]:,} columns"
)
print(f"Reports saved to: {output_dir}")
