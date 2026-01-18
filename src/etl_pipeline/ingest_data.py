import pandas as pd
import glob
import json
import os

def normalize_location_names(df):
    
    df["state"] = (
        df["state"]
        .astype(str)
        .str.strip()
        .str.replace("&", "And", regex=False)
        .str.title()
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
        .str.replace("&", "And", regex=False)
        .str.title()
    )

    return df


BASE_PATH = "../../data/raw_csvs"

def load_chunked_data(folder_name):
    path = os.path.join(BASE_PATH, folder_name, "*.csv")
    files = glob.glob(path)

    if not files:
        print(f"No files found for {folder_name}")
        return pd.DataFrame()

    print(f"Loading {len(files)} files from {folder_name}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def main():
    # Load datasets
    df_enrol = load_chunked_data("enrolment")
    df_bio = load_chunked_data("biometric")
    df_demo = load_chunked_data("demographic")

    df_enrol = normalize_location_names(df_enrol)
    df_bio   = normalize_location_names(df_bio)
    df_demo  = normalize_location_names(df_demo)


    # ===============================
    # SAFETY CHECKS (VERY IMPORTANT)
    # ===============================

    if df_enrol.empty:
        print("No enrolment data found. Stopping ETL.")
        return

    if df_bio.empty:
        print("No biometric data found. Stopping ETL.")
        return

    if df_demo.empty:
        print("No demographic data found. Stopping ETL.")
        return

    # -------------------------------
    # ENROLMENT → Total_Enrolment
    # -------------------------------
    df_enrol["total_enrolment"] = (
        df_enrol["age_0_5"] +
        df_enrol["age_5_17"] +
        df_enrol["age_18_greater"]
    )

    enrol_agg = (
        df_enrol
        .groupby(["state", "district"], as_index=False)
        .agg(total_enrolment=("total_enrolment", "sum"))
    )

    # -------------------------------
    # BIOMETRIC → FEMALE COUNT (proxy)
    # -------------------------------
    df_bio["female_count"] = (
        df_bio["bio_age_5_17"] +
        df_bio["bio_age_17_"]
    )

    bio_agg = (
        df_bio
        .groupby(["state", "district"], as_index=False)
        .agg(female_count=("female_count", "sum"))
    )

    # -------------------------------
    # DEMOGRAPHIC → Mobile_Number_Updates (proxy)
    # -------------------------------
    df_demo["mobile_update_volume"] = (
        df_demo["demo_age_5_17"] +
        df_demo["demo_age_17_"]
    )

    demo_agg = (
        df_demo
        .groupby(["state", "district"], as_index=False)
        .agg(mobile_update_volume=("mobile_update_volume", "sum"))
    )

    # -------------------------------
    # MERGE ALL
    # -------------------------------
    final_df = (
        enrol_agg
        .merge(bio_agg, on=["state", "district"], how="left")
        .merge(demo_agg, on=["state", "district"], how="left")
    )

    final_df.fillna(0, inplace=True)

    # INCLUSIVITY RATIO (DECIMAL)
    final_df["female_enrolment_pct"] = (
        final_df["female_count"] / final_df["total_enrolment"]
    ).round(3)

    # Rename columns to API contract
    final_df.rename(columns={
        "state": "State",
        "district": "District"
    }, inplace=True)

    final_df = final_df[[
        "State",
        "District", 
        "mobile_update_volume",
        "female_enrolment_pct",
        "total_enrolment"
    ]]

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    output_path = "processed_metrics.json"
    final_df.to_json(output_path, orient="records", indent=2)

    print("✅ processed_metrics.json generated successfully!")


if __name__ == "__main__":
    main()
