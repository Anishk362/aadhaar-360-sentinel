import pandas as pd
import glob
import json
import os

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

    # -------------------------------
    # ENROLMENT → Total_Enrolment
    # -------------------------------
    df_enrol["Total_Enrolment"] = (
        df_enrol["age_0_5"] +
        df_enrol["age_5_17"] +
        df_enrol["age_18_greater"]
    )

    enrol_agg = (
        df_enrol
        .groupby(["state", "district"], as_index=False)["Total_Enrolment"]
        .sum()
    )

    # -------------------------------
    # BIOMETRIC → Gender_Female (proxy)
    # -------------------------------
    df_bio["Gender_Female"] = (
        df_bio["bio_age_5_17"] +
        df_bio["bio_age_17_"]
    )

    bio_agg = (
        df_bio
        .groupby(["state", "district"], as_index=False)["Gender_Female"]
        .sum()
    )

    # -------------------------------
    # DEMOGRAPHIC → Mobile_Number_Updates (proxy)
    # -------------------------------
    df_demo["Mobile_Number_Updates"] = (
        df_demo["demo_age_5_17"] +
        df_demo["demo_age_17_"]
    )

    demo_agg = (
        df_demo
        .groupby(["state", "district"], as_index=False)["Mobile_Number_Updates"]
        .sum()
    )

    # -------------------------------
    # MERGE ALL
    # -------------------------------
    final_df = enrol_agg.merge(
        bio_agg, on=["state", "district"], how="left"
    ).merge(
        demo_agg, on=["state", "district"], how="left"
    )

    final_df.fillna(0, inplace=True)

    # Rename columns to API contract
    final_df.rename(columns={
        "state": "State",
        "district": "District"
    }, inplace=True)

    # -------------------------------
    # SAVE JSON
    # -------------------------------
    output_path = "processed_metrics.json"
    final_df.to_json(output_path, orient="records", indent=2)

    print("✅ processed_metrics.json generated successfully!")


if __name__ == "__main__":
    main()
