import pandas as pd
import glob
import json
import os
import re

# --- CONFIGURATION ---
# Points to the directory where the CSVs are stored
BASE_PATH = "data/raw_csvs"

# CRITICAL PATH FIX: Saves specifically inside the src/etl_pipeline folder 
# This ensures train_forecaster.py and app.py can find the 'Source of Truth'
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_metrics.json")

# Member 2: Optimized Lexical Standardization Map
# Collapses legacy fragments and typos into unified National entries
REPLACEMENT_MAP = {
    "WESTBENGAL": "WEST BENGAL",
    "WEST BANGAL": "WEST BENGAL",
    "JAMMU KASHMIR": "JAMMU & KASHMIR",
    "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DADRA AND NAGAR HAVELI": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DAMAN AND DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DAMAN DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
}

def clean_and_standardize(s):
    """
    Standardization Engine:
    1. Forces Uppercase & Strips whitespace.
    2. Removes symbols via Regex.
    3. Applies the hard replacement map for National unification.
    """
    if not isinstance(s, str):
        s = str(s)
    
    s = s.upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s) 
    s = " ".join(s.split()) 
    
    if s in REPLACEMENT_MAP:
        s = REPLACEMENT_MAP[s]
    return s

def load_chunked_data(folder_name):
    path = os.path.join(BASE_PATH, folder_name, "*.csv")
    files = glob.glob(path)
    if not files:
        print(f"⚠️ Warning: No files found in {folder_name}")
        return pd.DataFrame()

    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        temp_df.columns = [c.strip().lower() for c in temp_df.columns]
        df_list.append(temp_df)
    
    return pd.concat(df_list, ignore_index=True)

def main():
    print("🚀 Initializing ETL Intelligence Pipeline v3.0...")

    # 1. LOAD DATA
    df_enrol = load_chunked_data("enrolment")
    df_bio = load_chunked_data("biometric")
    df_demo = load_chunked_data("demographic")

    # 2. DATA INTEGRITY PASS (Fixes SettingWithCopyWarning)
    processed_dfs = []
    for df in [df_enrol, df_bio, df_demo]:
        # Filter and create an explicit copy to avoid Pandas indexing warnings
        clean_df = df[~df['state'].astype(str).str.isnumeric()].copy()
        
        clean_df['state'] = clean_df['state'].apply(clean_and_standardize)
        clean_df['district'] = clean_df['district'].apply(clean_and_standardize)
        processed_dfs.append(clean_df)
    
    df_enrol, df_bio, df_demo = processed_dfs

    # 3. TRANSFORMATION: ENROLMENT
    df_enrol["total_enrolment"] = (
        df_enrol["age_0_5"] + 
        df_enrol["age_5_17"] + 
        df_enrol["age_18_greater"]
    )
    enrol_agg = df_enrol.groupby(["state", "district"], as_index=False)["total_enrolment"].sum()

    # 4. TRANSFORMATION: BIOMETRIC (Calculates raw female count)
    df_bio["female_count"] = df_bio["bio_age_5_17"] + df_bio["bio_age_17_"]
    bio_agg = df_bio.groupby(["state", "district"], as_index=False)["female_count"].sum()

    # 5. TRANSFORMATION: DEMOGRAPHIC (Mobile Update Volume)
    df_demo["mobile_update_volume"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
    demo_agg = df_demo.groupby(["state", "district"], as_index=False)["mobile_update_volume"].sum()

    # 6. THE INTELLIGENCE MERGE
    final_df = enrol_agg.merge(bio_agg, on=["state", "district"], how="left")
    final_df = final_df.merge(demo_agg, on=["state", "district"], how="left")
    final_df.fillna(0, inplace=True)

    # 7. FEATURE ENGINEERING: DISTRICT-LEVEL RATIO
    final_df["female_enrolment_pct"] = (
        final_df["female_count"] / final_df["total_enrolment"]
    ).fillna(0).clip(upper=1.0)

    # 8. EXPORTING TO API CONTRACT
    final_df.rename(columns={"state": "State", "district": "District"}, inplace=True)
    
    output_df = final_df[[
        "State", 
        "District", 
        "total_enrolment", 
        "mobile_update_volume", 
        "female_enrolment_pct"
    ]]

    # Final Export
    output_df.to_json(OUTPUT_PATH, orient="records", indent=2)
    
    print(f"✅ ETL COMPLETE: {len(output_df)} Normalized Records Generated.")
    print(f"📂 File Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()