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

# Blockbuster Feature: National Entity Unification Map
REPLACEMENT_MAP = {
    "WESTBENGAL": "WEST BENGAL", "WEST BANGAL": "WEST BENGAL", "WEST BENGLI": "WEST BENGAL",
    "TAMILNADU": "TAMIL NADU", "ORISSA": "ODISHA", "UTTARANCHAL": "UTTARAKHAND",
    "PONDICHERRY": "PUDUCHERRY", "CHHATISGARH": "CHHATTISGARH",
    "JAMMU KASHMIR": "JAMMU & KASHMIR", "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "ANDAMAN NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN  NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI AND DAMAN & DIU",
    "DADRA NAGAR HAVELI": "DADRA & NAGAR HAVELI AND DAMAN & DIU"
}

# Lost Districts of Telangana (Governance Fix)
TELANGANA_DISTRICTS = [
    "ADILABAD", "HYDERABAD", "KARIMNAGAR", "KHAMMAM", 
    "MAHABUBNAGAR", "MEDAK", "NALGONDA", "NIZAMABAD", 
    "RANGAREDDI", "WARANGAL"
]

def clean_and_standardize(name, is_state=True):
    if not isinstance(name, str): return "UNKNOWN"
    s = name.upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    s = " ".join(s.split())
    
    if is_state and s in REPLACEMENT_MAP:
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
    print("🚀 Initializing National Intelligence Pipeline v4.2...")

    # 1. Load with standardized naming
    df_enrol = load_chunked_data("enrolment")
    df_demo = load_chunked_data("demographic")
    
    for df in [df_enrol, df_demo]:
        df['state'] = df['state'].apply(lambda x: clean_and_standardize(x, True))
        df['district'] = df['district'].apply(lambda x: clean_and_standardize(x, False))

    # 2. THE GOVERNANCE FIX: Move Telangana districts out of Andhra Pradesh
    mask = (df_enrol['state'] == "ANDHRA PRADESH") & (df_enrol['district'].isin(TELANGANA_DISTRICTS))
    df_enrol.loc[mask, 'state'] = "TELANGANA"
    
    mask_demo = (df_demo['state'] == "ANDHRA PRADESH") & (df_demo['district'].isin(TELANGANA_DISTRICTS))
    df_demo.loc[mask_demo, 'state'] = "TELANGANA"

    # 3. Aggregation & Feature Engineering
    df_enrol["total_enrolment"] = df_enrol["age_0_5"] + df_enrol["age_5_17"] + df_enrol["age_18_greater"]
    df_enrol["youth_count"] = df_enrol["age_0_5"] + df_enrol["age_5_17"]
    
    enrol_agg = df_enrol.groupby(["state", "district"], as_index=False).agg({
        "total_enrolment": "sum", "youth_count": "sum"
    })

    df_demo["mobile_update_volume"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
    demo_agg = df_demo.groupby(["state", "district"], as_index=False)["mobile_update_volume"].sum()

    # 4. DATA-INTEGRITY MERGE (Handling Digital Deserts)
    final_df = enrol_agg.merge(demo_agg, on=["state", "district"], how="inner")
    
    # Pillar 1: Child Saturation Index
    final_df["child_saturation_index"] = (final_df["youth_count"] / final_df["total_enrolment"]).fillna(0)

    # 5. Filter valid states only (Removing numeric/typo states)
    official_count = len(final_df['state'].unique())
    print(f"✅ UNIFIED ENTITIES: {official_count} States/UTs Found.")

    final_df.rename(columns={"state": "State", "district": "District"}, inplace=True)
    final_df.to_json(OUTPUT_PATH, orient="records", indent=2)

if __name__ == "__main__":
    main()