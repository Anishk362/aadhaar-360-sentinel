import pandas as pd
import glob
import json
import os
import re

# --- CONFIGURATION ---
BASE_PATH = "data/raw_csvs"
OUTPUT_PATH = "processed_metrics.json"

# Professor's Mandatory Replacement Map
# This fixes the lexical fragmentation discovered in your audit
REPLACEMENT_MAP = {
    "WESTBENGAL": "WEST BENGAL",
    "WEST BENGAL": "WEST BENGAL",
    "WESTBENGAL": "WEST BENGAL",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "ANUGAL": "ANGUL"
}

def clean_and_standardize(s):
    """
    Member 2's Cleaning Engine:
    1. Forces Uppercase & Strips spaces.
    2. Removes symbols (*, #, etc.) via Regex.
    3. Collapses multiple spaces into one.
    4. Applies the hard replacement map.
    """
    if not isinstance(s, str):
        s = str(s)
    
    # Standardize casing and symbols
    s = s.upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)  # Remove symbols like *
    s = " ".join(s.split())           # Fix double spaces like "WEST  BENGAL"
    
    # Apply hard replacements
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
        # Standardize headers while reading
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

    # 2. DATA INTEGRITY PASS
    for df in [df_enrol, df_bio, df_demo]:
        # Filter out rows where state/district is just numbers (the 100000 bug)
        df = df[~df['state'].astype(str).str.isnumeric()]
        
        # Apply the Standardization Engine
        df['state'] = df['state'].apply(clean_and_standardize)
        df['district'] = df['district'].apply(clean_and_standardize)

    # 3. TRANSFORMATION: ENROLMENT (Summing counts)
    df_enrol["total_enrolment"] = (
        df_enrol["age_0_5"] + 
        df_enrol["age_5_17"] + 
        df_enrol["age_18_greater"]
    )
    enrol_agg = df_enrol.groupby(["state", "district"], as_index=False)["total_enrolment"].sum()

    # 4. TRANSFORMATION: BIOMETRIC (Summing female counts)
    df_bio["female_count"] = df_bio["bio_age_5_17"] + df_bio["bio_age_17_"]
    bio_agg = df_bio.groupby(["state", "district"], as_index=False)["female_count"].sum()

    # 5. TRANSFORMATION: DEMOGRAPHIC (Mobile Update Volume)
    df_demo["mobile_update_volume"] = df_demo["demo_age_5_17"] + df_demo["demo_age_17_"]
    demo_agg = df_demo.groupby(["state", "district"], as_index=False)["mobile_update_volume"].sum()

    # 6. THE INTELLIGENCE MERGE
    final_df = enrol_agg.merge(bio_agg, on=["state", "district"], how="left")
    final_df = final_df.merge(demo_agg, on=["state", "district"], how="left")
    
    # Fill gaps with 0 before calculation
    final_df.fillna(0, inplace=True)

    # 7. FEATURE ENGINEERING: FEMALE RATIO (Inclusivity Pillar)
    # Important: Ratio = Sum(Female) / Sum(Total)
    final_df["female_enrolment_pct"] = (
        final_df["female_count"] / final_df["total_enrolment"]
    ).fillna(0)
    
    # Cap at 1.0 to prevent data anomalies
    final_df["female_enrolment_pct"] = final_df["female_enrolment_pct"].clip(upper=1.0)

    # 8. EXPORTING TO API CONTRACT
    final_df.rename(columns={"state": "State", "district": "District"}, inplace=True)
    
    # Final selection of columns for Member 4 & 5
    output_df = final_df[[
        "State", 
        "District", 
        "total_enrolment", 
        "mobile_update_volume", 
        "female_enrolment_pct"
    ]]

    # Save to project root
    output_df.to_json(OUTPUT_PATH, orient="records", indent=2)
    
    print(f"✅ ETL COMPLETE: {len(output_df)} Normalized Records Generated.")
    print(f"📂 File Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()