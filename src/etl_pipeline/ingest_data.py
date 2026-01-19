import numpy as np
import pandas as pd  # FIX: Added this to resolve NameError
import glob
import json
import os
import re

# --- CONFIGURATION ---
BASE_PATH = "data/raw_csvs"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_metrics.json")

# THE OFFICIAL 36 STATES/UTs
# --- STANDARDIZED CONFIGURATION FOR MAP SYNC ---

# Ensure these match the 'name' attribute in SMapIndia.instructions
OFFICIAL_ENTITIES = [
    "ANDAMAN AND NICOBAR ISLANDS", "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", 
    "CHANDIGARH", "CHHATTISGARH", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DELHI", "GOA", 
    "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "JHARKHAND", "KARNATAKA", 
    "KERALA", "LADAKH", "LAKSHADWEEP", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", 
    "MIZORAM", "NAGALAND", "ODISHA", "PUDUCHERRY", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", 
    "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL"
]

# Expanded Permutation Map to catch "Dirty" CSV data
PERMUTATION_MAP = {
    "ANDAMAN NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN & NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAAMAN NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DADRA NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "UTTARANCHAL": "UTTARAKHAND",
    "CHHATISGARH": "CHHATTISGARH",
    "WESTBENGAL": "WEST BENGAL",
    "WEST BANGAL": "WEST BENGAL",
    "WEST BENGLI": "WEST BENGAL",
    "JAMMU KASHMIR": "JAMMU AND KASHMIR"
}

TELANGANA_DISTRICTS = ["ADILABAD", "HYDERABAD", "KARIMNAGAR", "KHAMMAM", "MAHABUBNAGAR", "MEDAK", "NALGONDA", "NIZAMABAD", "RANGAREDDI", "WARANGAL"]

def canonicalize(name, is_state=True):
    if not isinstance(name, str) or any(char.isdigit() for char in name):
        return "REMOVE_ME"
    s = name.upper().strip().replace("&", "AND")
    s = re.sub(r'[^A-Z\s]', '', s)
    s = " ".join(s.split())
    if is_state:
        if s in PERMUTATION_MAP: return PERMUTATION_MAP[s]
        for official in OFFICIAL_ENTITIES:
            if official in s or s in official: return official
    return s

def load_chunked_data(folder_name):
    path = os.path.join(BASE_PATH, folder_name, "*.csv")
    files = glob.glob(path)
    if not files: return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def main():
    print("🚀 Initializing Aadhaar-Heal Sanitization Engine v4.9...")
    df_enrol = load_chunked_data("enrolment")
    df_demo = load_chunked_data("demographic")
    
    if df_enrol.empty or df_demo.empty:
        print("❌ Data folders empty!")
        return

    for df in [df_enrol, df_demo]:
        df.columns = [c.strip().lower() for c in df.columns]
        df['state'] = df['state'].apply(lambda x: canonicalize(x, True))
        df['district'] = df['district'].apply(lambda x: canonicalize(x, False))
        df.loc[df['district'].isin(TELANGANA_DISTRICTS), 'state'] = "TELANGANA"
    
    df_enrol = df_enrol[df_enrol['state'] != "REMOVE_ME"]
    for col in ["age_0_5", "age_5_17", "age_18_greater"]:
        df_enrol[col] = pd.to_numeric(df_enrol[col], errors='coerce').fillna(0)
    
    df_enrol["total"] = df_enrol["age_0_5"] + df_enrol["age_5_17"] + df_enrol["age_18_greater"]
    df_enrol["youth"] = df_enrol["age_0_5"] + df_enrol["age_5_17"]
    
    en_agg = df_enrol.groupby(["state", "district"], as_index=False).agg(total_enrolment=('total', 'sum'), youth_count=('youth', 'sum'))
    de_agg = df_demo.groupby(["state", "district"], as_index=False).agg(mobile_update_volume=('demo_age_17_', 'sum'))

    final_df = en_agg.merge(de_agg, on=["state", "district"], how="left").fillna(0)
    final_df = final_df[final_df['state'].isin(OFFICIAL_ENTITIES)]
    final_df["ratio"] = (final_df["youth_count"] / final_df["total_enrolment"]).replace([np.inf, -np.inf], 0).fillna(0)

    final_df.rename(columns={"state": "State", "district": "District"}, inplace=True)
    final_df.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"✅ SUCCESS: {len(final_df)} districts unified into 36 Entities.")

if __name__ == "__main__": main()