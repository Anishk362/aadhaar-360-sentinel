import numpy as np
import pandas as pd
import glob
import json
import os
import re

# --- CONFIGURATION ---
BASE_PATH = "data/raw_csvs"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_metrics.json")

# THE OFFICIAL 36 STATES/UTs
OFFICIAL_ENTITIES = [
    "ANDAMAN AND NICOBAR ISLANDS", "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", 
    "CHANDIGARH", "CHHATTISGARH", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "DELHI", "GOA", 
    "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "JHARKHAND", "KARNATAKA", 
    "KERALA", "LADAKH", "LAKSHADWEEP", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", 
    "MIZORAM", "NAGALAND", "ODISHA", "PUDUCHERRY", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", 
    "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL"
]

# RESTORED: Your Exact Permutation Map
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
    print("🚀 RGIPT Intelligence Unit: Initializing Multi-Stream Sanitization...")
    df_enrol = load_chunked_data("enrolment")
    df_demo = load_chunked_data("demographic")
    df_bio = load_chunked_data("biometric") # FIX: Integrated Biometric
    
    if df_enrol.empty or df_demo.empty:
        print("❌ CRITICAL: Data source folders are empty!")
        return

    for df in [df_enrol, df_demo, df_bio]:
        df.columns = [c.strip().lower() for c in df.columns]
        df['state'] = df['state'].apply(lambda x: canonicalize(x, True))
        df['district'] = df['district'].apply(lambda x: canonicalize(x, False))
        df.loc[df['district'].isin(TELANGANA_DISTRICTS), 'state'] = "TELANGANA"
        # Convert date for monthly normalization
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df['month_key'] = df['date'].dt.to_period('M')

    # Aggregation Strategy: Mean Monthly Pulse (Prevents 12x inflation)
    df_enrol["total"] = pd.to_numeric(df_enrol["age_0_5"], errors='coerce').fillna(0) + \
                        pd.to_numeric(df_enrol["age_5_17"], errors='coerce').fillna(0) + \
                        pd.to_numeric(df_enrol["age_18_greater"], errors='coerce').fillna(0)
    
    df_enrol["youth"] = pd.to_numeric(df_enrol["age_0_5"], errors='coerce').fillna(0) + \
                        pd.to_numeric(df_enrol["age_5_17"], errors='coerce').fillna(0)

    # 1. Group by District and Month to get Monthly Sums, then average those months
    en_agg = df_enrol.groupby(["state", "district", "month_key"]).agg(m_total=('total', 'sum'), m_youth=('youth', 'sum')).reset_index()
    en_final = en_agg.groupby(["state", "district"]).agg(total_enrolment=('m_total', 'mean'), youth_count=('m_youth', 'mean')).reset_index()

    de_agg = df_demo.groupby(["state", "district", "month_key"]).agg(m_vol=('demo_age_17_', 'sum')).reset_index()
    de_final = de_agg.groupby(["state", "district"]).agg(demo_vol=('m_vol', 'mean')).reset_index()

    bio_agg = df_bio.groupby(["state", "district", "month_key"]).agg(m_vol=('bio_age_17_', 'sum')).reset_index()
    bio_final = bio_agg.groupby(["state", "district"]).agg(bio_vol=('m_vol', 'mean')).reset_index()

    # Merge Streams
    final_df = en_final.merge(de_final, on=["state", "district"], how="left").merge(bio_final, on=["state", "district"], how="left").fillna(0)
    final_df = final_df[final_df['state'].isin(OFFICIAL_ENTITIES)]
    
    # Accuracy fix for Ratio
    final_df["ratio"] = (final_df["youth_count"] / (final_df["total_enrolment"] + 1)).clip(0.12, 0.98)
    final_df["mobile_update_volume"] = final_df["demo_vol"] + final_df["bio_vol"]

    final_df.rename(columns={"state": "State", "district": "District"}, inplace=True)
    final_df.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"✅ DATA READY: {len(final_df)} districts normalized to Monthly Average Loads.")

if __name__ == "__main__": main()