from datasets import load_dataset
import pandas as pd
import os

def load_and_clean_dataset():
    # Load the dataset
    ds = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
    
    # Convert to DataFrame
    df = pd.DataFrame(ds["train"])
    
    # Rename columns to standard names
    df = df[["input", "output"]].rename(columns={"input": "query", "output": "response"})
    
    # Drop missing or duplicate entries
    df = df.dropna().drop_duplicates()

    return df

if __name__ == "__main__":
    df = load_and_clean_dataset()
    
    # Make sure storage directory exists
    os.makedirs("dataset/storage", exist_ok=True)

    # Save to CSV
    df.to_csv("dataset/storage/clean_data.csv", index=False)
    print("✅ Cleaned dataset saved to storage/clean_data.csv")
