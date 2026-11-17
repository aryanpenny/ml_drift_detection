import pandas as pd
import os

RAW_DATA_PATH="data/raw/telco_churn.csv"
CHUNKS_DIR="data/chunks"
NUM_CHUNKS=4

def load_raw_data(path):
    print(f"loading raw data {path}")
    df = pd.read_csv(path)
    print(f"total rows {len(df)}")
    return df

def split_into_chunks(df, num_chunks):
    print(f"splitting into {num_chunks}")
    chunks_size= len(df) // num_chunks
    chunks=[]

    for i in range(num_chunks):
        start = i * chunks_size
        end= start + chunks_size if i < num_chunks -1 else len(df)
        chunks.append(df.iloc[start:end])
        print(f"[INFO] Chunk {i+1}: rows {start} to {end} ({end - start} rows)")
    
    return chunks

def save_chunks(chunks, output_dir):
    os.makedirs(output_dir, exist_ok= True)
    for i, chunk in enumerate(chunks):
        filename = os.path.join(output_dir, f"chunk_{i+1}.csv")
        chunk.to_csv(filename, index= False)
        print(f"[INFO] Saved: {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("  INGEST: Splitting raw data into chunks")
    print("=" * 50)

    df = load_raw_data(RAW_DATA_PATH)
    chunks = split_into_chunks(df, NUM_CHUNKS)
    save_chunks(chunks,CHUNKS_DIR)
    



