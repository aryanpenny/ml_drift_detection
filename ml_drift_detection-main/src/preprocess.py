import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CHUNK_DIR= "data/chunks"
PROCESSED_DIR= "data/processed"
TEST_SIZE= 0.2
RANDOM_STATE= 42

def load_chunk(chunk_number):
    path = os.path.join(CHUNK_DIR, f"chunk_{chunk_number}.csv")
    print(f"[info] path loading {path}")
    df = pd.read_csv(path)
    print(f"[info] rows loaded: {len(df)}")
    return df
    
def clean_data(df):
    # 1. Drop columns we don't need (like customerID because it doesn't help predict churn)
    if "customerID" in df.columns:
        df = df.drop(columns= ["customerID"])
    
    # 2. Some columns are meant to be numbers but might have empty spaces. Let's fix that.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 3. Convert all the "Yes"/"No", "Male"/"Female" text into numbers (0s and 1s)
    # We use LabelEncoder for this
    encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column]= encoder.fit_transform(df[column])
    return df

def split_and_save(df):
    os.makedirs(PROCESSED_DIR, exist_ok= True)

    X = df.drop(columns= ["Churn"])
    y = df["Churn"]

# Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

if __name__ == "__main__":
    chunk_number = int(sys.argv[1]) if len(sys.argv) >1 else 1
    df_raw = load_chunk(chunk_number)
    df_cleaned = clean_data(df_raw)
    split_and_save(df_cleaned)
