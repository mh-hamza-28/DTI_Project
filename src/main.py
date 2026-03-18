import pandas as pd
import os
from model import prepare_data, train_and_eval

def main():
    print("------------------------------------------")
    print("Drug-Target Interaction Prediction Project")
    print("------------------------------------------")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dti_data.csv")
    
    try:
        # Load CSV using pandas
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} sample rows.")
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return
        
    # Apply feature extraction & Combine features
    print("Extracting drug and protein features...")
    X, y = prepare_data(df)
    
    if len(X) == 0:
        print("No valid features could be extracted. Exiting.")
        return
        
    print(f"Successfully prepared features for {len(X)} samples.")
    
    # Train model using model.py
    print("Training models...")
    train_and_eval(X, y)

if __name__ == "__main__":
    main()
