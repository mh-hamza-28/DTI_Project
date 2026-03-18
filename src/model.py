import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from features import drug_features, protein_features

def prepare_data(df):
    """Extracts features iteratively across dataset."""
    X = []
    y = []
    for _, row in df.iterrows():
        df_feats = drug_features(row['smiles'])
        pf_feats = protein_features(row['sequence'])
        
        # Avoid crashing on bad data
        if df_feats is not None and pf_feats is not None:
            features = list(df_feats) + list(pf_feats)
            X.append(features)
            y.append(row['label'])
            
    return X, y

def train_and_eval(X, y):
    """Trains Logistic Regression & Random Forest models and evaluates."""
    # Split dataset into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression Model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    if len(set(y_test)) > 1:
        print(f"ROC-AUC score: {roc_auc_score(y_test, lr_probs):.4f}")
    else:
        print("ROC-AUC score: undefined (test set contains only one class).")
        
    # 2. Random Forest Model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    if len(set(y_test)) > 1:
        print(f"ROC-AUC score: {roc_auc_score(y_test, rf_probs):.4f}")
    else:
        print("ROC-AUC score: undefined (test set contains only one class).")
    
    return lr, rf
