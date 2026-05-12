"""
Customer Complaint Classifier
Classifies telecom complaints into: billing, network, service, fraud

Author: [Previous intern - no longer with company]
Note: "Model achieves 94% accuracy - ready for production!"

YOUR TASK: Find and fix the bugs in this pipeline.
There are 5 intentional bugs. Finding 3+ is a strong result.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def load_data(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "data", "complaints.csv")
    """Load the complaints dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Categories: {df['category'].unique()}")
    return df


def preprocess_text(text: str) -> str:
    """Clean and preprocess complaint text."""
    import re

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from the complaint data."""
    # Clean the text
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Extract text length as a feature
    df["text_length"] = df["clean_text"].apply(len)

    # Extract complaint ID prefix as a feature (seems useful for routing!)
    # BUG 1: complaint_id encodes the category (BIL-, NET-, SVC-, FRD-)
    # This causes massive data leakage - the model learns the ID prefix
    # instead of the actual text content
    df["id_prefix"] = df["complaint_id"].str[:3]

    return df


def build_model(df: pd.DataFrame):
    """Build and evaluate the classification model."""

    # Encode the target variable
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])

    # BUG 2: TF-IDF is fit on the ENTIRE dataset before splitting
    # This leaks information from test set into training (vocabulary & IDF weights)
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

    # Convert to DataFrame for easier manipulation
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
    )

    # Combine all features
    # BUG 1 (continued): id_prefix is one-hot encoded and included as a feature
    feature_df = pd.concat(
        [
            tfidf_df,
            pd.get_dummies(df["id_prefix"], prefix="prefix"),
            df[["text_length"]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, df["label"], test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    # BUG 4: Only reporting accuracy on an imbalanced dataset
    # With 60% billing class, a model that just predicts "billing" gets 60% accuracy
    # The 94% looks great but hides poor performance on minority classes
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nModel is ready for production! ✓")
    print(f"{'='*50}")

    # BUG 5: The classification report IS generated but the results are
    # printed with the WRONG label names due to LabelEncoder ordering
    # LabelEncoder sorts alphabetically: billing=0, fraud=1, network=2, service=3
    # But the report below uses a WRONG manual order
    print(f"\nDetailed Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["billing", "network", "service", "fraud"],
            #                          ^^^^^^^ WRONG ORDER - should be alphabetical
            #                          LabelEncoder maps: billing=0, fraud=1, network=2, service=3
            #                          But this says: billing=0, network=1, service=2, fraud=3
        )
    )

    # Feature importance
    importances = model.feature_importances_
    feature_names = feature_df.columns
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]

    print(f"\nTop 10 Most Important Features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

    return model, vectorizer, le


def main():
    print("=" * 50)
    print("CUSTOMER COMPLAINT CLASSIFIER")
    print("=" * 50)

    # Load data
    df = load_data()

    # Feature engineering
    df = engineer_features(df)

    # Build and evaluate model
    model, vectorizer, label_encoder = build_model(df)

    print(f"\n{'='*50}")
    print("Pipeline complete. See accuracy above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
