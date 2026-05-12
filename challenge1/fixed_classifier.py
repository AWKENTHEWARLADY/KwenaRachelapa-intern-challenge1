"""
Fixed Customer Complaint Classifier
Fixes:
1. Removed complaint_id leakage
2. Fixed TF-IDF data leakage
3. Corrected train/test pipeline order
4. Improved evaluation metrics
5. Fixed label mapping issue
"""

import pandas as pd
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


def load_data(filepath=None):
    """Load complaints dataset."""

    if filepath is None:
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "data", "complaints.csv")

    df = pd.read_csv(filepath)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total records: {len(df)}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())

    return df


def preprocess_text(text):
    """Clean complaint text."""

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_data(df):
    """Prepare clean features and labels."""

    # Clean text
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category"])

    # ONLY use complaint text
    # Removed complaint_id leakage
    X = df["clean_text"]

    y = df["label"]

    return X, y, label_encoder


def split_data(X, y):
    """Split dataset."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Fit TF-IDF ONLY on training data.
    Prevents test data leakage.
    """

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english"
    )

    # Fit ONLY on train
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test using same vocabulary
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


def train_model(X_train, y_train):
    """Train classifier."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model properly."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def show_top_features(model, vectorizer):
    """Display important words/features."""

    feature_names = vectorizer.get_feature_names_out()

    importances = model.feature_importances_

    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    print("\nTop Important Features:")
    print("-" * 50)

    for feature, importance in top_features:
        print(f"{feature:<20} {importance:.4f}")


def main():

    print("=" * 50)
    print("FIXED CUSTOMER COMPLAINT CLASSIFIER")
    print("=" * 50)

    # Load dataset
    df = load_data()

    # Prepare data
    X, y, label_encoder = prepare_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train,
        X_test
    )

    # Train model
    model = train_model(X_train_vec, y_train)

    # Evaluate model
    evaluate_model(
        model,
        X_test_vec,
        y_test,
        label_encoder
    )

    # Feature importance
    show_top_features(model, vectorizer)

    print("\n" + "=" * 50)
    print("Pipeline completed successfully")
    print("=" * 50)
if __name__ == "__main__":
    main()

