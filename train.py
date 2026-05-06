import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from features import EmailFeatureTransformer

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "phishing_emails.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_pipeline.joblib")


def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(
            {
                "text": [
                    "Your account has been suspended. Click here to verify your identity.",
                    "Dear customer, we need to update your payment information immediately.",
                    "Submit your password now to avoid service interruption.",
                    "Your package delivery is delayed. Confirm your address.",
                    "Please find the invoice attached for your recent order.",
                    "Meeting agenda for Monday attached. Let me know if you have questions.",
                    "Can we reschedule the demo to Wednesday afternoon?",
                    "Your subscription renewal is confirmed. Thank you for using our service.",
                    "Quarterly report is ready for review. Please check the shared folder.",
                    "Team lunch is planned for Friday at noon. RSVP if you can join.",
                ],
                "label": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"legitimate": "0", "phishing": "1"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("features", EmailFeatureTransformer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train()
