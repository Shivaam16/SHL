# Grammar Scoring Engine - SHL Assessment (Notebook Version)

import os
import pandas as pd
import whisper
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# ======================
# ⚙️ CONFIG
# ======================
SAMPLE_SIZE = 5  # Keep this low for testing; increase for full runs
WHISPER_MODEL = "base"

# Paths
train_csv = 'dataset/train.csv'
test_csv = 'dataset/test.csv'
train_audio_path = 'dataset/audios_train'
test_audio_path = 'dataset/audios_test'
os.makedirs("outputs", exist_ok=True)

# ======================
# 📥 Load CSVs
# ======================
train_df = pd.read_csv(train_csv).head(SAMPLE_SIZE)
test_df = pd.read_csv(test_csv).head(SAMPLE_SIZE)

# ======================
# 🔁 Load Whisper model
# ======================
print("🔁 Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)

# ======================
# 🧠 Safe transcription
# ======================
def transcribe(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return ""
    try:
        print(f"🎧 Transcribing: {audio_path}")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"⚠️ Error transcribing {audio_path}: {e}")
        return ""

# ======================
# 🎙️ Transcribe training audio
# ======================
print("📝 Transcribing training data...")
train_df["transcript"] = train_df["filename"].apply(lambda x: transcribe(os.path.join(train_audio_path, x)))

# ======================
# 📊 Train-validation split
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    train_df["transcript"], train_df["label"], test_size=0.2, random_state=42
)

# ======================
# 🧪 Model pipeline
# ======================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('regressor', RandomForestRegressor(random_state=42))
])

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# ======================
# 📈 Evaluation
# ======================
y_pred = pipeline.predict(X_val)
if len(y_val) >= 2:
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    pearson_corr, _ = pearsonr(y_val, y_pred)

    print(f"\n📊 Evaluation Metrics:")
    print(f"🔹 Mean Squared Error: {mse:.4f}")
    print(f"🔹 R^2 Score: {r2:.4f}")
    print(f"🔹 Pearson Correlation: {pearson_corr:.4f}")

    # Visualization: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_val, y=y_pred)
    plt.xlabel("Actual Grammar Score")
    plt.ylabel("Predicted Grammar Score")
    plt.title("Actual vs Predicted Grammar Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.show()
else:
    print("⚠️ Not enough validation samples to evaluate metrics or plot graphs.")

# ======================
# 💾 Save model
# ======================
print("💾 Saving model...")
joblib.dump(pipeline, "outputs/grammar_model.joblib")

# ======================
# 🧪 Transcribe test data
# ======================
print("🧪 Transcribing test data...")
test_df["transcript"] = test_df["filename"].apply(lambda x: transcribe(os.path.join(test_audio_path, x)))

# ======================
# 📤 Generate predictions
# ======================
print("📊 Generating predictions...")
test_df["label"] = pipeline.predict(test_df["transcript"])

# ======================
# 📁 Save submission
# ======================
submission = test_df[["filename", "label"]]
submission.to_csv("outputs/submission.csv", index=False)

print("\n✅ Done. Submission saved to outputs/submission.csv")
print(submission.head())
