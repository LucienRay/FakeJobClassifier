import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier # <--- 已修改
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time
import os

# --- 資料預處理 (與之前相同) ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- 主流程 ---

def load_dataset():
    try:
        df = pd.read_csv('fake_job_postings.csv')
        print("資料集載入成功！")
    except FileNotFoundError:
        print("錯誤：找不到 'fake_job_postings.csv'。請確保檔案存在於正確的路徑。")
        exit()

    df['text'] = df['title'].fillna('') + ' ' + df['company_profile'].fillna('') + ' ' + df['description'].fillna('')
    df = df[['text', 'fraudulent']]

    print("正在進行文本預處理...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df

def process_dataset(df):

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("資料分割完成！")
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(X_train, y_train):
    print("正在訓練隨機森林模型...")
    start_time = time.time()

    # --- 主要修改處 ---
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1  # 使用所有 CPU 核心加速
    )
    # --------------------

    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"模型訓練完成！ 花費時間: {end_time - start_time:.2f} 秒")
    return model

def evaluate_model(model, X_test, y_test):
    print("\n--- 模型評估報告 ---")
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.25
    y_pred = (y_pred_probs >= threshold).astype(int)
    print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("混淆矩陣 (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 20)
    print("---------------------\n")

def save_model(model, vectorizer):
    print("正在保存模型...")
    MODEL_DIRECTORY = "model"
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIRECTORY, 'rf_job_scam_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIRECTORY, 'tfidf_vectorizer.pkl'))
    print(f"模型已成功保存至'{MODEL_DIRECTORY}'資料夾！")

if __name__ == '__main__':
    df = load_dataset()
    X_train, X_test, y_train, y_test, vectorizer = process_dataset(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, vectorizer)

