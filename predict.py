import joblib
import re
import nltk
from nltk.corpus import stopwords
import os
import argparse

# --- 1. 設定命令列參數 ---
parser = argparse.ArgumentParser(description='讀取單一文字檔案，並分析其是否為詐騙職缺。')
parser.add_argument('input_file', type=str, help='包含單一職缺描述的文字檔案路徑 (例如: job_ad.txt)。')

args = parser.parse_args()

# --- 2. 預處理函式 (與之前相同) ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# --- 3. 載入模型 (與之前相同) ---
MODEL_PATH = os.path.join('model', 'rf_job_scam_model.pkl')
VECTORIZER_PATH = os.path.join('model', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    print(f"❌ 錯誤：找不到模型檔案。請確認 '{MODEL_PATH}' 和 '{VECTORIZER_PATH}' 是否存在。")
    exit()

# --- 5. 主流程 ---
if __name__ == '__main__':
    input_path = args.input_file

    print(f"正在讀取檔案: {input_path}")
    try:
        # 讀取檔案的全部內容
        with open(input_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        if not file_content.strip():
            print("錯誤：檔案內容為空。")
            exit()

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_path}'。")
        exit()
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        exit()

    # --- 分析與輸出 ---
    print("檔案讀取成功，正在進行分析...")

    # 預處理、轉換、預測
    cleaned_text = preprocess_text(file_content)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    # 在命令列上直接顯示結果
    print("=" * 30)
    print("          分析結果")
    print("=" * 30)
    if prediction == 1:
        confidence = probability[1]
        print(f"預測結果: 詐騙職缺 (Fraudulent)")
    else:
        confidence = probability[0]
        print(f"預測結果: 真實職缺 (Real)")

    print(f"信心分數: {confidence:.2%}")
