# FakeJobClassifier
這是一個基於機器學習的文字分類專案，旨在根據職缺的文字描述，自動偵測並識別出具有詐騙風險的不實職缺資訊，以提升求職者的安全意識。

本專案為「國立臺北科技大學資訊工程系」113學年度下學期「自然語言處理與文字探勘」課程之期末專題。


---
## 安裝與設定
複製專案：
```bash
git clone https://github.com/LucienRay/FakeJobClassifier.git
```

安裝必要函式庫：
```bash
pip install -r requirements.txt
```

---

## 使用教學
在使用任何預測腳本前，請先執行訓練腳本來產生模型檔案。

1. 訓練模型 (train.py)

    此腳本會讀取 fake_job_postings.csv，訓練模型，並將訓練好的模型和向量化工具儲存到 model/ 資料夾中。

    ```bash
    python train.py
    ```


2. 預測 (predict.py)
    
    此模式適用於快速測試單筆職缺描述。

    ```bash
    python predict.py <text_path>
   #範例
   #python predict.py test/job1.txt
    ```
