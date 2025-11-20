
👨‍💻 Author
Name : Singara Harendra

Roll Number : DA25M028

# DA5401 – Kaggle Data Challenge  
### Metric Learning for Conversational AI Evaluation  
**Author:** SINGARA HARENDRA  
**Roll No:** DA25M028  

---

## 📌 Overview  
This project solves the DA5401 Kaggle Challenge:  
**predicting how well a prompt–response pair matches a given evaluation metric**  
using embeddings + a neural network regression model.

The dataset provides metric definitions, user prompts, responses, and LLM-assigned scores (0–10).  
Our task is to model this alignment using metric embeddings and prompt–response embeddings.

---

## 📁 Dataset Structure  
The training dataset contains:

- **metric_name**
- **score** (target)
- **user_prompt**
- **response**
- **system_prompt** (many null values)

A **metric_names.json** file is also provided.  
Prompt/response texts appear in **multiple languages**.

---

## 🔧 Data Engineering  

### **1. System Prompt Fixing**
- Missing `system_prompt` entries were replaced with the corresponding **metric_name**.
- If metric names had combined values like `A/B`, the entire text was used.

### **2. Embedding Generation**
- Model used: **google/embeddinggemma-300m**
- Suitable for multilingual text.
- Generated embeddings for:
  - metric definitions  
  - prompt–response pairs  

---

## 🧩 Feature Construction  
Engineered a combined feature vector including:

- Concatenated embeddings:  
  `X = [E_metric || E_text]`
- Cosine similarity  
- Absolute difference: `|E_m – E_t|`
- Element-wise product: `E_m ⊙ E_t`

Saved as `feature_matrix.npy`.

---

## 🧪 EDA (No Visualizations)
- Scores range **0–10**, but **majority lie in 8–10**.
- **Low-score samples (<5) are scarce**, and 0–2 almost absent.
- Heavy class imbalance → required augmentation.

---

## 🛠️ Synthetic Augmentation  
Problem: Dataset lacked low-score examples.

**Solution:**  
- Randomly mismatched metric embeddings with prompt–response embeddings from different metrics.
- These mismatched pairs have **low cosine similarity**, so assigned **low scores**.
- Applied only to training data.

This balances the dataset and prevents the model from overfitting to high-score regions.

---

## 🤖 Model Development  

### **MLP Architecture (PyTorch)**
Hidden layers:  
`4095 → 512 → 256 → 128 → 64 → 1`

Each hidden layer uses:
- ReLU
- Dropout(0.1)

Final layer outputs **1 regression value**.

---

## 📚 Model Architecture Table

| Layer No. | Layer Type             | Output Shape | Parameters |
|----------:|------------------------|--------------|-----------:|
| 1         | Linear (Input→512)     | [-1,512]     | 2,098,176  |
| 2         | ReLU                   | [-1,512]     | 0          |
| 3         | Dropout(0.1)           | [-1,512]     | 0          |
| 4         | Linear (512→256)       | [-1,256]     | 131,328    |
| 5         | ReLU                   | [-1,256]     | 0          |
| 6         | Dropout(0.1)           | [-1,256]     | 0          |
| 7         | Linear (256→128)       | [-1,128]     | 32,896     |
| 8         | ReLU                   | [-1,128]     | 0          |
| 9         | Dropout(0.1)           | [-1,128]     | 0          |
| 10        | Linear (128→64)        | [-1,64]      | 8,256      |
| 11        | ReLU                   | [-1,64]      | 0          |
| 12        | Dropout(0.1)           | [-1,64]      | 0          |
| 13        | Linear (64→1)          | [-1,1]       | 65         |

Total parameters: **~2.27M**

---

## 🏋️ Training Procedure  

- Framework: **PyTorch**
- Optimizer: **AdamW (lr = 1e-3)**
- Loss: **MSELoss**
- Batch size: **256**
- Epochs: **50**
- **5-Fold Cross Validation**
- Best model saved per fold based on validation RMSE.

---

## 📊 Training & Validation Results  

| Fold | Epoch10 | Epoch20 | Epoch40 | Best RMSE | Best Epoch |
|-----:|---------|---------|---------|-----------|-----------:|
| 0 | 2.461 | 2.277 | 2.162 | 2.113 | 39 |
| 1 | 2.507 | 2.233 | 2.191 | 2.161 | 24 |
| 2 | 2.485 | 2.265 | 2.203 | 2.159 | 38 |
| 3 | 2.481 | 2.218 | 2.252 | 2.094 | 35 |
| 4 | 2.446 | 2.380 | 2.096 | 2.075 | 43 |

- OOF RMSE gives realistic performance estimate.
- Test predictions = average of 5 folds.

---

## 📤 Submission  
- Predictions saved in required `ID,score` format.
- Averaged across all folds.
- Uploaded to Kaggle DA5401 competition leaderboard.

---

## ✅ Conclusion  
Using multilingual embeddings, engineered interaction features, synthetic augmentation, and a cross-validated MLP model produced stable performance despite dataset imbalance.  
The approach generalizes well and effectively estimates alignment between metrics and prompt–response pairs.

---

## 📎 Files in Repository  

