# 🧠 BBC News Text Classification using RNN

This project is a Natural Language Processing (NLP) application that classifies news articles from the BBC dataset into one of five categories: **business**, **entertainment**, **politics**, **sport**, or **tech** using a **Recurrent Neural Network (RNN)** built with TensorFlow/Keras.

---

## 📂 Dataset

- **Source**: [BBC Text Classification Dataset](https://www.kaggle.com/datasets/cpmarket/bbc-news)
- **Structure**:
  - `category`: The label (e.g., sport, business, etc.)
  - `text`: The article content

---

## 🧰 Tools & Libraries

- Python 🐍
- TensorFlow / Keras
- NLTK (for stopword removal)
- Scikit-learn (Label Encoding, Train/Test Split, Evaluation)
- Seaborn & Matplotlib (Visualization)

---

## 🧼 Data Preprocessing

✅ The following steps were performed:
- Lowercasing all text
- Stopword removal using NLTK
- Tokenization using Keras Tokenizer
- Sequence padding (post-padding)
- Label encoding of target classes

---

## 🧠 Model Architecture

| Layer         | Description                                      |
|---------------|--------------------------------------------------|
| Embedding     | Converts word indices into dense vectors         |
| Dropout       | Reduces overfitting                              |
| SimpleRNN ×2  | Learns sequential word dependencies              |
| Dense (ReLU)  | Fully connected hidden layer                     |
| Dense (Softmax)| Final output layer with 5-class probabilities   |

```python
Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    Dropout(0.2),
    SimpleRNN(units=64, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(units=64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])


Loss Function: sparse_categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

Epochs: 20 (can be tuned)

Batch Size: 32

Train/Test Split: 80/20

📈 Evaluation
Accuracy on test set: 🟢 68 %

Classification Report:

Precision, Recall, and F1-score per class

Confusion Matrix:

Visualized with Seaborn heatmap




<img width="652" height="691" alt="لقطة شاشة 2025-07-13 071349" src="https://github.com/user-attachments/assets/4607f0be-2956-430b-9b01-57726656de90" />



<img width="681" height="564" alt="لقطة شاشة 2025-07-13 071401" src="https://github.com/user-attachments/assets/5e7a8584-5c04-43fa-b6f2-32c17450fe3b" />
