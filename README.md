# NLP-ON-YASSIR-FEEDBACKS

This project applies advanced NLP techniques to analyze and model customer feedback from the Yassir platform. The analysis includes sentiment classification, topic modeling (LDA), clustering, wordclouds, statistical visualizations, and deep learning (LSTM and DziriBERT).

---

## 📌 Features

- Sentiment distribution by rating and language
- Wordclouds per sentiment category
- Text length analysis
- Topic modeling using LDA
- KMeans clustering with TF-IDF
- Bigram frequency analysis
- Chi-squared feature selection
- Logistic Regression classification
- LSTM-based deep learning for sentiment prediction
- t-SNE visualization
- BERT-based sentiment analysis using [DziriBERT](https://huggingface.co/asafaya/dziribert)

---

## 📁 Dataset

> **Note:** The dataset used in this project is **private** and cannot be shared due to confidentiality restrictions.

It contains multilingual product reviews with columns such as:

- `Review_Text`: Raw review text
- `Sentiment`: Sentiment label (e.g., Positive, Negative)
- `Language`: Language of the review
- `Aspect_Focus`: Key aspect of the review
- `Rating (1-5)`: Numerical customer rating
- `Clean_Text` and `Filtered_Text`: Preprocessed versions used in modeling

---

## 📊 Visualizations & Explorations

- Histograms, boxplots, and countplots using `matplotlib` and `seaborn`
- Correlation matrix
- t-SNE for feature space visualization
- Wordclouds for sentiment categories

---

## 🧠 Models

- **Logistic Regression**: Simple baseline classifier
- **LSTM Model**: Deep learning model trained on tokenized and padded sequences
- **DziriBERT Fine-Tuning**: HuggingFace `Trainer` API used to fine-tune DziriBERT on the dataset

---


