import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

# Load dataset
data = pd.read_csv('/data_reviews.csv')
data = data.drop(columns=['Unnamed: 0', 'Country'])

# Rename for consistency
data_review = data.copy()

# Check missing values
print(data_review.isna().sum())

# Distribution plots
sns.histplot(data_review['Rating (1-5)'], bins=5, kde=True)
plt.title('Distribution of Ratings')
plt.show()

sns.countplot(x='Language', data=data_review)
plt.title('Count by Language')
plt.show()

sns.boxplot(x='Language', y='Rating (1-5)', data=data_review)
plt.title('Ratings by Language')
plt.show()

# Heatmap
sns.heatmap(data_review.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Count of reviews per Aspect_Focus
plt.figure(figsize=(12, 6))
sns.countplot(y='Aspect_Focus', data=data_review, order=data_review['Aspect_Focus'].value_counts().index)
plt.title("Nombre d'Avis par Aspect_Focus")
plt.show()

# Sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=data_review, palette="viridis")
plt.title("Répartition des Sentiments")
plt.show()

# WordCloud by Sentiment
for sentiment in data_review['Sentiment'].unique():
    text = " ".join(data_review[data_review['Sentiment'] == sentiment]['Review_Text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Nuage de Mots pour le Sentiment : {sentiment}")
    plt.show()

# Text length by sentiment
data_review['text_length'] = data_review['Review_Text'].astype(str).apply(len)
avg_text_length = data_review.groupby('Sentiment')['text_length'].mean()
avg_text_length.plot(kind='bar', color='coral')
plt.title("Longueur Moyenne des Textes par Sentiment")
plt.show()

# Clean label
data_review['Sentiment'] = data_review['Sentiment'].replace({'Positif': 'Positive'})

# LDA
vectorizer = CountVectorizer(stop_words='english')
text_matrix = vectorizer.fit_transform(data_review['Clean_Text'].dropna())
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(text_matrix)
terms = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda_model.components_):
    print(f"Sujet {idx+1}: {[terms[i] for i in topic.argsort()[-10:]]}\n")

# Clustering
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data_review['Clean_Text'].fillna(''))
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(tfidf_matrix)
data_review['Cluster'] = kmeans.labels_

# Top bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = bigram_vectorizer.fit_transform(data_review['Filtered_Text'].fillna(''))
bigram_counts = bigram_matrix.sum(axis=0).A1
bigram_features = bigram_vectorizer.get_feature_names_out()
bigram_freq = sorted(zip(bigram_features, bigram_counts), key=lambda x: x[1], reverse=True)[:10]
bigram_words, bigram_counts = zip(*bigram_freq)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(bigram_words), y=list(bigram_counts), palette='coolwarm')
plt.title("Top 10 Bigrammes Après Filtrage")
plt.xticks(rotation=45)
plt.show()

# Chi2 + Correlation
data_review['Sentiment_Numeric'] = data_review['Sentiment'].factorize()[0]
X = tfidf_matrix
chi2_scores, _ = chi2(X, data_review['Sentiment_Numeric'])
chi2_results = pd.Series(chi2_scores, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)
print("Top 10 mots associés au sentiment:\n", chi2_results.head(10))

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, data_review['Sentiment_Numeric'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X.toarray())
data_review['x_tsne'], data_review['y_tsne'] = X_reduced[:, 0], X_reduced[:, 1]
sns.scatterplot(data=data_review, x='x_tsne', y='y_tsne', hue='Sentiment', palette='coolwarm')
plt.title("t-SNE Visualization")
plt.show()

# Tokenization for LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_review['Filtered_Text'].fillna(''))
sequences = tokenizer.texts_to_sequences(data_review['Filtered_Text'])
X_seq = pad_sequences(sequences, maxlen=100, padding='post')
encoder = LabelEncoder()
y_seq = encoder.fit_transform(data_review['Sentiment'])
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# LSTM Model
vocab_size = len(tokenizer.word_index) + 1
model_lstm = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_seq)), activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('french') + stopwords.words('english') + stopwords.words('arabic'))

def remove_stop_words(text, stop_words):
    if isinstance(text, str):
        words = nltk.word_tokenize(text)
        return " ".join([w for w in words if w.lower() not in stop_words])
    return text

data_review['Filtered_Text'] = data_review['Clean_Text'].apply(lambda x: remove_stop_words(x, stop_words)).astype(str)

# ✅ Sentiment Prediction with DZIRIBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load DZIRIBERT
model_name = "boudinfl/dziribert-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dziribert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
dziribert_pipe = pipeline("sentiment-analysis", model=dziribert_model, tokenizer=tokenizer)

# Predict sentiment for test reviews (first 10 for demo)
test_texts = data_review['Filtered_Text'].dropna().sample(10, random_state=42).tolist()
dziribert_results = dziribert_pipe(test_texts)

# Print predictions
for text, result in zip(test_texts, dziribert_results):
    print(f"Texte : {text[:100]}...\nSentiment prédit : {result['label']}, Score : {round(result['score'], 2)}\n")
