import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

class SentimentAnalysisModel:
    def __init__(self, glove_path, max_epochs=50, batch_size=32):
        self.glove_path = glove_path
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.glove_embeddings = {}
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_length = 0
        self.vocab_size = 0
        self.embedding_matrix = None

    def load_glove_embeddings(self):
        """Load GloVe embeddings into a dictionary."""
        with open(self.glove_path, "r") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                self.glove_embeddings[word] = vector
        print("GloVe embeddings loaded.")

    def preprocess_data(self, df):
        """Tokenize and pad sequences, and prepare embedding matrix."""
        df['processed'] = df['processed'].apply(str)
        self.max_length = df['processed'].apply(lambda x: len(x.split())).max()
        self.tokenizer.fit_on_texts(df['processed'])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        encoded_tweets = self.tokenizer.texts_to_sequences(df['processed'])
        padded_tweets = pad_sequences(encoded_tweets, maxlen=self.max_length, padding='post')

        # Create the embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 50))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.glove_embeddings.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        print("Data preprocessing complete.")
        return padded_tweets, df['label']

    def build_model(self):
        """Build the BiLSTM model."""
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size,
                      output_dim=50,
                      weights=[self.embedding_matrix],
                      input_length=self.max_length,
                      trainable=False),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(64)),
            Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model built and compiled.")

    def train_model(self, x_train, y_train):
        """Train the model."""
        callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[callback]
        )
        print("Model training complete.")
        return history

    def evaluate_model(self, x_test, y_test):
        """Evaluate the model and print metrics."""
        predictions = self.model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        report = classification_report(y_test, predictions)

        print("Evaluation Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Classification Report:")
        print(report)

    def run_pipeline(self, df):
        """Full pipeline: preprocess, split, build, train, and evaluate."""
        self.load_glove_embeddings()
        padded_tweets, labels = self.preprocess_data(df)
        x_train, x_test, y_train, y_test = train_test_split(padded_tweets, labels, test_size=0.2, stratify=labels)
        self.build_model()
        self.train_model(x_train, y_train)
        self.evaluate_model(x_test, y_test)
