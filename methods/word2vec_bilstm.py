import pandas as pd
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import keras
from gensim.models import KeyedVectors
import os



word2vec_path = r'G:\Coding\PycharmProject\pusht_formal\glove_space\model\glove.twitter.27B.200d.txt'
w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def w2v_bilstm(df):
    df = df.dropna(subset=['processed'])

    df = df[['processed', 'label']]
    df['label'] = df['label'].map({'neu':0, 'pos':1, 'neg':2})

    df['processed'] =  [x.split() for x in df['processed']]
    X_train, X_test, y_train, y_test = train_test_split (df['processed'], df['label'] , test_size=0.2)

    words = set(w2v_model.index_to_key)
    word_to_idx = {word: i+1 for i, word in enumerate(words)}  # 0 is reserved for padding

    # Convert text to sequences of indices
    def text_to_sequence(text):
        return [word_to_idx.get(word, 0) for word in text if word in words]

    X_train_seq = [text_to_sequence(text) for text in X_train]
    X_test_seq = [text_to_sequence(text) for text in X_test]

    max_length = max(len(seq) for seq in X_train_seq)

    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    # Create embedding matrix
    vocab_size = len(word_to_idx) + 1  # +1 for padding token
    print(f'vocab size: {vocab_size}')
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_to_idx.items():
        embedding_matrix[i] = w2v_model[word]
    # print(embedding_matrix)

    # To stop the training of the model earlier if 3 consecutive loss stays the same (does not decrease)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # Step 4: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_padded, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[callback])

    # Step 6: Predict on test data
    y_pred = model.predict(X_test_padded)
    y_pred

    # Convert probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_labels

    # Convert y_pred_labels to a pandas Series
    y_pred_series = pd.Series(y_pred_labels, index=y_test.index)
    y_pred_series

    # step 8: results (word2vec (pretrained-googlenews) + bidirectional LSTM)
    # Tesing on original_df dataset

    # {'neu':0, 'pos':1, 'neg':2}
    target_names = ["neu", "pos", "neg"]

    report_dict = classification_report(y_test, y_pred_series, target_names=target_names, output_dict=True)

    # Convert the report dictionary into a DataFrame for better readability
    report_df = pd.DataFrame(report_dict).transpose()

    # Create an empty row with NaN values
    empty_row = pd.DataFrame([[" "] * len(report_df.columns)], columns=report_df.columns)

    # Insert the empty row after 'neg' to separate the row for readability
    report_df = pd.concat([report_df.loc[:'neg'], empty_row, report_df.loc['accuracy':]], ignore_index=False)

    report_df.index.values[3] = ' '

    # Display the summary
    print("")
    print(report_df)