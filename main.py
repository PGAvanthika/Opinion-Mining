import os
import numpy as np
import pandas as pd
import pickle
from keras.losses import MeanSquaredError
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# After creating vocab2int
vocab2int = create_vocab_with_pos(df)

# Define vocab_size and embedding_size
vocab_size = len(vocab2int) + 1  # +1 for the <UNK> token
embedding_size = 100  # You can adjust this value based on your needs

with open("/content/data/vocab2int.pickle", "wb") as f:
    pickle.dump(vocab2int, f)

print("Vocabulary with POS tags saved.")

# Continue with your existing code...

# Check if the model is already saved
model_path = '/content/data/review_model.h5'

if not os.path.exists(model_path):
    # Load your dataset
    df = pd.read_csv('/content/data/Reviewsfull.csv')  # Update the path to your dataset

    # Preprocess the reviews and tokenize them
    X = np.zeros((len(df),), dtype=object)
    y = df['Score'].values  # Assuming the scores are in a column named 'Score'

    for i in range(len(df)):
        X[i] = tokenize_words(clean_text(df['Text'].iloc[i]), vocab2int)

    X = pad_sequences(X, maxlen=sequence_length, padding='post')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the model
    model = get_model(vocab_size, sequence_length, embedding_size)
    model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

    # Train the model
    epochs = 2  # You can adjust this based on your need
    batch_size = 64
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(model_path)
    print("Model trained and saved.")
else:
    # Load the model
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    print("Model loaded from disk.")


def evaluate_review(review):
    review = tokenize_words(clean_text(review), vocab2int)
    x = pad_sequences([review], maxlen=sequence_length)
    prediction = model.predict(x)[0][0]
    return prediction

sample_review = input('Please input your review:\n')
predicted_score = evaluate_review(sample_review)

if predicted_score > 3.75:
    print("Positive review")
elif 2 < predicted_score <= 3.75:
    print("Neutral review")
else:
    print("Negative review")

print(f"Predicted Score: {predicted_score:.2f}/5")
