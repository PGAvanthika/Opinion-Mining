Overview

This project performs opinion mining (sentiment analysis) on textual data, analyzing sentiments in food-related reviews. It preprocesses the text, converts words to numerical representations, and applies machine learning techniques for classification.

Project Structure

├── food_analysis_code.txt  # Main script containing the code logic
├── main.py                 # Entry point for executing the sentiment analysis
├── preprocess.py           # Preprocessing script for text cleaning and transformation
├── vocab2int.pickle        # Dictionary mapping words to integers for model input

Requirements

Python 3.x

Pandas

NumPy

Scikit-learn

NLTK

Pickle

You can install the required dependencies using:

pip install pandas numpy scikit-learn nltk

Usage

Preprocessing

Run preprocess.py to clean and preprocess the dataset.

It tokenizes text, removes stopwords, and converts words to their integer representation.

Running Sentiment Analysis

Execute main.py to perform sentiment classification using the preprocessed data.

The script loads the vocabulary mapping from vocab2int.pickle.

python main.py

Expected Output

The model classifies reviews into positive, neutral, or negative sentiments.

The results are displayed on the console or saved as an output file (optional).

Dataset

The dataset for this project was sourced from Kaggle. You can access it here:
Dataset Link

Notes

food_analysis_code.txt contains the core logic, which can be integrated into main.py.

Ensure vocab2int.pickle is available for mapping words to integers.

Future Enhancements

Implement deep learning models (e.g., LSTMs) for better accuracy.

Add a web interface for user input and sentiment analysis visualization.

Expand dataset and fine-tune preprocessing for improved performance.
