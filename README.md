# Sentiment Analysis with Logistic Regression

This project performs sentiment analysis on tweets using a Logistic Regression model. The model predicts whether a tweet expresses a **positive** sentiment (`1`) or a **negative** sentiment (`0`).

---

## Features

- **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M labeled tweets)
- **Preprocessing**:
  - Text cleaning, lowercasing, stopword removal, and stemming
  - TF-IDF vectorization for feature extraction
- **Model**: Logistic Regression with `max_iter=1000`

---

## How to Use

1. Clone the repository and install dependencies:
   ```bash
   pip install numpy pandas nltk scikit-learn
   ```
2. Place the training.1600000.processed.noemoticon.csv dataset in the working directory.

3. Run the script to preprocess data, train the model, and evaluate performance.

4. Save the trained model:
   ```python
   import pickle
   pickle.dump(model, open('model.pkl', 'wb'))
   ```

## Results

The model achieves high accuracy in predicting tweet sentiment. Performance metrics are printed after evaluation.
