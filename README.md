
# Twitter Sentiment Analysis

## Project Overview
This project performs sentiment analysis on Twitter data using the Sentiment140 dataset. It predicts whether a tweet has a **positive** or **negative** sentiment. The project leverages natural language processing (NLP) techniques and machine learning models to classify the tweets.

---

## Dataset
The [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset contains **1.6 million tweets**, each labeled with:
- **`0`**: Negative sentiment
- **`1`**: Positive sentiment

The dataset includes the following columns:
- `target`: Sentiment label (0 = Negative, 1 = Positive)
- `id`: Tweet ID
- `date`: Date the tweet was posted
- `flag`: Query flag
- `user`: Username of the author
- `text`: The tweet content

---

## Libraries Used
- `numpy`: For numerical operations
- `pandas`: For data manipulation and processing
- `nltk`: For natural language processing tasks
- `scikit-learn`: For machine learning algorithms and utilities
- `re`: For regular expressions
- `pickle`: For saving the trained model

---

## Data Preprocessing
1. **Loading the Data**: The dataset is loaded into a pandas DataFrame, and relevant columns are selected.
2. **Handling Missing Values**: Checked for missing values; none were found.
3. **Label Encoding**: The target sentiment labels were encoded with `0` for negative sentiment and `1` for positive sentiment.
4. **Text Preprocessing**:
   - Removed non-alphabetic characters.
   - Converted text to lowercase.
   - Removed stopwords (commonly occurring words that do not contribute to sentiment).
   - Applied stemming to reduce words to their root form.

---

## Model Training
1. **Feature Extraction**: 
   - Used `TfidfVectorizer` to convert the tweet text into numerical vectors, which is the input for the machine learning model.
2. **Model Choice**: 
   - Used **Logistic Regression** as the classification model.
   - The model was trained on 80% of the data and tested on the remaining 20%.
3. **Evaluation**: 
   - The model achieved an accuracy of **73.65%** on the test data.

---

## Results
- The model performs well with a decent accuracy of 73.65% on the sentiment classification task.
- **Further improvements** can be made by exploring more complex models or fine-tuning hyperparameters.

---

## Model Saving
The trained Logistic Regression model was saved using `pickle` for later use in making predictions on new tweets.

```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```
---
## Future Work
- Implement deep learning models such as RNNs or LSTMs to improve accuracy.
- Explore hyperparameter tuning for better performance.
- Analyze other sentiment classes beyond just positive and negative.

---

## Conclusion
This project demonstrates a simple but effective approach to sentiment analysis on Twitter data. The accuracy of 73.65% shows promising results, and future work can focus on improving this through advanced techniques.

