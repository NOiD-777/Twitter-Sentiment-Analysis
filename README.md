# 🐦 Twitter Sentiment Analysis Bot

This project is a machine learning-based sentiment analysis bot that analyzes tweets and classifies them as **positive** or **negative** using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

## 📂 File

- `tsa_ml.ipynb` – Jupyter/Colab notebook containing the entire pipeline from preprocessing to model training and evaluation.

## 🗃 Dataset

- Source: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- It contains 1.6 million tweets labeled as:
  - `0` – Negative sentiment
  - `4` – Positive sentiment

## ⚙️ Features

- CSV loading and data cleaning
- Text preprocessing (lowercasing, removing stopwords, punctuation, etc.)
- TF-IDF vectorization
- Logistic Regression model training
- Evaluation with metrics: accuracy, confusion matrix, classification report

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
2. Upload the dataset (`training.1600000.processed.noemoticon.csv`) in the same environment.
3. Open `tsa_ml.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
4. Follow the cells step-by-step to preprocess, train, and evaluate the model.

## 📊 Sample Output

> Example results might include:
>
> - Accuracy: `0.82`
> - Confusion Matrix:
>   ```
>   [[6500 1500]
>    [1300 6700]]
>   ```

## 🧠 Algorithms Used

- TF-IDF for feature extraction
- Logistic Regression for classification

## 🔧 TODO

- [ ] Add more classifiers (e.g., SVM, Random Forest)
- [ ] Add emoji sentiment handling
- [ ] Improve preprocessing with lemmatization
- [ ] Deploy with Flask or Streamlit

## 🧾 Requirements

If running locally, install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn


## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

