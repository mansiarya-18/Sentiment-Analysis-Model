# Sentiment Analysis Model

Capstone task: classify social media text as **Positive** or **Negative** using TF-IDF + Logistic Regression.

## Dataset
[Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) (Kaggle).
Download `sentiment_dataset.csv` from that page and place it in this repo's root folder.

## What's inside
- `sentiment_analysis.ipynb` — full pipeline: load → clean text → TF-IDF vectorize → train Logistic Regression → evaluate (accuracy, F1, confusion matrix) → test on 3 custom sentences → written summary.

## How to run
```bash
pip install scikit-learn nltk pandas
jupyter notebook sentiment_analysis.ipynb
```
Run all cells top to bottom. The dataset CSV must be in the same folder.

## Approach summary
Raw text is lowercased, URLs/mentions/punctuation stripped, and stopwords removed. Cleaned text is converted to TF-IDF vectors (top 5000 features) and fed into a Logistic Regression classifier (80/20 train-test split). Model is evaluated with Accuracy: 0.8571 and F1-score: 0.9231, then tested on 3 hand-written sentences.

## Limitation
Model uses word-frequency features only — no real understanding of context or sarcasm (e.g. "Oh great, another wasted day" can be misread as positive due to the word "great"). A larger dataset or transformer-based model would help.
