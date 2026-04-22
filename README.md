# FakeShield

FakeShield is a Streamlit-based fake news detection app that combines classical machine learning, BERT, domain credibility checks, Google News cross-referencing, and optional Google Fact Check API lookups.

## Features

- Logistic Regression and Naive Bayes text classifiers
- Optional BERT-based classifier
- Domain credibility scoring from URL input
- Google News cross-reference for article validation
- Google Fact Check API integration
- LIME explanations for word-level model reasoning

## Project Structure

- `app_final.py` - main Streamlit application
- `lr_model.pkl` - trained Logistic Regression model
- `nb_model.pkl` - trained Naive Bayes model
- `preprocessor.pkl` - text preprocessing pipeline
- `bert_model/` - optional local BERT model files
- `Dataset/` - sample training data used in development
- `FakeNewsDetection_Colab_Fixed (3).ipynb` - training notebook
- `FakeShield_NLP_Report_231168.pdf` - project report

## Requirements

Install dependencies with:

```bash
pip install streamlit scikit-learn nltk lime requests transformers torch matplotlib plotly joblib numpy
```

## Run Locally

```bash
streamlit run app_final.py
```

## Notes

- The app can work without the Google Fact Check API key, but that feature will remain disabled.
- BERT is optional; if the local model folder is not present, the app still runs with the other models.
- The file `bert_model/model.safetensors` is intentionally not tracked in Git because of GitHub size limits.
	Place it locally under `bert_model/` to enable BERT inference.