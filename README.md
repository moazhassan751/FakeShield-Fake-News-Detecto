# FakeShield

<p align="center">
	<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit badge">
	<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge">
	<img src="https://img.shields.io/badge/License-MIT-111827?style=for-the-badge" alt="License badge">
</p>

FakeShield is a Streamlit-based fake news detection app that combines classical machine learning, optional BERT scoring, domain credibility checks, Google News cross-referencing, and Google Fact Check API lookups.

## Highlights

- Logistic Regression and Naive Bayes text classifiers
- Optional BERT-based classifier for extra signal
- Domain credibility scoring from article URLs
- Google News cross-reference for live coverage checks
- Google Fact Check API integration for claim lookup
- LIME explanations for word-level model reasoning

## How It Works

1. You paste an article and optionally add a URL.
2. The text is cleaned with the saved preprocessing pipeline.
3. Logistic Regression, Naive Bayes, and optional BERT generate predictions.
4. The URL is checked against trusted and suspicious domain patterns.
5. Google News cross-reference checks whether the story appears in credible sources.
6. If you provide a Google Fact Check API key, the app searches for matching claims.
7. LIME highlights the words that most influenced the final result.

## Repository Contents

- `app_final.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - project overview and usage guide
- `lr_model.pkl` - trained Logistic Regression model
- `nb_model.pkl` - trained Naive Bayes model
- `preprocessor.pkl` - text preprocessing pipeline
- `bert_model/` - optional local BERT model files
- `Dataset/` - training datasets used in development
- `FakeNewsDetection_Colab_Fixed (3).ipynb` - training and experimentation notebook
- `FakeShield_NLP_Report_231168.pdf` - project report

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app_final.py
```

## Notes

- The app can work without the Google Fact Check API key, but that feature will remain disabled.
- BERT is optional; if the local model folder is not present, the app still runs with the other models.
- The file `bert_model/model.safetensors` is intentionally not tracked in Git because of GitHub size limits.
	Place it locally under `bert_model/` to enable BERT inference.