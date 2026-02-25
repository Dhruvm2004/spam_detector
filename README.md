# Email/SMS Spam Classifier

A Machine Learning-based web application that classifies messages as **Spam** or **Not Spam** using Natural Language Processing and Multinomial Naive Bayes.

🌐 **Live App:**  
👉 https://spamdetector-mn.streamlit.app/

---

## Features

- Text preprocessing:
  - Lowercasing
  - Tokenization (NLTK)
  - Stopword removal
  - Stemming (Porter Stemmer)
- TF-IDF Vectorization (3000 features)
- Multinomial Naive Bayes classifier
- Real-time predictions
- Confidence score display
- Deployed using Streamlit Cloud

---

## Machine Learning Pipeline

1. Text Cleaning
2. Tokenization
3. Stopword Removal
4. Stemming
5. TF-IDF Vectorization
6. Multinomial Naive Bayes Classification

---

## Model Performance

- Accuracy: ~97%
- High precision in spam detection
- Dataset: SMS Spam Collection Dataset

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy

---

##  Project Structure
spam_detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

---

##  Run Locally

1. Clone the repository

```bash
git clone https://github.com/Dhruvm2004/spam_detector.git
cd spam_detector

2. Install dependencies
pip install -r requirements.txt
3️. Run the application
streamlit run app.py


