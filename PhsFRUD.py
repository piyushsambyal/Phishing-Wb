import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import re
import os
import logging
from collections import Counter
import tldextract

st.set_page_config(layout="centered", page_title="Enhanced Phishing URL Detector")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    <h1 class='title'>🔐 Enhanced Phishing URL Detector</h1>
    <p class='subtitle'>Advanced detection of phishing threats with improved accuracy!</p>
""", unsafe_allow_html=True)

def extract_features(url):
    try:
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed_url.hostname or ''
        path = parsed_url.path or ''
        query = parsed_url.query or ''
        extracted = tldextract.extract(url)
        domain = extracted.domain
        suffix = extracted.suffix
        url_length = len(url)
        subdomains = hostname.split('.')
        num_subdomains = len(subdomains) - 2 if hostname and len(subdomains) > 1 else 0
        suspicious_keywords = ['paypal', 'login', 'secure', 'webscr', 'signin', 'account', 'verify', 'update', 'processing', 'bank']
        has_suspicious = int(any(keyword in url.lower() for keyword in suspicious_keywords))
        special_chars = len(re.findall(r'[@?=&#]', url))
        has_https = int(parsed_url.scheme == 'https')
        has_ip = int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname)))
        tld_length = len(suffix) if suffix else 0
        path_depth = len(path.split('/')) - 1 if path else 0
        digit_to_letter_ratio = len(re.findall(r'\d', url)) / (len(re.findall(r'[a-zA-Z]', url)) + 1)
        url_shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl']
        is_shortened = int(any(shortener in hostname for shortener in url_shorteners))
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        domain_entropy = -sum((count/len(domain)) * np.log2(count/len(domain)) for count in Counter(domain).values()) if domain else 0
        query_length = len(query)
        return [
            url_length, num_subdomains, has_suspicious, special_chars, has_https,
            has_ip, tld_length, path_depth, digit_to_letter_ratio, is_shortened,
            num_dots, num_hyphens, domain_entropy, query_length
        ]
    except Exception as e:
        logger.error(f"Error extracting features for URL {url}: {e}")
        return [0] * 14

@st.cache_data
def load_data():
    try:
        data = []
        for i in range(2, 1001):
            url_key = f'row{i}'
            if url_key in globals() and isinstance(globals()[url_key], str):
                parts = globals()[url_key].split(',')
                if len(parts) == 2:
                    url, label = parts
                    data.append({'URL': url.strip(), 'Label': 1 if label.strip().lower() == 'bad' else 0})
        df_phishing = pd.DataFrame(data)
        if df_phishing.empty:
            raise ValueError("No valid data extracted from provided dataset")
        legit_urls = [
            'google.com', 'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
            'microsoft.com', 'apple.com', 'wikipedia.org', 'youtube.com', 'github.com',
            'nytimes.com', 'bbc.com', 'cnn.com', 'reddit.com', 'stackoverflow.com'
        ] * 67
        df_legit = pd.DataFrame({'URL': legit_urls[:999], 'Label': 0})
        df_combined = pd.concat([df_phishing, df_legit], ignore_index=True)
        df_combined = df_combined.dropna(subset=['URL'])
        df_combined = df_combined[df_combined['URL'].str.strip() != '']
        df_combined['Features'] = df_combined['URL'].apply(extract_features)
        feature_columns = [
            'url_length', 'num_subdomains', 'has_suspicious', 'special_chars', 'has_https',
            'has_ip', 'tld_length', 'path_depth', 'digit_to_letter_ratio', 'is_shortened',
            'num_dots', 'num_hyphens', 'domain_entropy', 'query_length'
        ]
        feature_df = pd.DataFrame(df_combined['Features'].tolist(), columns=feature_columns)
        df_combined = pd.concat([df_combined[['URL', 'Label']], feature_df], axis=1)
        logger.info(f"Dataset loaded with shape: {df_combined.shape}")
        return df_combined
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    try:
        if df.empty or 'url_length' not in df.columns:
            raise ValueError("Invalid dataset for training")
        feature_columns = [
            'url_length', 'num_subdomains', 'has_suspicious', 'special_chars', 'has_https',
            'has_ip', 'tld_length', 'path_depth', 'digit_to_letter_ratio', 'is_shortened',
            'num_dots', 'num_hyphens', 'domain_entropy', 'query_length'
        ]
        X = df[feature_columns]
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test accuracy: {accuracy:.2f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        return best_model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"Error training model: {e}")
        return None

def main():
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! Refresh the app.")
    df = load_data()
    if df.empty:
        st.error("Failed to load dataset. Please check the dataset and try again.")
        return
    model = train_model(df)
    if model is None:
        st.error("Failed to train model. Please check the dataset and try again.")
        return
    whitelist = {
        'google.com', 'maps.google.com', 'amazon.com', 'facebook.com', 'twitter.com',
        'linkedin.com', 'microsoft.com', 'apple.com', 'wikipedia.org', 'youtube.com',
        'github.com', 'nytimes.com', 'bbc.com', 'cnn.com', 'reddit.com', 'stackoverflow.com'
    }
    user_url = st.text_input("Enter URL to Analyze", placeholder="e.g., https://example.com")
    if st.button("Predict"):
        if user_url:
            with st.spinner("Analyzing URL..."):
                features = extract_features(user_url)
                feature_columns = [
                    'url_length', 'num_subdomains', 'has_suspicious', 'special_chars', 'has_https',
                    'has_ip', 'tld_length', 'path_depth', 'digit_to_letter_ratio', 'is_shortened',
                    'num_dots', 'num_hyphens', 'domain_entropy', 'query_length'
                ]
                features_df = pd.DataFrame([features], columns=feature_columns)
                try:
                    prediction = model.predict(features_df)[0]
                    probability = model.predict_proba(features_df)[0][1]
                    parsed_url = urlparse(user_url)
                    hostname = parsed_url.hostname or ''
                    is_whitelisted = any(domain in hostname for domain in whitelist)
                    if is_whitelisted and prediction == 1 and probability < 0.9:
                        prediction = 0
                        probability = 0.1
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        result = "Phishing" if prediction == 1 else "Legitimate"
                        color = "#e74c3c" if prediction == 1 else "#2ecc71"
                        st.markdown(f"""
                            <div style='padding: 20px 30px; background-color: {color}; color: white; border-radius: 12px; text-align: center;'>
                                <h3 style='margin: 0;'>🔍 Prediction: {result}</h3>
                                <p style='margin: 5px 0 0;'>Phishing Probability: {probability:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if is_whitelisted and prediction == 0:
                            st.write("**Note:** Overridden as a known legitimate domain.")
                    with col2:
                        with st.expander("View Extracted Features"):
                            st.markdown("""
                                <div style='background-color: #f9f9f9; padding: 15px 20px; border-radius: 10px;'>
                            """, unsafe_allow_html=True)
                            feature_names = [
                                'URL Length', 'Number of Subdomains', 'Suspicious Keywords Present',
                                'Special Characters Count', 'HTTPS Present', 'IP Address Present',
                                'TLD Length', 'Path Depth', 'Digit to Letter Ratio', 'URL Shortener Used',
                                'Number of Dots', 'Number of Hyphens', 'Domain Entropy', 'Query Length'
                            ]
                            for name, value in zip(feature_names, features):
                                if name in ['Suspicious Keywords Present', 'HTTPS Present', 'IP Address Present', 'URL Shortener Used']:
                                    value = 'Yes' if value else 'No'
                                elif isinstance(value, float):
                                    value = f"{value:.2f}"
                                st.write(f"- {name}: {value}")
                            st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Error making prediction: {e}")
                    st.error(f"Error making prediction: {e}")
        else:
            st.warning("Please enter a URL.")

if __name__ == "__main__":
    main()