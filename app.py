import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detection App")
st.write("Check whether an email is **Spam** or **Not Spam**")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gmail_spam_dataset_5000_rows.csv")

    return df

df = load_data()

# -------------------------------
# Prepare data
# -------------------------------
df['text'] = df['text'].str.lower()

X_text = df['text']
y = df['label']

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=6000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

# -------------------------------
# Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# User input
# -------------------------------
st.subheader("✉️ Enter Email Text")
email_input = st.text_area("Paste email content here")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter email text")
    else:
        email_vector = vectorizer.transform([email_input.lower()])
        spam_prob = model.predict_proba(email_vector)[0][1]

        # 🔥 Decide using probability (not predict)
        if spam_prob > 0.5:
            st.error(f"🚨 This email is SPAM\n\nSpam Probability: {spam_prob:.2f}")
        else:
            st.success(f"✅ This email is NOT SPAM\n\nSpam Probability: {spam_prob:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using TF-IDF & Logistic Regression | ML Project")

