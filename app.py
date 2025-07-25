# --- Streamlit Version of Mental Health Post Analyzer (With Upload + Navigation + Objectives Reflected) ---

import streamlit as st
import sqlite3
import pandas as pd
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import nltk
nltk.download('punkt')

# --- Database Setup ---
conn = sqlite3.connect('users.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    username TEXT,
    date TEXT,
    sentiment REAL,
    keywords TEXT,
    flagged INTEGER,
    recommendation TEXT
)''')
conn.commit()

# --- Analyzer Functions ---
def analyze_posts(posts):
    all_text = " ".join(posts)
    keywords = Counter(all_text.split()).most_common(5)
    sentiments = [TextBlob(post).sentiment.polarity for post in posts]
    average_sentiment = sum(sentiments) / len(sentiments)
    flagged = average_sentiment < -0.2
    recommendation = "Consider reaching out for support." if flagged else "Keep monitoring your mood."
    return average_sentiment, keywords, flagged, recommendation, sentiments

# --- Chart Function ---
def show_sentiment_chart(timestamps, sentiments):
    dates = [datetime.strptime(t, "%Y-%m-%d") for t in timestamps]
    fig, ax = plt.subplots()
    ax.plot(dates, sentiments, marker='o', linestyle='-')
    ax.set_title('Sentiment Trend Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Polarity')
    ax.grid(True)
    st.pyplot(fig)

# --- Save to Logs ---
def log_analysis(username, sentiment, keywords, flagged, recommendation):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    keyword_str = ", ".join([kw[0] for kw in keywords])
    cursor.execute('''INSERT INTO logs (username, date, sentiment, keywords, flagged, recommendation)
                      VALUES (?, ?, ?, ?, ?, ?)''',
                   (username, date, sentiment, keyword_str, int(flagged), recommendation))
    conn.commit()

# --- User Authentication ---
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone() is not None

# --- Streamlit App ---
st.set_page_config(page_title="Mental Health Post Analyzer", layout="centered")
st.title("🧠 Mental Health Post Analyzer")

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Navigation Menu ---
if st.session_state.logged_in:
    menu = ["Home", "Prediction", "History", "Logout"]
else:
    menu = ["Login", "Register"]

choice = st.sidebar.selectbox("Menu", menu)

# --- Registration Page ---
if choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Account created successfully! Go to Login.")
        else:
            st.error("Username already exists.")

# --- Login Page ---
elif choice == "Login":
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

# --- Home Page (Aim & Objectives) ---
elif choice == "Home":
    st.subheader("🎯 Aim and Objectives of Study")
    st.markdown("""
    **Aim:**  
    The aim of this study is to assess and compare data mining techniques in detecting early signs of mental health issues from social media activities.

    **Objectives:**  
    1. To extract and preprocess mental health-relevant data from social media platforms.  
    2. To apply and evaluate data mining classification techniques such as Logistic Regression, Naïve Bayes, SVM, Random Forest, Gradient Boosting for identifying early warning signs.  
    3. To compare the performance of these techniques in terms of accuracy, precision, recall, F1-score, and ROC-AUC.  
    4. To recommend the most effective data mining technique for early detection of mental health issues via social media activities.
    """)

# --- Prediction Page ---
elif choice == "Prediction":
    st.subheader("📊 Analyze Your Mental Health Posts")

    uploaded_file = st.file_uploader("📄 Upload CSV File (with 'username', 'post_text', 'timestamp')", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df.head())

        if 'username' not in df.columns or 'post_text' not in df.columns or 'timestamp' not in df.columns:
            st.error("❌ CSV must contain columns: 'username', 'post_text', 'timestamp'")
        else:
            user_data = df[df['username'] == st.session_state.username]
            user_posts = user_data['post_text'].tolist()
            timestamps = user_data['timestamp'].tolist()
            
            if st.button("Run Analysis"):
                if not user_posts:
                    st.warning("No posts found for this user.")
                else:
                    # Reflect Objective 2
                    st.info("Applying basic NLP model (TextBlob) for sentiment classification...")

                    sentiment, keywords, flagged, recommendation, sentiments = analyze_posts(user_posts)

                    # Reflect Objective 1
                    st.write(f"**Step 1 (Extracted Posts):** {len(user_posts)} posts found.")
                    st.write(f"**Step 2 (Sentiment Score):** {sentiment:.2f}")
                    st.write(f"**Step 3 (Top Keywords):** {', '.join([kw[0] for kw in keywords])}")
                    st.write(f"**Step 4 (Flagged):** {'Yes' if flagged else 'No'}")
                    st.write(f"**Step 5 (Recommendation):** {recommendation}")

                    # Reflect Objective 3 (Evaluation Placeholder)
                    st.markdown("**🔍 Model Evaluation (Placeholder):**")
                    st.table(pd.DataFrame({
                        "Model": ["TextBlob (baseline)"],
                        "Accuracy": ["N/A"],
                        "Precision": ["N/A"],
                        "Recall": ["N/A"],
                        "F1-Score": ["N/A"],
                        "ROC-AUC": ["N/A"]
                    }))

                    # Reflect Objective 4
                    st.success("✔️ Based on the sentiment analysis, the model has provided a recommendation above.")

                    show_sentiment_chart(timestamps, sentiments)
                    log_analysis(st.session_state.username, sentiment, keywords, flagged, recommendation)

                    if st.button("Export Result"):
                        with open("analysis_result.txt", "w") as file:
                            file.write(f"User: {st.session_state.username}\n")
                            file.write(f"Sentiment Score: {sentiment:.2f}\n")
                            file.write(f"Top Keywords: {', '.join([kw[0] for kw in keywords])}\n")
                            file.write(f"Flagged: {'Yes' if flagged else 'No'}\n")
                            file.write(f"Recommendation: {recommendation}\n")
                        st.success("Result exported to analysis_result.txt")

# --- History Page ---
elif choice == "History":
    st.subheader("📜 Analysis History")
    cursor.execute("SELECT * FROM logs WHERE username=?", (st.session_state.username,))
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Username", "Date", "Sentiment", "Keywords", "Flagged", "Recommendation"])
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="history.csv")
    else:
        st.info("No analysis history found.")

# --- Logout Page ---
elif choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("You have been logged out.")
    st.experimental_rerun()
