import streamlit as st
import pandas as pd
import sqlite3
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import re

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS analysis_logs (username TEXT, sentiment REAL, keywords TEXT, flagged INTEGER, recommendation TEXT, timestamp TEXT)''')
conn.commit()

# ---------- UTILITY FUNCTIONS ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    hashed = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def authenticate_user(username, password):
    hashed = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed))
    return c.fetchone() is not None

def extract_keywords(texts):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)
    keywords = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    return sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

def analyze_posts(posts):
    sentiments = [TextBlob(post).sentiment.polarity for post in posts]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    keywords = extract_keywords(posts)
    flagged = avg_sentiment < -0.2
    recommendation = "Needs urgent attention" if flagged else "Monitor periodically"
    return avg_sentiment, keywords, flagged, recommendation, sentiments

def log_analysis(username, sentiment, keywords, flagged, recommendation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO analysis_logs VALUES (?, ?, ?, ?, ?, ?)",
              (username, sentiment, ','.join([kw[0] for kw in keywords]), int(flagged), recommendation, timestamp))
    conn.commit()

def show_sentiment_chart(user, timestamps, sentiments):
    fig, ax = plt.subplots()
    ax.plot(timestamps, sentiments, marker='o')
    ax.set_title(f"Sentiment Over Time for {user}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Sentiment")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------- STREAMLIT APP ----------
st.title("🧠 Mental Health Post Analyzer")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

menu = ["Login", "Register"] if not st.session_state.logged_in else ["Home", "Prediction", "History", "Logout"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type='password')
    if st.button("Register"):
        if register_user(new_user, new_pass):
            st.success("Account created! Please login.")
        else:
            st.error("Username already exists.")

elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials.")

elif choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully.")
    st.experimental_rerun()

elif choice == "Home":
    st.subheader("Dashboard")
    st.write(f"Logged in as **{st.session_state.username}**")
    st.write("Welcome to the Mental Health Post Analyzer. Upload posts and analyze user sentiment.")

elif choice == "Prediction":
    st.subheader("Upload and Analyze Posts (All Users)")
    uploaded_file = st.file_uploader("📄 Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df.head())

        required_columns = {'username', 'post_text', 'timestamp'}
        if not required_columns.issubset(df.columns):
            st.error("CSV must contain: 'username', 'post_text', 'timestamp'")
        else:
            grouped = df.groupby('username')
            results = []

            if st.button("Run Analysis"):
                for username, group in grouped:
                    posts = group['post_text'].tolist()
                    timestamps = group['timestamp'].tolist()

                    sentiment, keywords, flagged, recommendation, sentiments = analyze_posts(posts)
                    results.append({
                        "Username": username,
                        "Num Posts": len(posts),
                        "Sentiment": round(sentiment, 2),
                        "Top Keywords": ", ".join([kw[0] for kw in keywords]),
                        "Flagged": "Yes" if flagged else "No",
                        "Recommendation": recommendation
                    })

                    if username == st.session_state.username:
                        log_analysis(username, sentiment, keywords, flagged, recommendation)

                    show_sentiment_chart(username, timestamps, sentiments)

                result_df = pd.DataFrame(results)
                st.success("✔️ Analysis complete for all users.")
                st.dataframe(result_df)
                st.download_button("Download Results", result_df.to_csv(index=False), file_name="all_user_analysis.csv")

elif choice == "History":
    st.subheader("Analysis History")
    c.execute("SELECT * FROM analysis_logs WHERE username=?", (st.session_state.username,))
    logs = c.fetchall()
    if logs:
        df = pd.DataFrame(logs, columns=["Username", "Sentiment", "Keywords", "Flagged", "Recommendation", "Timestamp"])
        st.dataframe(df)
    else:
        st.info("No past analysis found.")
