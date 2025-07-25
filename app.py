# --- Streamlit Mental Health Post Analyzer ---
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
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    username TEXT,
    date TEXT,
    sentiment REAL,
    keywords TEXT,
    flagged INTEGER,
    recommendation TEXT
)
''')
conn.commit()

# --- Analyzer Function ---
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

# --- Streamlit App Setup ---
st.set_page_config(page_title="Mental Health Post Analyzer", layout="centered")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Navigation Menu ---
menu = ["Login", "Register"]
if st.session_state.logged_in:
    menu = ["Dashboard", "Analyze", "History", "Logout"]

choice = st.sidebar.selectbox("Menu", menu)

# --- Login/Register Logic ---
if choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Account created successfully! Please login.")
        else:
            st.error("Username already exists.")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

# --- Dashboard with Study Objectives ---
elif choice == "Dashboard" and st.session_state.logged_in:
    st.title("🧠 Mental Health Post Analyzer Dashboard")
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. To develop a system that evaluates users' social media posts using sentiment analysis techniques.  
    2. To track and visualize the emotional trends of users over time.  
    3. To generate mental health alerts when negative sentiment thresholds are exceeded.  
    4. To enable exporting of mental health reports for further evaluation.  
    5. To store historical records of analyses for all users.
    """)

# --- Analyze Page ---
elif choice == "Analyze" and st.session_state.logged_in:
    st.title("📊 Analyze Uploaded Mental Health Posts (CSV Format)")
    uploaded_file = st.file_uploader("Upload CSV File (with 'username', 'post_text', 'timestamp')", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in ['username', 'post_text', 'timestamp']):
                st.error("CSV must contain 'username', 'post_text', and 'timestamp' columns.")
            else:
                unique_users = df['username'].unique().tolist()
                st.write(f"Found {len(unique_users)} users in uploaded file.")

                if st.button("Run Analysis for All Users"):
                    for uname in unique_users:
                        user_data = df[df['username'] == uname]
                        posts = user_data['post_text'].dropna().tolist()
                        timestamps = user_data['timestamp'].dropna().tolist()
                        
                        if len(posts) == 0:
                            st.warning(f"Skipping user '{uname}' because no valid posts were found.")
                            continue
                        
                        sentiment, keywords, flagged, recommendation, sentiments = analyze_posts(posts)
                        log_analysis(uname, sentiment, keywords, flagged, recommendation)


                    st.success("Sentiment analysis completed for all users and saved to history.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- History Page with Trend Visualization ---
elif choice == "History" and st.session_state.logged_in:
    st.title("📂 Analysis History")

    # Load all logs
    cursor.execute("SELECT * FROM logs ORDER BY date ASC")
    records = cursor.fetchall()

    if records:
        df_logs = pd.DataFrame(records, columns=["Username", "Date", "Sentiment", "Keywords", "Flagged", "Recommendation"])
        df_logs['Date'] = pd.to_datetime(df_logs['Date'])  # Ensure datetime format

        # Display table
        st.subheader("🗃️ Full Log History")
        st.dataframe(df_logs)
        
        if st.button("🗑️ Clear All Analysis History"):
            cursor.execute("DELETE FROM logs")
            conn.commit()
            st.success("All analysis history has been cleared.")
            st.rerun()

        # User selection for trend visualization
        selected_user = st.selectbox("📌 Select a user to visualize emotional trends:", df_logs['Username'].unique())

        user_data = df_logs[df_logs['Username'] == selected_user]

        if not user_data.empty:
            st.subheader(f"📈 Emotional Trend for **{selected_user}**")
            fig, ax = plt.subplots()
            ax.plot(user_data['Date'], user_data['Sentiment'], marker='o', linestyle='-', color='blue')
            ax.set_title(f'Sentiment Trend for {selected_user}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sentiment Polarity')
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("No sentiment history available for this user.")
    else:
        st.info("No analysis history found.")


# --- Logout ---
elif choice == "Logout" and st.session_state.logged_in:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully.")
    st.rerun()
