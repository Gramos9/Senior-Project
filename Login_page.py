import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import subprocess
import webbrowser

streamlit_url = "https://senior-project-stock-predicition.streamlit.app/"

# Create/connect to the SQLite database
def setup_database():
    conn = sqlite3.connect("user_database.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Login", "Register"])

if page == "Login":
    st.header("Login")
    login_username = st.text_input("Username", key="login-username")
    login_password = st.text_input("Password", type="password", key="login-password")

    if st.button("Login"):
        conn = sqlite3.connect("user_database.db")
        cursor = conn.cursor()

        # Check if the username exists
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (login_username,))
        user = cursor.fetchone()

        if user and pbkdf2_sha256.verify(login_password, user[1]):
            st.success("Logged in as {}".format(login_username))
            # Navigate to the Main_page.py
            webbrowser.open(streamlit_url)
        else:
            st.error("Invalid username or password")

        conn.close()

if page == "Register":
    st.header("Registration")
    new_username = st.text_input("Username", key="registration-username")
    new_email = st.text_input("Email", key="registration-email")
    new_password = st.text_input("Password", type="password", key="registration-password")

    if st.button("Register"):
        conn = sqlite3.connect("user_database.db")
        cursor = conn.cursor()

        # Check if the username or email is already in use
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (new_username, new_email))
        existing_user = cursor.fetchone()

        if existing_user:
            st.error("Username or email is already in use.")
        else:
            # Hash the password
            hashed_password = pbkdf2_sha256.hash(new_password)

            # Insert the new user into the database
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (new_username, new_email, hashed_password))
            conn.commit()
            conn.close()
            st.success("User registered. You can now log in.")









