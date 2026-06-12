import sqlite3
from datetime import datetime
import streamlit as st
import os

DB_PATH = "predictions.db"


ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "12345678")

def check_admin():
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        pwd = st.text_input("Enter admin password to view database", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()  # stops rendering the rest of the page

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT,
        predicted_age REAL,
        predicted_gender TEXT,
        gender_confidence REAL,
        face_confidence REAL,
        sharpness REAL,
        feedback TEXT,
        corrected_age REAL,
        corrected_gender TEXT,
        reviewer_comment TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_prediction(
    image_path,
    predicted_age,
    predicted_gender,
    gender_confidence,
    face_confidence,
    sharpness
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
        timestamp,
        image_path,
        predicted_age,
        predicted_gender,
        gender_confidence,
        face_confidence,
        sharpness,
        feedback,
        corrected_age,
        corrected_gender,
        reviewer_comment
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_path,
        predicted_age,
        predicted_gender,
        gender_confidence,
        face_confidence,
        sharpness,
        None,
        None,
        None,
        None
    ))

    conn.commit()
    conn.close()


def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()

    columns = [
        "id",
        "timestamp",
        "image_path",
        "predicted_age",
        "predicted_gender",
        "gender_confidence",
        "face_confidence",
        "sharpness",
        "feedback",
        "corrected_age",
        "corrected_gender",
        "reviewer_comment"
    ]

    return rows, columns


def update_feedback(
    prediction_id,
    feedback,
    corrected_age,
    corrected_gender,
    reviewer_comment
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE predictions
    SET feedback = ?,
        corrected_age = ?,
        corrected_gender = ?,
        reviewer_comment = ?
    WHERE id = ?
    """, (
        feedback,
        corrected_age,
        corrected_gender,
        reviewer_comment,
        prediction_id
    ))

    conn.commit()
    conn.close()