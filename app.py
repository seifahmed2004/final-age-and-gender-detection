import os
import cv2
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import timm
import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib

from datetime import datetime
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


# ============================================================
# App Config
# ============================================================
st.set_page_config(
    page_title="Age & Gender Human-in-the-Loop System",
    layout="wide"
)

SNAPSHOT_DIR = "snapshots"
DB_PATH = "predictions.db"
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@admin.com")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ============================================================
# Paths
# ============================================================
AGE_MODEL_PATH = "models/best_age_efficientnet_b4_finetuned.pth"
GENDER_MODEL_PATH = "models/best_gender_utkface.pth"
YOLO_FACE_MODEL_PATH = "models/yolov8n-face-lindevs.pt"


# ============================================================
# Auth Helpers
# ============================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# Database
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )
    """)

    # Login logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        timestamp TEXT,
        snapshot_path TEXT
    )
    """)

    # Predictions table (with user_email column)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        timestamp TEXT,
        image_path TEXT,
        face_image_path TEXT,
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

    # Add user_email column if it doesn't exist (for existing DBs)
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_email TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()


def register_user(email: str, password: str) -> tuple[bool, str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email.lower().strip(), hash_password(password),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        conn.close()


def login_user(email: str, password: str) -> tuple[bool, str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password_hash FROM users WHERE email = ?",
        (email.lower().strip(),)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False, "Email not found."
    if row[0] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful."


def log_login(email: str, snapshot_path: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO login_logs (email, timestamp, snapshot_path) VALUES (?, ?, ?)",
        (email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), snapshot_path)
    )
    conn.commit()
    conn.close()


def insert_prediction(
    user_email,
    image_path,
    face_image_path,
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
        user_email,
        timestamp,
        image_path,
        face_image_path,
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
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_email,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_path,
        face_image_path,
        float(predicted_age),
        predicted_gender,
        float(gender_confidence),
        float(face_confidence),
        float(sharpness),
        None, None, None, None
    ))

    conn.commit()
    conn.close()


def load_predictions(user_email: str = None):
    conn = sqlite3.connect(DB_PATH)
    if user_email == ADMIN_EMAIL:
        # Admin sees everything
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    else:
        # Regular user sees only their own
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE user_email = ? ORDER BY id DESC",
            conn,
            params=(user_email,)
        )
    conn.close()
    return df


def load_login_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM login_logs ORDER BY id DESC", conn)
    conn.close()
    return df


def load_all_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, email, created_at FROM users ORDER BY id DESC", conn
    )
    conn.close()
    return df


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
        float(corrected_age),
        corrected_gender,
        reviewer_comment,
        int(prediction_id)
    ))

    conn.commit()
    conn.close()


init_db()


# ============================================================
# Age Model Architecture
# ============================================================
class AgeEfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0
        )

        self.shared = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.age_head = nn.Linear(256, 1)
        self.bin_head = nn.Linear(256, 7)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        age = self.age_head(x)
        age_bin = self.bin_head(x)

        return age, age_bin


# ============================================================
# Gender Model Architecture
# ============================================================
class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Load Models
# ============================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    age_model = AgeEfficientNetB4().to(device)
    age_model.load_state_dict(
        torch.load(AGE_MODEL_PATH, map_location=device)
    )
    age_model.eval()

    gender_model = GenderCNN().to(device)
    gender_model.load_state_dict(
        torch.load(GENDER_MODEL_PATH, map_location=device)
    )
    gender_model.eval()

    face_detector = YOLO(YOLO_FACE_MODEL_PATH)

    return device, age_model, gender_model, face_detector


device, age_model, gender_model, face_detector = load_models()


# ============================================================
# Transforms
# ============================================================
age_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

gender_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# Helper Functions
# ============================================================
def calculate_sharpness(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def predict_age_and_gender(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    age_tensor = age_transform(face_pil).unsqueeze(0).to(device)
    gender_tensor = gender_transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        age_output, _ = age_model(age_tensor)
        predicted_age = age_output.item()

        gender_logit = gender_model(gender_tensor)
        gender_prob = torch.sigmoid(gender_logit).item()

    if gender_prob >= 0.5:
        predicted_gender = "Female"
        gender_confidence = gender_prob
    else:
        predicted_gender = "Male"
        gender_confidence = 1.0 - gender_prob

    return predicted_age, predicted_gender, gender_confidence


def crop_face_with_padding(frame, box, padding_ratio=0.20):
    x1, y1, x2, y2 = box.astype(int)

    w = x2 - x1
    h = y2 - y1
    pad = int(padding_ratio * max(w, h))

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(frame.shape[1], x2 + pad)
    y2p = min(frame.shape[0], y2 + pad)

    face_crop = frame[y1p:y2p, x1p:x2p]

    return face_crop, x1p, y1p, x2p, y2p


def draw_label(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_y = max(30, y1 - 10)
    label_width = 520

    cv2.rectangle(
        frame,
        (x1, label_y - 28),
        (min(x1 + label_width, frame.shape[1]), label_y + 8),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


def save_snapshot(full_frame_bgr, face_crop_bgr, user_email):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_email = user_email.replace("@", "_").replace(".", "_")

    user_dir = os.path.join(SNAPSHOT_DIR, safe_email)
    os.makedirs(user_dir, exist_ok=True)

    full_path = os.path.join(user_dir, f"snapshot_{timestamp}.jpg")
    face_path = os.path.join(user_dir, f"face_{timestamp}.jpg")

    cv2.imwrite(full_path, full_frame_bgr)
    cv2.imwrite(face_path, face_crop_bgr)

    return full_path, face_path


def process_camera_image(camera_image, yolo_conf):
    file_bytes = camera_image.getvalue()

    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None, "Could not read image."

    annotated_frame = frame.copy()

    results = face_detector(
        frame,
        conf=float(yolo_conf),
        verbose=False
    )

    result = results[0]
    best_score = -1
    best_data = None

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, face_conf in zip(boxes, confidences):
            face_crop, x1p, y1p, x2p, y2p = crop_face_with_padding(frame, box)

            if face_crop.size == 0:
                continue

            pred_age, pred_gender, gender_conf = predict_age_and_gender(face_crop)

            sharpness = calculate_sharpness(face_crop)
            face_area = (x2p - x1p) * (y2p - y1p)

            score = (
                float(face_conf) * 1000
                + float(sharpness) * 0.1
                + float(face_area) * 0.0001
            )

            label = (
                f"Age: {pred_age:.1f} | "
                f"{pred_gender}: {gender_conf * 100:.1f}%"
            )

            draw_label(annotated_frame, x1p, y1p, x2p, y2p, label)

            if score > best_score:
                best_score = score
                best_data = {
                    "annotated_frame": annotated_frame.copy(),
                    "face_crop": face_crop.copy(),
                    "predicted_age": pred_age,
                    "predicted_gender": pred_gender,
                    "gender_confidence": gender_conf,
                    "face_confidence": float(face_conf),
                    "sharpness": float(sharpness)
                }

    if best_data is None:
        return None, "No face detected."

    return best_data, None


# ============================================================
# Session State Init
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None


# ============================================================
# Login / Register Page
# ============================================================
def show_auth_page():
    st.title("Age & Gender Detection System")
    st.write("Please sign in or create an account to continue.")

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        st.subheader("Sign In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In", type="primary", key="login_btn"):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                success, message = login_user(email, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email.lower().strip()
                    log_login(email.lower().strip())
                    st.rerun()
                else:
                    st.error(message)

    with tab_register:
        st.subheader("Create Account")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")

        if st.button("Create Account", type="primary", key="reg_btn"):
            if not new_email or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, message = register_user(new_email, new_password)
                if success:
                    st.success(message + " You can now sign in.")
                else:
                    st.error(message)


# ============================================================
# Main App (requires login)
# ============================================================
def show_main_app():
    user_email = st.session_state.user_email
    is_admin = (user_email == ADMIN_EMAIL)

    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Signed in as: **{user_email}**")
    if is_admin:
        st.sidebar.success("Admin")

    pages = ["Live Camera", "Human-in-the-Loop Review", "Leader Dashboard"]
    if is_admin:
        pages.append("Admin Panel")

    page = st.sidebar.radio("Go to", pages)

    st.sidebar.divider()
    st.sidebar.write("Device:", str(device))

    if st.sidebar.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

    # ============================================================
    # Page 1: Live Camera
    # ============================================================
    if page == "Live Camera":
        st.title("Live Camera Prediction")

        st.info(
            "Take a snapshot from your browser camera. "
            "The model will detect the face, predict age/gender, "
            "save the best face snapshot, and store the decision for human review."
        )

        yolo_conf = st.slider(
            "YOLO face confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05
        )

        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            with st.spinner("Running YOLO + age/gender models..."):
                best_data, error_message = process_camera_image(camera_image, yolo_conf)

            if error_message is not None:
                st.warning(error_message)
            else:
                full_path, face_path = save_snapshot(
                    best_data["annotated_frame"],
                    best_data["face_crop"],
                    user_email
                )

                insert_prediction(
                    user_email=user_email,
                    image_path=full_path,
                    face_image_path=face_path,
                    predicted_age=best_data["predicted_age"],
                    predicted_gender=best_data["predicted_gender"],
                    gender_confidence=best_data["gender_confidence"],
                    face_confidence=best_data["face_confidence"],
                    sharpness=best_data["sharpness"]
                )

                # Log this snapshot with login
                log_login(user_email, snapshot_path=full_path)

                st.success("Prediction saved successfully.")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        cv2.cvtColor(best_data["annotated_frame"], cv2.COLOR_BGR2RGB),
                        caption="Model prediction",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        cv2.cvtColor(best_data["face_crop"], cv2.COLOR_BGR2RGB),
                        caption="Detected face crop",
                        use_container_width=True
                    )

                st.write("### Prediction Result")

                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric("Predicted Age", round(best_data["predicted_age"], 1))
                result_col2.metric("Predicted Gender", best_data["predicted_gender"])
                result_col3.metric(
                    "Gender Confidence",
                    f"{best_data['gender_confidence'] * 100:.2f}%"
                )

                st.write("Face confidence:", round(best_data["face_confidence"], 2))
                st.write("Sharpness:", round(best_data["sharpness"], 2))

                st.info("Go to Human-in-the-Loop Review to mark this decision as Good or Bad.")

    # ============================================================
    # Page 2: Human-in-the-Loop Review
    # ============================================================
    elif page == "Human-in-the-Loop Review":
        st.title("Human-in-the-Loop Review")

        if "feedback_success" not in st.session_state:
            st.session_state.feedback_success = False

        if st.session_state.feedback_success:
            st.success("Feedback saved successfully.")
            st.session_state.feedback_success = False

        df = load_predictions(user_email)

        if df.empty:
            st.warning("No predictions saved yet.")
        else:
            st.write("### Saved Predictions")
            st.dataframe(df, use_container_width=True)

            selected_id = st.selectbox("Select prediction ID", df["id"].tolist())
            selected_row = df[df["id"] == selected_id].iloc[0]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    selected_row["image_path"],
                    caption="Full snapshot",
                    use_container_width=True
                )

                if (
                    "face_image_path" in selected_row
                    and pd.notna(selected_row["face_image_path"])
                ):
                    st.image(
                        selected_row["face_image_path"],
                        caption="Face crop",
                        use_container_width=True
                    )

            with col2:
                st.write("### Model Decision")
                st.write("Prediction ID:", int(selected_row["id"]))
                st.write("Timestamp:", selected_row["timestamp"])
                st.write("Predicted age:", round(selected_row["predicted_age"], 1))
                st.write("Predicted gender:", selected_row["predicted_gender"])
                st.write(
                    "Gender confidence:",
                    round(selected_row["gender_confidence"] * 100, 2), "%"
                )
                st.write("Face confidence:", round(selected_row["face_confidence"], 2))
                st.write("Sharpness:", round(selected_row["sharpness"], 2))

                st.divider()
                st.write("### Human Review")

                feedback = st.radio(
                    "Was the model decision good?",
                    ["Good", "Bad"],
                    horizontal=True
                )

                corrected_age = st.number_input(
                    "Corrected age",
                    min_value=0,
                    max_value=100,
                    value=int(round(selected_row["predicted_age"]))
                )

                gender_options = ["Male", "Female"]
                default_gender_index = (
                    gender_options.index(selected_row["predicted_gender"])
                    if selected_row["predicted_gender"] in gender_options
                    else 0
                )

                corrected_gender = st.selectbox(
                    "Corrected gender",
                    gender_options,
                    index=default_gender_index
                )

                reviewer_comment = st.text_area("Reviewer comment")

                if st.button("Submit Feedback", type="primary"):
                    update_feedback(
                        prediction_id=int(selected_id),
                        feedback=feedback,
                        corrected_age=float(corrected_age),
                        corrected_gender=corrected_gender,
                        reviewer_comment=reviewer_comment
                    )

                    st.session_state.feedback_success = True
                    st.rerun()

    # ============================================================
    # Page 3: Leader Dashboard
    # ============================================================
    elif page == "Leader Dashboard":
        st.title("Leader Dashboard")

        df = load_predictions(user_email)

        if df.empty:
            st.warning("No data available yet.")
        else:
            total_predictions = len(df)
            reviewed_count = df["feedback"].notna().sum()
            good_count = (df["feedback"] == "Good").sum()
            bad_count = (df["feedback"] == "Bad").sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", total_predictions)
            col2.metric("Reviewed", reviewed_count)
            col3.metric("Good Feedback", good_count)
            col4.metric("Bad Feedback", bad_count)

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:
                fig_age = px.histogram(df, x="predicted_age", nbins=20, title="Predicted Age Distribution")
                st.plotly_chart(fig_age, use_container_width=True)

            with col_b:
                fig_gender = px.pie(df, names="predicted_gender", title="Predicted Gender Distribution")
                st.plotly_chart(fig_gender, use_container_width=True)

            st.divider()

            if reviewed_count > 0:
                reviewed_df = df[df["feedback"].notna()].copy()

                col_c, col_d = st.columns(2)

                with col_c:
                    fig_feedback = px.pie(reviewed_df, names="feedback", title="Human Feedback: Good vs Bad")
                    st.plotly_chart(fig_feedback, use_container_width=True)

                with col_d:
                    feedback_counts = reviewed_df["feedback"].value_counts()
                    st.bar_chart(feedback_counts)

                corrected_df = reviewed_df.dropna(subset=["corrected_age"]).copy()

                if not corrected_df.empty:
                    corrected_df["age_error_after_review"] = abs(
                        corrected_df["predicted_age"] - corrected_df["corrected_age"]
                    )

                    avg_review_error = corrected_df["age_error_after_review"].mean()
                    st.metric("Average Age Error Against Human Correction", round(avg_review_error, 2))

                    fig_error = px.histogram(
                        corrected_df, x="age_error_after_review", nbins=20,
                        title="Age Error After Human Review"
                    )
                    st.plotly_chart(fig_error, use_container_width=True)

                    gender_mismatch = (
                        corrected_df["predicted_gender"] != corrected_df["corrected_gender"]
                    ).sum()
                    st.metric("Gender Corrections", int(gender_mismatch))
            else:
                st.info("No human feedback submitted yet.")

            st.divider()
            st.write("### Raw Data")
            st.dataframe(df, use_container_width=True)

    # ============================================================
    # Page 4: Admin Panel (admin only)
    # ============================================================
    elif page == "Admin Panel" and is_admin:
        st.title("Admin Panel")

        tab1, tab2, tab3 = st.tabs(["All Predictions", "Login Logs", "Registered Users"])

        with tab1:
            st.subheader("All Predictions (All Users)")
            df_all = load_predictions(ADMIN_EMAIL)
            if df_all.empty:
                st.warning("No predictions yet.")
            else:
                # Filter by user
                users_list = ["All"] + sorted(df_all["user_email"].dropna().unique().tolist())
                selected_user = st.selectbox("Filter by user", users_list)

                if selected_user != "All":
                    df_all = df_all[df_all["user_email"] == selected_user]

                st.dataframe(df_all, use_container_width=True)

                # Show photos for selected row
                if not df_all.empty:
                    selected_id = st.selectbox("View photos for prediction ID", df_all["id"].tolist())
                    row = df_all[df_all["id"] == selected_id].iloc[0]

                    col1, col2 = st.columns(2)
                    with col1:
                        if row["image_path"] and os.path.exists(row["image_path"]):
                            st.image(row["image_path"], caption=f"Full snapshot — {row['user_email']}", use_container_width=True)
                    with col2:
                        if pd.notna(row["face_image_path"]) and os.path.exists(row["face_image_path"]):
                            st.image(row["face_image_path"], caption="Face crop", use_container_width=True)

        with tab2:
            st.subheader("Login Logs")
            df_logs = load_login_logs()
            if df_logs.empty:
                st.warning("No login logs yet.")
            else:
                st.dataframe(df_logs, use_container_width=True)

                # Show snapshot from login log if exists
                if "snapshot_path" in df_logs.columns:
                    logs_with_photos = df_logs[df_logs["snapshot_path"].notna()]
                    if not logs_with_photos.empty:
                        st.write("#### Login Snapshots")
                        selected_log = st.selectbox(
                            "Select log entry to view photo",
                            logs_with_photos["id"].tolist()
                        )
                        log_row = logs_with_photos[logs_with_photos["id"] == selected_log].iloc[0]
                        if os.path.exists(log_row["snapshot_path"]):
                            st.image(
                                log_row["snapshot_path"],
                                caption=f"{log_row['email']} — {log_row['timestamp']}",
                                use_container_width=True
                            )

        with tab3:
            st.subheader("Registered Users")
            df_users = load_all_users()
            st.dataframe(df_users, use_container_width=True)
            st.metric("Total Registered Users", len(df_users))


# ============================================================
# Entry Point
# ============================================================
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_app()