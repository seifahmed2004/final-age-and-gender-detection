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
    page_title="Age & Gender Detection",
    layout="wide",
    initial_sidebar_state="collapsed"   # better default on mobile
)

SNAPSHOT_DIR = "snapshots"
DB_PATH = "predictions.db"
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@admin.com")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ============================================================
# Paths
# ============================================================
AGE_MODEL_PATH    = "models/best_age_efficientnet_b4_finetuned.pth"
GENDER_MODEL_PATH = "models/best_gender_utkface.pth"
YOLO_FACE_MODEL_PATH = "models/yolov8n-face-lindevs.pt"


# ============================================================
# Mobile-friendly CSS
# ============================================================
st.markdown("""
<style>
/* Fluid containers */
.block-container { padding: 1rem 1rem 2rem 1rem !important; max-width: 100% !important; }

/* Larger tap targets on buttons */
div.stButton > button {
    width: 100%;
    padding: 0.6rem 1rem;
    font-size: 1rem;
    border-radius: 8px;
}

/* Stack columns on narrow screens */
@media (max-width: 640px) {
    [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        timestamp TEXT,
        snapshot_path TEXT
    )
    """)

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

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_email TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def register_user(email: str, password: str):
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


def login_user(email: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE email = ?", (email.lower().strip(),))
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


def insert_prediction(user_email, image_path, face_image_path,
                      predicted_age, predicted_gender, gender_confidence,
                      face_confidence, sharpness):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (
        user_email, timestamp, image_path, face_image_path,
        predicted_age, predicted_gender, gender_confidence,
        face_confidence, sharpness, feedback, corrected_age,
        corrected_gender, reviewer_comment
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_path, face_image_path,
        float(predicted_age), predicted_gender,
        float(gender_confidence), float(face_confidence), float(sharpness),
        None, None, None, None
    ))
    conn.commit()
    conn.close()


def load_predictions_all():
    """All predictions — used by dashboard (every user sees full history)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def load_predictions_mine(user_email: str):
    """Only the current user's predictions — used by Review page."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE user_email = ? ORDER BY id DESC",
        conn, params=(user_email,)
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


def update_feedback(prediction_id, feedback, corrected_age,
                    corrected_gender, reviewer_comment):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE predictions
    SET feedback = ?, corrected_age = ?, corrected_gender = ?, reviewer_comment = ?
    WHERE id = ?
    """, (feedback, float(corrected_age), corrected_gender, reviewer_comment, int(prediction_id)))
    conn.commit()
    conn.close()


init_db()


# ============================================================
# Model Architectures
# ============================================================
class AgeEfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.shared = nn.Sequential(
            nn.Linear(1792, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.age_head = nn.Linear(256, 1)
        self.bin_head = nn.Linear(256, 7)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.age_head(x), self.bin_head(x)


class GenderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# Load Models
# ============================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    age_model = AgeEfficientNetB4().to(device)
    age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=device))
    age_model.eval()
    gender_model = GenderCNN().to(device)
    gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=device))
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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
gender_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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
    age_tensor    = age_transform(face_pil).unsqueeze(0).to(device)
    gender_tensor = gender_transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        age_output, _ = age_model(age_tensor)
        predicted_age  = age_output.item()
        gender_prob    = torch.sigmoid(gender_model(gender_tensor)).item()
    if gender_prob >= 0.5:
        return predicted_age, "Female", gender_prob
    return predicted_age, "Male", 1.0 - gender_prob


def crop_face_with_padding(frame, box, padding_ratio=0.20):
    x1, y1, x2, y2 = box.astype(int)
    pad = int(padding_ratio * max(x2-x1, y2-y1))
    x1p = max(0, x1-pad);  y1p = max(0, y1-pad)
    x2p = min(frame.shape[1], x2+pad); y2p = min(frame.shape[0], y2+pad)
    return frame[y1p:y2p, x1p:x2p], x1p, y1p, x2p, y2p


def draw_label(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    label_y = max(30, y1-10)
    cv2.rectangle(frame, (x1, label_y-28), (min(x1+520, frame.shape[1]), label_y+8), (0,255,0), -1)
    cv2.putText(frame, label, (x1+5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)


def save_snapshot(full_frame_bgr, face_crop_bgr, user_email):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_email = user_email.replace("@","_").replace(".","_")
    user_dir   = os.path.join(SNAPSHOT_DIR, safe_email)
    os.makedirs(user_dir, exist_ok=True)
    full_path = os.path.join(user_dir, f"snapshot_{timestamp}.jpg")
    face_path = os.path.join(user_dir, f"face_{timestamp}.jpg")
    cv2.imwrite(full_path, full_frame_bgr)
    cv2.imwrite(face_path, face_crop_bgr)
    return full_path, face_path


def process_camera_image(camera_image, yolo_conf):
    np_arr = np.frombuffer(camera_image.getvalue(), np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "Could not read image."
    annotated_frame = frame.copy()
    results = face_detector(frame, conf=float(yolo_conf), verbose=False)
    best_score, best_data = -1, None
    if results[0].boxes is not None:
        for box, face_conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                   results[0].boxes.conf.cpu().numpy()):
            face_crop, x1p, y1p, x2p, y2p = crop_face_with_padding(frame, box)
            if face_crop.size == 0:
                continue
            pred_age, pred_gender, gender_conf = predict_age_and_gender(face_crop)
            sharpness = calculate_sharpness(face_crop)
            score = float(face_conf)*1000 + float(sharpness)*0.1 + (x2p-x1p)*(y2p-y1p)*0.0001
            draw_label(annotated_frame, x1p, y1p, x2p, y2p,
                       f"Age: {pred_age:.1f} | {pred_gender}: {gender_conf*100:.1f}%")
            if score > best_score:
                best_score = score
                best_data  = {
                    "annotated_frame": annotated_frame.copy(),
                    "face_crop": face_crop.copy(),
                    "predicted_age": pred_age,
                    "predicted_gender": pred_gender,
                    "gender_confidence": gender_conf,
                    "face_confidence": float(face_conf),
                    "sharpness": float(sharpness)
                }
    return (best_data, None) if best_data else (None, "No face detected.")


# ============================================================
# Session State
# ============================================================
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "user_email"  not in st.session_state: st.session_state.user_email  = None
if "active_page" not in st.session_state: st.session_state.active_page = "Live Camera"


# ============================================================
# Login / Register Page
# ============================================================
def show_auth_page():
    st.title("Age & Gender Detection System")
    st.write("Please sign in or create an account to continue.")

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        st.subheader("Sign In")
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", type="primary", use_container_width=True, key="login_btn"):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                success, message = login_user(email, password)
                if success:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email.lower().strip()
                    log_login(email.lower().strip())
                    st.rerun()
                else:
                    st.error(message)

    with tab_register:
        st.subheader("Create Account")
        new_email    = st.text_input("Email",            key="reg_email")
        new_password = st.text_input("Password",         type="password", key="reg_password")
        confirm_pwd  = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Create Account", type="primary", use_container_width=True, key="reg_btn"):
            if not new_email or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_pwd:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, message = register_user(new_email, new_password)
                st.success(message + " You can now sign in.") if success else st.error(message)


# ============================================================
# Main App
# ============================================================
def show_main_app():
    user_email = st.session_state.user_email
    is_admin   = (user_email == ADMIN_EMAIL)

    # ── Sidebar navigation ──────────────────────────────────
    with st.sidebar:
        st.title("Navigation")
        st.write(f"Signed in as: **{user_email}**")
        if is_admin:
            st.success("Admin")

        pages = ["Live Camera", "Human-in-the-Loop Review", "Leader Dashboard"]
        if is_admin:
            pages.append("Admin Panel")

        page = st.radio("Go to", pages)
        st.divider()
        st.write("Device:", str(device))
        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in  = False
            st.session_state.user_email = None
            st.rerun()

    # ── Mobile top-bar (shown when sidebar is collapsed) ────
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.caption(f"Signed in as **{user_email}**" + (" · Admin" if is_admin else ""))


    # ===========================================================
    # Page 1: Live Camera
    # ===========================================================
    if page == "Live Camera":
        st.title("Live Camera Prediction")
        st.info("Take a snapshot — the model detects your face and predicts age & gender.")

        yolo_conf = st.slider("YOLO face confidence", 0.1, 0.9, 0.4, 0.05)
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            with st.spinner("Running models..."):
                best_data, error_message = process_camera_image(camera_image, yolo_conf)

            if error_message:
                st.warning(error_message)
            else:
                full_path, face_path = save_snapshot(
                    best_data["annotated_frame"], best_data["face_crop"], user_email
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
                log_login(user_email, snapshot_path=full_path)
                st.success("Prediction saved.")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(best_data["annotated_frame"], cv2.COLOR_BGR2RGB),
                             caption="Model prediction", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(best_data["face_crop"], cv2.COLOR_BGR2RGB),
                             caption="Detected face crop", use_container_width=True)

                st.write("### Prediction Result")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Age",       round(best_data["predicted_age"], 1))
                c2.metric("Predicted Gender",    best_data["predicted_gender"])
                c3.metric("Gender Confidence",   f"{best_data['gender_confidence']*100:.2f}%")
                st.write("Face confidence:", round(best_data["face_confidence"], 2))
                st.write("Sharpness:",       round(best_data["sharpness"], 2))
                st.info("Go to Human-in-the-Loop Review to mark this decision.")

    # ===========================================================
    # Page 2: Human-in-the-Loop Review  (own predictions only)
    # ===========================================================
    elif page == "Human-in-the-Loop Review":
        st.title("Human-in-the-Loop Review")

        if st.session_state.get("feedback_success"):
            st.success("Feedback saved successfully.")
            st.session_state.feedback_success = False

        df = load_predictions_mine(user_email)

        if df.empty:
            st.warning("You have no predictions yet.")
        else:
            st.write("### Your Predictions")
            st.dataframe(df.drop(columns=["image_path","face_image_path"], errors="ignore"),
                         use_container_width=True)

            selected_id  = st.selectbox("Select prediction ID", df["id"].tolist())
            selected_row = df[df["id"] == selected_id].iloc[0]

            # Show photos only to admin
            if is_admin:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if selected_row["image_path"] and os.path.exists(selected_row["image_path"]):
                        st.image(selected_row["image_path"], caption="Full snapshot",
                                 use_container_width=True)
                    if pd.notna(selected_row.get("face_image_path")) and \
                       os.path.exists(selected_row["face_image_path"]):
                        st.image(selected_row["face_image_path"], caption="Face crop",
                                 use_container_width=True)
                details_col = col2
            else:
                details_col = st.container()

            with details_col:
                st.write("### Model Decision")
                st.write("Prediction ID:", int(selected_row["id"]))
                st.write("Timestamp:",     selected_row["timestamp"])
                st.write("Predicted age:", round(selected_row["predicted_age"], 1))
                st.write("Predicted gender:", selected_row["predicted_gender"])
                st.write("Gender confidence:",
                         round(selected_row["gender_confidence"]*100, 2), "%")
                st.write("Face confidence:", round(selected_row["face_confidence"], 2))
                st.write("Sharpness:",       round(selected_row["sharpness"], 2))
                st.divider()
                st.write("### Human Review")

                feedback = st.radio("Was the model decision good?", ["Good","Bad"], horizontal=True)
                corrected_age = st.number_input(
                    "Corrected age", 0, 100,
                    value=int(round(selected_row["predicted_age"]))
                )
                gender_options = ["Male","Female"]
                corrected_gender = st.selectbox(
                    "Corrected gender", gender_options,
                    index=gender_options.index(selected_row["predicted_gender"])
                    if selected_row["predicted_gender"] in gender_options else 0
                )
                reviewer_comment = st.text_area("Reviewer comment")

                if st.button("Submit Feedback", type="primary", use_container_width=True):
                    update_feedback(int(selected_id), feedback,
                                    float(corrected_age), corrected_gender, reviewer_comment)
                    st.session_state.feedback_success = True
                    st.rerun()

    # ===========================================================
    # Page 3: Leader Dashboard  (ALL predictions, no photos)
    # ===========================================================
    elif page == "Leader Dashboard":
        st.title("Leader Dashboard")

        df = load_predictions_all()   # ← full history for everyone

        if df.empty:
            st.warning("No data available yet.")
        else:
            total        = len(df)
            reviewed     = df["feedback"].notna().sum()
            good_count   = (df["feedback"] == "Good").sum()
            bad_count    = (df["feedback"] == "Bad").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Predictions", total)
            c2.metric("Reviewed",          reviewed)
            c3.metric("Good Feedback",     good_count)
            c4.metric("Bad Feedback",      bad_count)

            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.histogram(df, x="predicted_age", nbins=20,
                                 title="Predicted Age Distribution"),
                    use_container_width=True
                )
            with col_b:
                st.plotly_chart(
                    px.pie(df, names="predicted_gender",
                           title="Predicted Gender Distribution"),
                    use_container_width=True
                )

            # Contribution per user (visible to all, no photos)
            st.divider()
            if "user_email" in df.columns:
                user_counts = df["user_email"].value_counts().reset_index()
                user_counts.columns = ["user", "predictions"]
                st.plotly_chart(
                    px.bar(user_counts, x="user", y="predictions",
                           title="Predictions per User"),
                    use_container_width=True
                )

            st.divider()
            if reviewed > 0:
                reviewed_df = df[df["feedback"].notna()].copy()
                col_c, col_d = st.columns(2)
                with col_c:
                    st.plotly_chart(
                        px.pie(reviewed_df, names="feedback",
                               title="Human Feedback: Good vs Bad"),
                        use_container_width=True
                    )
                with col_d:
                    st.bar_chart(reviewed_df["feedback"].value_counts())

                corrected_df = reviewed_df.dropna(subset=["corrected_age"]).copy()
                if not corrected_df.empty:
                    corrected_df["age_error"] = abs(
                        corrected_df["predicted_age"] - corrected_df["corrected_age"]
                    )
                    st.metric("Avg Age Error vs Human Correction",
                              round(corrected_df["age_error"].mean(), 2))
                    st.plotly_chart(
                        px.histogram(corrected_df, x="age_error", nbins=20,
                                     title="Age Error After Human Review"),
                        use_container_width=True
                    )
                    gender_mismatch = (
                        corrected_df["predicted_gender"] != corrected_df["corrected_gender"]
                    ).sum()
                    st.metric("Gender Corrections", int(gender_mismatch))
            else:
                st.info("No human feedback submitted yet.")

            st.divider()
            # Raw table — hide photo paths for non-admins
            st.write("### Raw Data")
            display_cols = [c for c in df.columns
                            if c not in ("image_path","face_image_path") or is_admin]
            st.dataframe(df[display_cols], use_container_width=True)

    # ===========================================================
    # Page 4: Admin Panel
    # ===========================================================
    elif page == "Admin Panel" and is_admin:
        st.title("Admin Panel")

        tab1, tab2, tab3 = st.tabs(["All Predictions", "Login Logs", "Registered Users"])

        with tab1:
            st.subheader("All Predictions")
            df_all = load_predictions_all()
            if df_all.empty:
                st.warning("No predictions yet.")
            else:
                users_list    = ["All"] + sorted(df_all["user_email"].dropna().unique().tolist())
                selected_user = st.selectbox("Filter by user", users_list)
                if selected_user != "All":
                    df_all = df_all[df_all["user_email"] == selected_user]
                st.dataframe(df_all, use_container_width=True)

                if not df_all.empty:
                    selected_id = st.selectbox("View photos for ID", df_all["id"].tolist())
                    row = df_all[df_all["id"] == selected_id].iloc[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        if row["image_path"] and os.path.exists(row["image_path"]):
                            st.image(row["image_path"],
                                     caption=f"Full — {row['user_email']}",
                                     use_container_width=True)
                    with col2:
                        if pd.notna(row["face_image_path"]) and os.path.exists(row["face_image_path"]):
                            st.image(row["face_image_path"], caption="Face crop",
                                     use_container_width=True)

        with tab2:
            st.subheader("Login Logs")
            df_logs = load_login_logs()
            if df_logs.empty:
                st.warning("No login logs yet.")
            else:
                st.dataframe(df_logs, use_container_width=True)
                logs_with_photos = df_logs[df_logs["snapshot_path"].notna()]
                if not logs_with_photos.empty:
                    st.write("#### Login Snapshots")
                    sel = st.selectbox("Select log entry", logs_with_photos["id"].tolist())
                    log_row = logs_with_photos[logs_with_photos["id"] == sel].iloc[0]
                    if os.path.exists(log_row["snapshot_path"]):
                        st.image(log_row["snapshot_path"],
                                 caption=f"{log_row['email']} — {log_row['timestamp']}",
                                 use_container_width=True)

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