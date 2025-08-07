import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comprehension Cam", layout="centered")

# 🔶 Title and Info
st.markdown("<h1 style='color:orange;'>🧠 Real-Time Student Comprehension Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#003366;'>This tool uses your webcam and facial expression detection to estimate student comprehension in real-time using MediaPipe.</p>", unsafe_allow_html=True)

# 🔲 Label toggle in sidebar
show_labels = st.sidebar.checkbox("Show comprehension labels on live video", value=False)

# Session states
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'student_logs' not in st.session_state:
    st.session_state.student_logs = []
if 'log_times' not in st.session_state:
    st.session_state.log_times = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0

# Controls
col1, col2 = st.columns(2)
with col1:
    start_button = st.button('▶️ Start Camera', use_container_width=True)
with col2:
    end_button = st.button('⏹ End Session and Show Results', use_container_width=True)
FRAME_WINDOW = st.image([])

# Comprehension mapping
comprehension_map = {
    'high': 'High Comprehension',
    'medium': 'Medium Comprehension',
    'low': 'Low Comprehension'
}

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Start session
if start_button:
    st.session_state.camera_running = True
    st.session_state.student_logs = []
    st.session_state.log_times = []
    st.session_state.start_time = time.time()

# End session
if end_button:
    st.session_state.camera_running = False

# Run camera
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        face_count = 0
        current_time = time.time()

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)

            # Make sure we have enough student logs
            while len(st.session_state.student_logs) < face_count:
                st.session_state.student_logs.append([])
                st.session_state.log_times.append([])

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                ih, iw, _ = rgb_frame.shape
                landmark_array = np.array([(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks.landmark])

                # Feature calculations
                left_eye = landmark_array[159][1] - landmark_array[145][1]
                right_eye = landmark_array[386][1] - landmark_array[374][1]
                eye_avg = (left_eye + right_eye) / 2

                mouth_open = landmark_array[14][1] - landmark_array[13][1]
                brow_raise = landmark_array[70][1] - landmark_array[105][1]

                # 🧠 Comprehension scoring
                if eye_avg > 8 and mouth_open > 16 and brow_raise > 9:
                    comprehension = 'low'
                elif eye_avg > 5 or mouth_open > 9:
                    comprehension = 'medium'
                else:
                    comprehension = 'high'

                x_min = np.min(landmark_array[:, 0])
                y_min = np.min(landmark_array[:, 1])
                x_max = np.max(landmark_array[:, 0])
                y_max = np.max(landmark_array[:, 1])
                cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 120, 255), 2)

                # 🏷️ Label with toggle
                label = f"Student {i+1}"
                if show_labels:
                    label += f" - {comprehension_map[comprehension]}"
                cv2.putText(rgb_frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                st.session_state.student_logs[i].append(comprehension_map[comprehension])
                st.session_state.log_times[i].append(current_time)

        cv2.putText(rgb_frame, f"Students detected: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        FRAME_WINDOW.image(rgb_frame)

        if end_button:
            break
    cap.release()

# 📊 Results summary
if not st.session_state.camera_running and st.session_state.student_logs:
    st.markdown("### 📊 Comprehension Summary Per Student")
    summary_data = []
    for i, log in enumerate(st.session_state.student_logs):
        total = len(log)
        if total > 0:
            high = log.count("High Comprehension")
            medium = log.count("Medium Comprehension")
            low = log.count("Low Comprehension")
            st.write(f"**Student {i+1}:** 🟢 High: {round(100*high/total,1)}% | 🟡 Medium: {round(100*medium/total,1)}% | 🔴 Low: {round(100*low/total,1)}%")
            summary_data.append({"Student": i+1, "High %": round(100*high/total,1), "Medium %": round(100*medium/total,1), "Low %": round(100*low/total,1)})

            fig, ax = plt.subplots()
            time_series = [t - st.session_state.log_times[i][0] for t in st.session_state.log_times[i]]
            scores = [1 if s == "High Comprehension" else (0.5 if s == "Medium Comprehension" else 0) for s in log]
            ax.plot(time_series, scores, marker='o', color='darkorange')
            ax.set_title(f"Comprehension Trend - Student {i+1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score: 1=High, 0.5=Medium, 0=Low")
            st.pyplot(fig)

    # Export to CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary CSV", csv, "comprehension_summary.csv", "text/csv")

