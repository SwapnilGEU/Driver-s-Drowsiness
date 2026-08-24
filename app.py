import streamlit as st
import cv2

from detection.detector import DrowsinessDetector

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="WakeSafe AI",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }

    .stTextInput>div>div>input {
        font-size: 18px;
    }

    .stTextArea textarea {
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SESSION STATE
# ==========================================
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# ==========================================
# TITLE
# ==========================================
st.markdown(
    """
    <h1 style='text-align:center; color:white;'>
    🚗 WakeSafe AI Driver Monitoring System
    </h1>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SETUP SCREEN
# ==========================================
if not st.session_state.monitoring:

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.subheader("Emergency Contact Setup")

        phone_number = st.text_input(
            "Enter Emergency Phone Number",
            placeholder="+91XXXXXXXXXX"
        )

        voice_message = st.text_area(
            "Voice Call Message",
            value=(
                "Warning. The driver may be drowsy. "
                "Please contact them immediately."
            ),
            height=120
        )

        sms_message = st.text_area(
            "SMS Message",
            value=(
                "WakeSafe AI ALERT:\n"
                "Possible drowsiness detected.\n"
                "Please contact the driver immediately."
            ),
            height=120
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚨 START MONITORING"):

            if phone_number == "":

                st.error("Please enter a phone number")

            else:

                st.session_state.monitoring = True

                st.session_state.phone_number = phone_number

                st.session_state.voice_message = voice_message

                st.session_state.sms_message = sms_message

                st.rerun()

# ==========================================
# MONITORING SCREEN
# ==========================================
else:

    st.success("Monitoring Started")

    detector = DrowsinessDetector()

    # ======================================
    # CENTER WEBCAM
    # ======================================
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with center_col:

        frame_placeholder = st.empty()

    # ======================================
    # STOP BUTTON
    # ======================================
    stop_col1, stop_col2, stop_col3 = st.columns([1, 1, 1])

    with stop_col2:

        stop_button = st.button("🛑 STOP MONITORING")

    # ======================================
    # CAMERA
    # ======================================
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            st.error("Camera Error")
            break

        # ==================================
        # SMALLER WEBCAM SIZE
        # ==================================
        frame = cv2.resize(frame, (420, 300))

        # ==================================
        # PROCESS FRAME
        # ==================================
        processed_frame, is_drowsy = detector.process_frame(
            frame,
            st.session_state.phone_number,
            st.session_state.voice_message,
            st.session_state.sms_message
        )

        # ==================================
        # BGR TO RGB
        # ==================================
        processed_frame = cv2.cvtColor(
            processed_frame,
            cv2.COLOR_BGR2RGB
        )

        # ==================================
        # DISPLAY CENTERED WEBCAM
        # ==================================
        with center_col:

            frame_placeholder.image(
                processed_frame,
                channels="RGB",
                width=500
            )

        # ==================================
        # STOP MONITORING
        # ==================================
        if stop_button:

            break

    # ======================================
    # CLEANUP
    # ======================================
    cap.release()

    st.session_state.monitoring = False