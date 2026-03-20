import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import datetime
import time
from PIL import Image, ImageOps 
from ultralytics import YOLO
import easyocr
import base64
import streamlit.components.v1 as components # Required for the scroll feature

# --- CONFIG ---
MODEL_PATH = 'models/best.pt'
CSV_FILE = 'residents.csv'
TOTAL_SPOTS = 50

# --- SOUND CONFIG ---
SOUND_GRANTED = "beep.mp3"
SOUND_DENIED = "denied.mp3"

st.set_page_config(page_title="SmartEntry ALPR", page_icon="🚗", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'parking_spots' not in st.session_state:
    st.session_state.parking_spots = TOTAL_SPOTS

if 'history' not in st.session_state:
    st.session_state.history = []

if 'cars_in_parking' not in st.session_state:
    st.session_state.cars_in_parking = []

if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None 

# --- RESET FUNCTION ---
def reset_state():
    """Clears the previous AI results when a new file is uploaded."""
    st.session_state.last_scan = None

# --- SCROLL FUNCTION (NEW) ---
def scroll_to_top():
    """Forces the page to scroll to the top to show the status message."""
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTop = 0;
    </script>
    """
    components.html(js, height=0)

# --- AUDIO PLAYBACK FUNCTION ---
def play_sound(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            sound_container.empty()
            unique_id = f"sound-{int(time.time() * 1000)}"
            js_code = f"""
                <audio id="{unique_id}" autoplay="autoplay" style="display:none;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                <script>
                    var audio = document.getElementById("{unique_id}");
                    audio.oncanplay = function() {{ audio.play(); }};
                </script>
            """
            sound_container.markdown(js_code, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# --- LOAD RESIDENTS DATABASE ---
if os.path.exists(CSV_FILE):
    df_residents = pd.read_csv(CSV_FILE)
else:
    data = {"Name": ["Amirul", "Hazim"], "Phone": ["012-111", "013-222"], "Plate": ["VFX7126", "WA1234"]}
    df_residents = pd.DataFrame(data)
    df_residents.to_csv(CSV_FILE, index=False)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False) 
    return yolo_model, reader

try:
    with st.spinner("Loading AI Models..."):
        model, reader = load_models()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# --- PREPROCESSING ---
def check_and_enhance_brightness(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    avg_brightness = np.mean(v)
    if avg_brightness < 80:
        return cv2.convertScaleAbs(image_array, alpha=1.5, beta=40), True, avg_brightness
    return image_array, False, avg_brightness

def preprocess_plate(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale_factor = 3
    resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# --- UI START ---
st.title("SmartEntry: Automated Access Control")
sound_container = st.empty()
dashboard_placeholder = st.empty()

# PLACEHOLDER FOR BIG STATUS MESSAGES
status_placeholder = st.empty() 

def update_dashboard():
    with dashboard_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("🅿️ Parking Available", f"{st.session_state.parking_spots}/{TOTAL_SPOTS}")
        col2.metric("📋 Total Log", len(st.session_state.history))
        col3.metric("📅 Date", datetime.date.today().strftime("%B %d, %Y"))
        st.divider()

update_dashboard()

def log_entry(plate, name, status):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, {"Time": timestamp, "Plate": plate, "Owner": name, "Status": status})
    update_dashboard()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.45)

uploaded_file = st.sidebar.file_uploader(
    "Upload Car Image", 
    type=['jpg', 'png', 'jpeg'], 
    on_change=reset_state 
)

if uploaded_file is not None:
    # 1. LOAD IMAGE
    image = Image.open(uploaded_file)
    
    # --- AUTO-ROTATE ---
    image = ImageOps.exif_transpose(image)
    # --------------------------------------------

    img_array = np.array(image)
    final_img, was_enhanced, brightness_val = check_and_enhance_brightness(img_array)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📸 Input Image")
        # --- RESIZED IMAGE: Width fixed to 350px ---
        st.image(final_img, width=350) 
        if was_enhanced:
            st.warning(f"🌙 Auto-Enhanced (Brightness: {int(brightness_val)})")

    # --- 1. AI PROCESSING ---
    if st.sidebar.button("🚀 Analyze Vehicle"):
        # CLEAR PREVIOUS STATUS
        status_placeholder.empty()

        with st.spinner("Processing..."):
            results = model.predict(final_img, conf=conf_threshold)
            result = results[0]
            annotated_frame = result.plot()
            
            scan_data = {
                "annotated_frame": annotated_frame,
                "clean_text": "",
                "cropped_plate": None,
                "processed_plate": None,
                "found": False
            }

            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cropped_plate = final_img[int(y1):int(y2), int(x1):int(x2)]
                processed_plate = preprocess_plate(cropped_plate)
                
                scan_data["cropped_plate"] = cropped_plate
                scan_data["processed_plate"] = processed_plate
                
                ocr_result = reader.readtext(processed_plate)
                detected_text = "".join([text for _, text, prob in ocr_result if prob > 0.3])
                clean_text = "".join(e for e in detected_text if e.isalnum()).upper()
                
                scan_data["clean_text"] = clean_text
                scan_data["found"] = True

                # AUTO LOGIC
                found_user = df_residents[df_residents['Plate'] == clean_text]
                if not found_user.empty:
                    owner_name = found_user.iloc[0]['Name']
                    if clean_text in st.session_state.cars_in_parking: 
                        st.session_state.cars_in_parking.remove(clean_text)
                        if st.session_state.parking_spots < TOTAL_SPOTS: st.session_state.parking_spots += 1
                        
                        play_sound(SOUND_GRANTED)
                        log_entry(clean_text, owner_name, "EXIT")
                        
                        # --- STATUS MESSAGE: GOODBYE ---
                        status_placeholder.success(f"### 👋 GOODBYE, {owner_name}!")
                        scroll_to_top() # <--- SCROLL UP

                    else: 
                        if st.session_state.parking_spots > 0:
                            st.session_state.cars_in_parking.append(clean_text)
                            st.session_state.parking_spots -= 1
                            
                            play_sound(SOUND_GRANTED)
                            log_entry(clean_text, owner_name, "ENTRY")

                            # --- STATUS MESSAGE: WELCOME ---
                            status_placeholder.success(f"### ✅ WELCOME, {owner_name}!")
                            scroll_to_top() # <--- SCROLL UP

                        else:
                            play_sound(SOUND_DENIED)
                            log_entry(clean_text, owner_name, "DENIED (FULL)")
                            
                            # --- STATUS MESSAGE: FULL ---
                            status_placeholder.error(f"### ⛔ PARKING FULL!")
                            scroll_to_top() # <--- SCROLL UP

                else:
                    play_sound(SOUND_DENIED)
                    log_entry(clean_text, "Unknown", "DENIED")
                    
                    # --- STATUS MESSAGE: DENIED ---
                    status_placeholder.error(f"### ❌ ACCESS DENIED: Unknown Vehicle ({clean_text})")
                    scroll_to_top() # <--- SCROLL UP

            else:
                play_sound(SOUND_DENIED)
                status_placeholder.warning("### ⚠️ NO PLATE DETECTED")
                scroll_to_top() # <--- SCROLL UP
            
            st.session_state.last_scan = scan_data

    # --- 2. RESULT DISPLAY ---
    if st.session_state.last_scan:
        data = st.session_state.last_scan
        
        with col2:
            st.subheader("🎯 AI Analysis")
            # --- RESIZED IMAGE: Width fixed to 350px ---
            st.image(data["annotated_frame"], width=350)

        if data["found"]:
            with st.expander("🔍 See What the OCR Sees (Debug View)", expanded=True):
                c_a, c_b = st.columns(2)
                # --- RESIZED DEBUG IMAGES: Width fixed to 200px ---
                c_a.image(data["cropped_plate"], caption="Original Crop", width=200)
                c_b.image(data["processed_plate"], caption="Preprocessed (B&W)", width=200, channels='GRAY')
            
            st.divider()
            st.markdown(f"### Detected: **{data['clean_text']}**")

            st.info("If the result above is wrong, fix it here:")
            
            with st.expander("🛠️ Manual Override Panel", expanded=True):
                col_man1, col_man2 = st.columns([3, 1])
                manual_plate = col_man1.text_input("Correct Plate Number", value=data['clean_text'])
                
                if col_man2.button("Force Entry/Exit"):
                    # CLEAR PREVIOUS STATUS
                    status_placeholder.empty()

                    manual_user = df_residents[df_residents['Plate'] == manual_plate]
                    
                    if not manual_user.empty:
                        m_name = manual_user.iloc[0]['Name']
                        
                        if manual_plate in st.session_state.cars_in_parking: 
                            st.session_state.cars_in_parking.remove(manual_plate)
                            if st.session_state.parking_spots < TOTAL_SPOTS: 
                                st.session_state.parking_spots += 1
                            
                            play_sound(SOUND_GRANTED)
                            log_entry(manual_plate, m_name, "EXIT (MANUAL)")
                            status_placeholder.success(f"### 👋 GOODBYE (Manual), {m_name}!")
                            scroll_to_top() # <--- SCROLL UP
                            
                        elif st.session_state.parking_spots > 0: 
                            st.session_state.cars_in_parking.append(manual_plate)
                            st.session_state.parking_spots -= 1
                            
                            play_sound(SOUND_GRANTED)
                            log_entry(manual_plate, m_name, "ENTRY (MANUAL)")
                            status_placeholder.success(f"### ✅ WELCOME (Manual), {m_name}!")
                            scroll_to_top() # <--- SCROLL UP
                            
                        else:
                            st.error("Parking Full")
                            play_sound(SOUND_DENIED)
                            log_entry(manual_plate, m_name, "DENIED (FULL)")
                            status_placeholder.error(f"### ⛔ PARKING FULL!")
                            scroll_to_top() # <--- SCROLL UP

                    else:
                        st.error(f"❌ {manual_plate} not in DB")
                        play_sound(SOUND_DENIED)
                        log_entry(manual_plate, "Unknown", "DENIED (MANUAL)")
                        status_placeholder.error(f"### ❌ ACCESS DENIED: {manual_plate} not found")
                        scroll_to_top() # <--- SCROLL UP

        else:
            st.warning("⚠️ No plate detected.")

# 3. HISTORY TABLE
st.divider()
st.subheader("Live Entry Log")
if len(st.session_state.history) > 0:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No vehicles have entered yet.")
