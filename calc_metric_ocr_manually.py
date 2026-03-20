import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image, ImageOps
from ultralytics import YOLO
import easyocr

# --- CONFIGURATION ---
# The folder where your 60 images are stored
# Ensure this path is correct relative to where you run the script
IMAGE_FOLDER = r"dark_images" 
MODEL_PATH = 'models/best.pt'

st.set_page_config(page_title="OCR Accuracy Evaluator", page_icon="📊", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = []  # Stores data: {'file': name, 'type': clear/angle/dark, 'detected': 1/0, 'correct': 1/0}

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if 'image_list' not in st.session_state:
    # Load all images (jpg, png, jpeg)
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    st.session_state.image_list = sorted(files)

# --- LOAD MODELS (Cached) ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, reader

try:
    with st.spinner("Loading AI Models..."):
        model, reader = load_models()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# --- HELPER FUNCTIONS (Reused from app.py) ---
def check_and_enhance_brightness(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    avg_brightness = np.mean(v)
    if avg_brightness < 80:
        return cv2.convertScaleAbs(image_array, alpha=1.5, beta=40), True
    return image_array, False

def preprocess_plate(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale_factor = 3
    resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_image_category(filename):
    name = os.path.basename(filename).lower()
    if name.startswith("clear"):
        return "Clear Day"
    elif name.startswith("angle"):
        return "Angled View"
    elif name.startswith("dark") or name.startswith("low"):
        return "Low Light"
    else:
        return "Other"

# --- MAIN APP LOGIC ---
st.title("📊 OCR Accuracy Evaluation Tool")

total_images = len(st.session_state.image_list)

# CHECK IF DONE
if st.session_state.current_index >= total_images:
    st.success("🎉 Evaluation Complete!")
    
    # --- CALCULATE FINAL STATISTICS ---
    df_res = pd.DataFrame(st.session_state.results)
    
    # Define the template categories
    categories = ["Clear Day", "Angled View", "Low Light"]
    
    summary_data = []
    
    total_samples = 0
    total_detected = 0
    total_correct = 0

    for cat in categories:
        # Filter data for this category
        subset = df_res[df_res['type'] == cat]
        
        n_samples = len(subset)
        n_detected = subset['detected'].sum() if n_samples > 0 else 0
        n_correct = subset['correct'].sum() if n_samples > 0 else 0
        
        summary_data.append({
            "Condition": cat,
            "Samples": n_samples,
            "Detected": n_detected,
            "Text Read Correctly": n_correct
        })
        
        total_samples += n_samples
        total_detected += n_detected
        total_correct += n_correct

    # Add Total Row
    summary_data.append({
        "Condition": "Total",
        "Samples": total_samples,
        "Detected": total_detected,
        "Text Read Correctly": total_correct
    })
    
    # Create Final DataFrame
    final_df = pd.DataFrame(summary_data)
    
    st.subheader("Final Results Table")
    st.table(final_df)
    
    # Download Button
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, "ocr_evaluation_results.csv", "text/csv")
    
    if st.button("🔄 Restart Evaluation"):
        st.session_state.current_index = 0
        st.session_state.results = []
        st.rerun()

else:
    # --- SHOW CURRENT IMAGE ---
    current_file = st.session_state.image_list[st.session_state.current_index]
    filename = os.path.basename(current_file)
    category = get_image_category(filename)
    
    # Progress Bar
    progress = (st.session_state.current_index + 1) / total_images
    st.progress(progress, text=f"Processing Image {st.session_state.current_index + 1} of {total_images}: {filename} ({category})")
    
    # Load & Process
    try:
        pil_image = Image.open(current_file)
        pil_image = ImageOps.exif_transpose(pil_image)
        img_array = np.array(pil_image)
        
        # Preprocessing (Brightness)
        final_img, _ = check_and_enhance_brightness(img_array)
        
        # Run AI
        results = model.predict(final_img, conf=0.45) # Use same conf as app.py default
        result = results[0]
        
        # Layout
        col1, col2 = st.columns([1, 1])
        
        detected_plate_text = "NOT DETECTED"
        is_detected = 0
        
        # Crop & OCR
        if len(result.boxes) > 0:
            is_detected = 1
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cropped_plate = final_img[int(y1):int(y2), int(x1):int(x2)]
            processed_plate = preprocess_plate(cropped_plate)
            
            ocr_result = reader.readtext(processed_plate)
            raw_text = "".join([text for _, text, prob in ocr_result if prob > 0.3])
            detected_plate_text = "".join(e for e in raw_text if e.isalnum()).upper()
            
            # Draw Box for visualization
            annotated_frame = result.plot()
            
            with col1:
                st.image(annotated_frame, caption=f"YOLO Detection", use_container_width=True)
                
            with col2:
                st.image(cropped_plate, caption="Cropped Plate", width=200)
                st.metric("AI Read Result", detected_plate_text)
                
        else:
            with col1:
                st.image(final_img, caption="Original Image (No Plate Found)", use_container_width=True)
            with col2:
                st.warning("⚠️ No License Plate Detected by YOLO")

        # --- USER VERIFICATION ---
        st.divider()
        st.subheader(f"📝 Verify Result for: {filename}")
        
        c1, c2, c3 = st.columns(3)
        
        def save_result(correct_flag):
            st.session_state.results.append({
                'file': filename,
                'type': category,
                'detected': is_detected,
                'correct': 1 if correct_flag else 0
            })
            st.session_state.current_index += 1
            st.rerun() # Force reload for next image

        with c1:
            if st.button("✅ Text is CORRECT", type="primary", use_container_width=True):
                save_result(True)
        
        with c2:
            if st.button("❌ Text is WRONG", type="secondary", use_container_width=True):
                save_result(False)
                
        with c3:
            if st.button("⚠️ Not Detected / Skip", type="secondary", use_container_width=True):
                # If not detected, it is obviously not correct text
                save_result(False)

    except Exception as e:
        st.error(f"Error processing {filename}: {e}")
        if st.button("Skip Image"):
            st.session_state.current_index += 1
            st.rerun()