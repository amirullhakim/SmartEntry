========================================================================
   SMARTENTRY: AUTOMATED VEHICLE ACCESS CONTROL SYSTEM VIA ALPR
========================================================================

AUTHORS:
------------------------------------------------------------------------
GROUP 07 TT2L
1. Amirul (1211111890)
2. Hazim (1211112351)
3. Munif (1211110167)
4. Salahuddin (1211108880)

Faculty of Computing and Informatics, Multimedia University
Subject: CDS6334 Visual Information Processing

========================================================================
1. PROJECT OVERVIEW
========================================================================
SmartEntry is an Automated License Plate Recognition (ALPR) system designed 
to manage vehicle access in residential areas. It utilizes Deep Learning 
(YOLOv8) for plate detection and Optical Character Recognition (EasyOCR) 
to read license plate numbers. The system tracks entry/exit activity, 
manages parking capacity, and verifies residents against a database.

========================================================================
2. PREREQUISITES
========================================================================
- Python 3.8 or higher

========================================================================
3. INSTALLATION INSTRUCTIONS
========================================================================
1. Unzip the project folder.
2. Open a terminal or command prompt in the project directory.
3. Install the required dependencies using the following command:

   pip install -r requirements.txt

========================================================================
4. HOW TO RUN THE APPLICATION
========================================================================
To launch the main dashboard:

1. Open your terminal in the project folder.
2. Run the command:

   streamlit run app.py

3. The application will open automatically in your web browser.

========================================================================
5. PROJECT STRUCTURE & FILE DESCRIPTIONS
========================================================================
- /dataset                    : Contains training data (images/labels).
- /demo_images                : Sample images (Clear, Angled, Dark) for testing.
- /models                     : Contains the trained YOLO model (best.pt).
- app.py                      : The main Streamlit application source code.
- calc_metric_ocr_manually.py : A separate tool to calculate OCR accuracy metrics manually.
- residents.csv               : The database file storing registered residents.
- requirements.txt            : List of Python libraries required.
- beep.mp3                    : Sound effect for access granted.
- denied.mp3                  : Sound effect for access denied.

========================================================================
6. SYSTEM FEATURES
========================================================================
- Live Parking Counter: Automatically updates available spots (Total: 50).
- AI Analysis: Visualizes YOLOv8 bounding box detection.
- Debug View: Shows the cropping and image preprocessing steps 
  (Grayscale -> Gaussian Blur -> Otsu Thresholding).
- Access Logic:
  - Entry: Checks database -> Grants Access -> Deducts Parking Spot.
  - Exit: Checks database -> Grants Exit -> Frees Parking Spot.
  - Denied: Unknown plate or Full Parking.
- Manual Override: Allows security guards to manually correct OCR errors.
- Live Entry Log: A real-time table recording all vehicle activities.

========================================================================
7. HOW TO RUN THE ACCURACY EVALUATION TOOL
========================================================================
To run the separate script for calculating detection and OCR accuracy metrics:

1. Run the command:
   
   streamlit run calc_metric_ocr_manually.py

2. Follow the on-screen instructions to verify detection results manually.

========================================================================
8. TROUBLESHOOTING
========================================================================
- Image Rotation Issue: The system includes auto-rotation for phone 
  uploads. If an image appears sideways, the app handles it automatically.
- Model Not Found: Ensure 'best.pt' is located inside the 'models/' folder.
- OCR Errors: Use the "Manual Override" panel in the sidebar to correct 
  misread plates (e.g., if 'M' is read as 'W').

========================================================================