# AI-Powered Video Surveillance System

This project is an AI-powered video surveillance system that detects **loitering behavior** and **violence** in videos using deep learning models.

---

## Features

- **Loitering Detection**: Identifies individuals who remain in a specific area for longer than a designated threshold time.
- **Violence Detection**: Detects violent actions in video frames using a MobileNetV2-based model.
- **RESTful API**: Provides endpoints for both image and video processing.
- **Real-time Processing**: Capable of processing video streams in real-time.

---

## Project Structure

```
├── main.py                # FastAPI application entry point
├── violence_detector.py   # Violence detection module
├── loitering_detector.py  # Loitering detection module
├── requirements.txt       # Python dependencies
├── start.sh               # Script to start the application
├── .gitignore             # Git ignore file
└── violence_detection.h5  # Pre-trained violence detection model
```

---

## Setup and Installation

### Clone the repository:

```bash
git clone https://github.com/Venkateshsanwal/AI_POWERED_VIDEO_SURVILLEANCE_SYSTEM_FYP.git
cd AI_POWERED_VIDEO_SURVILLEANCE_SYSTEM_FYP
```

### Make the start script executable and run it:

```bash
chmod +x start.sh
./start.sh
```

### Or set up manually:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### Loitering Detection

- **Detect Loitering in an Image**  
  URL: `/detect/loitering`  
  Method: POST  
  Parameters:  
  - `file`: Image file (required)  
  - `threshold`: Time in seconds (default: 4.0)  
  - `roi`: Region of interest as comma-separated coordinates (optional)  

- **Detect Loitering in a Video**  
  URL: `/detect/loitering/video`  
  Method: POST  
  Parameters:  
  - `file`: Video file (required)  
  - `threshold`: Time in seconds (default: 4.0)  
  - `roi`: Region of interest as comma-separated coordinates (optional)  
  - `sample_rate`: Process every Nth frame (default: 1)  

---

### Violence Detection

- **Detect Violence in an Image**  
  URL: `/detect/violence`  
  Method: POST  
  Parameters:  
  - `file`: Image file (required)  

- **Detect Violence in a Video**  
  URL: `/detect/violence/video`  
  Method: POST  
  Parameters:  
  - `file`: Video file (required)  
  - `sample_rate`: Process every Nth frame (default: 1)  

---

## Required Models

- **YOLOv8**: For person detection in the loitering module (automatically downloaded)  
- **Violence Detection Model**: Pre-trained MobileNetV2 model (`violence_detection.h5`)  

---

## Hosting Options (Free Tiers)

- Render  
- Heroku  
- PythonAnywhere  
- Google Cloud Run  
- Hugging Face Spaces  

---

## Environment Variables

| Variable              | Description                                   |
|-----------------------|-----------------------------------------------|
| MODEL_PATH            | Path to the violence detection model          |
| LOITERING_THRESHOLD   | Default time threshold for loitering detection|
| CONFIDENCE_THRESHOLD  | Confidence threshold for YOLO detection       |
| OUTPUT_DIR            | Directory to save output files                 |
| DEBUG                 | Enable debug mode (`true`/`false`)             |

---

## Notes for Frontend Integration

The API returns processed frames and detection results in JSON format. For video processing, it returns detection results along with sample frames as base64-encoded images.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.