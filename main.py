from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import tempfile
import os
import time
import base64
from typing import Optional
import shutil
from datetime import datetime

# Import our detector modules
from loitering_detector import LoiteringDetectionSystem
from violence_detector import ViolenceDetector

app = FastAPI(
    title="AI-Powered Video Surveillance System",
    description="API for loitering detection and violence detection in video",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
loitering_detector = LoiteringDetectionSystem({
    "model": "yolov8n.pt",
    "loitering_threshold": 4.0,
    "confidence": 0.25,
    "output_dir": "output",
    "debug": True
})

# Initialize violence detector
violence_detector = ViolenceDetector("FYP_AI_POWERED_VID_SURVILLEANCE/violence_detection_model.h5")

@app.get("/")
async def root():
    return {"message": "AI-Powered Video Surveillance System API", "status": "online"}

@app.post("/detect/loitering")
async def detect_loitering(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(4.0),
    roi: Optional[str] = Form(None)
):
    """
    Detect loitering in an uploaded image or video frame.
    
    - **file**: Image file to analyze
    - **threshold**: Time in seconds to consider as loitering (only applies when processing video)
    - **roi**: Region of interest as comma-separated coordinates (x1,y1,x2,y2,...)
    """
    try:
        # Update detector config if needed
        loitering_detector.loitering_threshold = threshold
        if roi:
            loitering_detector.roi_coords = loitering_detector.parse_roi(roi)
            
        # Read and process the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the frame
        processed_img, results = loitering_detector.process_frame(img)
        
        # Encode the processed image to base64
        img_base64 = loitering_detector.encode_image_to_base64(processed_img)
        
        # Prepare response
        response = {
            "success": True,
            "loitering_detected": results["loitering_detected"],
            "person_count": results["person_count"],
            "loitering_ids": results["loitering_ids"],
            "detections": results["detections"],
            "processed_image": img_base64
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/detect/loitering/video")
async def process_video_loitering(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(4.0),
    roi: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(1)  # Process every Nth frame
):
    """
    Process video for loitering detection and return summary results.
    
    - **file**: Video file to analyze
    - **threshold**: Time in seconds to consider as loitering
    - **roi**: Region of interest as comma-separated coordinates (x1,y1,x2,y2,...)
    - **sample_rate**: Process every Nth frame (default: 1, process all frames)
    """
    try:
        # Create a temporary directory to store the uploaded video
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save the uploaded file to the temporary location
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
            
            # Update detector config
            loitering_detector.loitering_threshold = threshold
            if roi:
                loitering_detector.roi_coords = loitering_detector.parse_roi(roi)
            
            # Open the video file
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Process video
            results = {
                "video_info": {
                    "filename": file.filename,
                    "frame_count": frame_count,
                    "fps": fps,
                    "duration_seconds": duration
                },
                "loitering_events": [],
                "summary": {
                    "frames_processed": 0,
                    "total_detections": 0,
                    "loitering_instances": 0
                },
                "sample_frames": []
            }
            
            frame_idx = 0
            loitering_detected = False
            
            # Create output directory for sample frames
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/video_{timestamp_str}"
            os.makedirs(output_dir, exist_ok=True)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame
                if frame_idx % sample_rate == 0:
                    frame_time = frame_idx / fps if fps > 0 else 0
                    
                    # Process the frame
                    processed_frame, frame_results = loitering_detector.process_frame(frame, time.time())
                    
                    # Update overall results
                    results["summary"]["frames_processed"] += 1
                    results["summary"]["total_detections"] += frame_results["person_count"]
                    
                    # Check for loitering instances
                    if frame_results["loitering_detected"]:
                        results["summary"]["loitering_instances"] += len(frame_results["loitering_ids"])
                        loitering_detected = True
                        
                        # Save this as a loitering event
                        event = {
                            "frame": frame_idx,
                            "time": frame_time,
                            "loitering_ids": frame_results["loitering_ids"],
                            "detections": frame_results["detections"]
                        }
                        results["loitering_events"].append(event)
                        
                        # Save sample frame
                        if len(results["sample_frames"]) < 5:  # Limit to 5 sample frames
                            filename = f"{output_dir}/loitering_frame_{frame_idx}.jpg"
                            cv2.imwrite(filename, processed_frame)
                            
                            # Add to sample frames
                            results["sample_frames"].append({
                                "frame": frame_idx,
                                "time": frame_time,
                                "image": loitering_detector.encode_image_to_base64(processed_frame)
                            })
                
                frame_idx += 1
            
            # Release resources
            cap.release()
            
            # Add final summary
            results["summary"]["loitering_detected"] = loitering_detected
            
            return JSONResponse(content={"success": True, "results": results})
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/detect/violence")
async def detect_violence(file: UploadFile = File(...)):
    """
    Detect violence in an uploaded image or video frame.
    
    - **file**: Image file to analyze
    """
    try:
        # Read and process the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the frame
        processed_img, results = violence_detector.process_image(img)
        
        # Encode the processed image to base64
        img_base64 = violence_detector.encode_image_to_base64(processed_img)
        
        # Prepare response
        response = {
            "success": True,
            "violence_detected": results["violence_detected"],
            "violence_score": results["violence_score"],
            "processed_image": img_base64
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/detect/violence/video")
async def process_video_violence(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = Form(1)  # Process every Nth frame
):
    """
    Process video for violence detection and return summary results.
    
    - **file**: Video file to analyze
    - **sample_rate**: Process every Nth frame (default: 1, process all frames)
    """
    try:
        # Create a temporary directory to store the uploaded video
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save the uploaded file to the temporary location
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
            
            # Set max frames based on sample rate
            max_frames = None  # Process all frames by default
            if sample_rate > 1:
                # For larger videos, set a reasonable limit
                max_frames = 1000 * sample_rate  # Process ~1000 actual frames
            
            # Process the video
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/violence_output_{timestamp_str}.avi"
            
            results = violence_detector.process_video(
                video_path=temp_file_path, 
                output_path=output_path,
                limit_frames=max_frames
            )
            
            if not results.get("success", False):
                return JSONResponse(
                    status_code=500, 
                    content={"success": False, "error": results.get("error", "Unknown error")}
                )
            
            return JSONResponse(content={"success": True, "results": results})
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Start the API server
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
