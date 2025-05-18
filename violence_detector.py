import numpy as np
import cv2
import os
import time
from keras.models import load_model
from collections import deque
import base64
from datetime import datetime
from PIL import Image
import io

class ViolenceDetector:
    def __init__(self, model_path="FYP_AI_POWERED_VID_SURVILLEANCE/violence_detection_model.h5"):
        """Initialize Violence Detection system with pre-trained model."""
        self.model_path = model_path
        
        # Create output directory if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')
            
        # Load the model
        print(f"Loading violence detection model from {model_path}...")
        try:
            self.model = load_model(model_path)
            print("Violence detection model loaded successfully")
        except Exception as e:
            print(f"Error loading violence detection model: {e}")
            self.model = None
            
        # Initialize prediction queue for averaging
        self.prediction_queue = deque(maxlen=128)
        
    def predict_single_frame(self, frame):
        """Process a single frame and return violence prediction."""
        if self.model is None:
            return {
                "error": "Model not loaded. Please check model path.",
                "violence_detected": False,
                "violence_score": 0.0
            }
            
        # Process the frame
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(processed_frame, (128, 128)).astype("float32")
        processed_frame = processed_frame / 255.0
        
        # Make prediction
        preds = self.model.predict(np.expand_dims(processed_frame, axis=0))[0]
        self.prediction_queue.append(preds)
        
        # Calculate averaged results
        results = np.array(self.prediction_queue).mean(axis=0)
        
        # Get violence prediction
        violence_score = float(preds[0])
        violence_detected = violence_score > 0.50
        
        return {
            "violence_detected": violence_detected,
            "violence_score": violence_score,
            "avg_violence_score": float(results[0])
        }
        
    def process_image(self, image_data):
        """Process an image for violence detection.
        
        Args:
            image_data: Can be a file path, numpy array, or base64 encoded image
            
        Returns:
            Processed image and detection results
        """
        # Handle different input types
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # Base64 encoded image
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # File path
                frame = cv2.imread(image_data)
        elif isinstance(image_data, np.ndarray):
            # Already a numpy array
            frame = image_data
        else:
            raise ValueError("Unsupported image data format")
            
        # Get a copy of the original frame
        output_frame = frame.copy()
        
        # Get prediction
        result = self.predict_single_frame(frame)
        
        # Visualize the result
        output_frame = self.visualize_result(output_frame, result)
        
        return output_frame, result
        
    def visualize_result(self, frame, result):
        """Add visualization of violence detection result to the frame."""
        # Get prediction results
        violence_detected = result["violence_detected"]
        violence_score = result.get("violence_score", 0.0)
        
        # Set text based on prediction
        if violence_detected:
            text = f"Violence: True ({violence_score:.2f})"
            color = (0, 0, 255)  # Red for violence
        else:
            text = f"Violence: False ({violence_score:.2f})"
            color = (0, 255, 0)  # Green for no violence
            
        # Add text to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (35, 50), font, 1.25, color, 3)
        
        return frame
        
    def process_video(self, video_path, output_path=None, limit_frames=None):
        """Process a video file for violence detection.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video (default: output/v_output.avi)
            limit_frames: Maximum number of frames to process (default: None, process all)
            
        Returns:
            Dictionary with violence detection results
        """
        if self.model is None:
            return {
                "error": "Model not loaded. Please check model path.",
                "success": False
            }
            
        # Reset prediction queue
        self.prediction_queue = deque(maxlen=128)
        
        # Set default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/violence_output_{timestamp}.avi"
            
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "error": f"Could not open video file: {video_path}",
                "success": False
            }
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        # Tracking statistics
        frame_count = 0
        violence_frames = 0
        violence_segments = []
        current_segment = None
        results = {
            "video_info": {
                "filename": os.path.basename(video_path),
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": total_frames / fps if fps > 0 else 0
            },
            "processing_summary": {
                "frames_processed": 0,
                "violence_detected": False,
                "violence_segments": [],
                "violence_percentage": 0.0
            },
            "sample_frames": []
        }
        
        try:
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Optional limit on frames to process
                if limit_frames is not None and frame_count >= limit_frames:
                    break
                    
                # Process frame
                frame_time = frame_count / fps if fps > 0 else 0
                prediction = self.predict_single_frame(frame)
                
                # Update statistics
                frame_count += 1
                
                if prediction["violence_detected"]:
                    violence_frames += 1
                    
                    # Track violence segments
                    if current_segment is None:
                        current_segment = {
                            "start_frame": frame_count,
                            "start_time": frame_time,
                            "end_frame": frame_count,
                            "end_time": frame_time
                        }
                    else:
                        current_segment["end_frame"] = frame_count
                        current_segment["end_time"] = frame_time
                else:
                    # End of violence segment
                    if current_segment is not None:
                        violence_segments.append(current_segment)
                        current_segment = None
                
                # Visualize the result
                output_frame = self.visualize_result(frame, prediction)
                
                # Save sample frames (up to 5)
                if len(results["sample_frames"]) < 5 and (
                    prediction["violence_detected"] or frame_count % (total_frames // 5) == 0
                ):
                    # Convert frame to base64 for JSON response
                    _, buffer = cv2.imencode('.jpg', output_frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    results["sample_frames"].append({
                        "frame": frame_count,
                        "time": frame_time,
                        "violence_detected": prediction["violence_detected"],
                        "violence_score": prediction["violence_score"],
                        "image": f"data:image/jpeg;base64,{img_str}"
                    })
                
                # Write frame to output video
                writer.write(output_frame)
                
            # Final segment if video ends during violence
            if current_segment is not None:
                violence_segments.append(current_segment)
                
            # Update results summary
            results["processing_summary"]["frames_processed"] = frame_count
            results["processing_summary"]["violence_detected"] = violence_frames > 0
            results["processing_summary"]["violence_segments"] = violence_segments
            
            if frame_count > 0:
                results["processing_summary"]["violence_percentage"] = (violence_frames / frame_count) * 100
                
            results["success"] = True
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
        finally:
            # Release resources
            cap.release()
            writer.release()
            
    def encode_image_to_base64(self, image):
        """Convert an OpenCV image to base64 encoding."""
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

# # For testing purposes
# def test_violence_detector():
#     detector = ViolenceDetector("violence_detection.h5")
    
#     # Test with a single image
#     test_image = "path/to/test/image.jpg"
#     if os.path.exists(test_image):
#         processed_image, results = detector.process_image(test_image)
#         print(f"Detection results: {results}")
        
#         # Save processed image
#         output_path = "output/test_violence_processed.jpg"
#         cv2.imwrite(output_path, processed_image)
#         print(f"Processed image saved to {output_path}")

# if __name__ == "__main__":
#     test_violence_detector()