import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
import os
from datetime import datetime
import base64
import io
from PIL import Image as PILImage

class LoiteringDetectionSystem:
    def __init__(self, config=None):
        """Initialize the Loitering Detection System with configuration parameters."""
        # Default configuration
        default_config = {
            "model": "yolov8n.pt",
            "loitering_threshold": 4.0,
            "confidence": 0.25,
            "roi": None,
            "output_dir": "output",
            "debug": False
        }
        
        # Use provided config or default
        self.config = config if config else default_config
        
        # Configuration parameters
        self.loitering_threshold = self.config.get("loitering_threshold", 4.0)
        self.roi_coords = self.parse_roi(self.config.get("roi")) if self.config.get("roi") else None
        self.confidence_threshold = self.config.get("confidence", 0.25)
        self.output_dir = self.config.get("output_dir")
        self.debug = self.config.get("debug", False)
        
        # Ensure output directory exists
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize YOLO model
        print(f"Loading YOLOv8 model: {self.config.get('model', 'yolov8n.pt')}")
        self.model = YOLO(self.config.get("model", "yolov8n.pt"))
        
        # Track detections over time
        self.person_tracks = defaultdict(list)  # {track_id: [(timestamp, bbox, centroid), ...]}
        self.loitering_alerts = set()  # Set of track_ids currently loitering
        
        # For visualizing tracks
        self.track_colors = {}
        
    def parse_roi(self, roi_str):
        """Parse ROI coordinates from string format 'x1,y1,x2,y2,...'"""
        if not roi_str:
            return None
            
        coords = [int(c) for c in roi_str.split(',')]
        if len(coords) % 2 != 0 or len(coords) < 6:
            raise ValueError("ROI must be specified as x1,y1,x2,y2,... with at least 3 points")
            
        # Reshape into [(x1,y1), (x2,y2), ...]
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        return np.array(points, np.int32)
        
    def point_in_roi(self, point):
        """Check if a point is within the defined ROI."""
        if self.roi_coords is None:
            return True  # If no ROI defined, consider all points in ROI
            
        # Use cv2.pointPolygonTest to check if point is inside polygon
        return cv2.pointPolygonTest(self.roi_coords, point, False) >= 0
        
    def process_frame(self, frame, timestamp=None):
        """Process a video frame for loitering detection."""
        if timestamp is None:
            timestamp = time.time()
            
        # Run YOLOv8 detection with tracking
        results = self.model.track(frame, persist=True, conf=self.confidence_threshold, classes=[0])  # class 0 is person
        
        # Initialize detection results
        detection_results = {
            "loitering_detected": False,
            "loitering_ids": [],
            "person_count": 0,
            "detections": []
        }
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return frame, detection_results  # No detections
            
        # Get boxes and track IDs
        boxes = results[0].boxes.xywh.cpu().numpy()  # (x_center, y_center, width, height)
        track_ids = results[0].boxes.id
        
        if track_ids is None:
            return frame, detection_results  # No tracking IDs available
            
        track_ids = track_ids.int().cpu().numpy()
        detection_results["person_count"] = len(track_ids)
        
        # Process each detection
        for box, track_id in zip(boxes, track_ids):
            x_center, y_center, width, height = box
            centroid = (int(x_center), int(y_center))
            
            # Skip if not in ROI
            if not self.point_in_roi(centroid):
                continue
                
            # Convert box to (x1, y1, x2, y2) format for drawing
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            bbox = (x1, y1, x2, y2)
            
            # Add to tracking history
            self.person_tracks[track_id].append((timestamp, bbox, centroid))
            
            # Check for loitering
            is_loitering = self.check_loitering(track_id, frame, timestamp)
            if is_loitering:
                detection_results["loitering_detected"] = True
                detection_results["loitering_ids"].append(int(track_id))
            
            # Add detection info to results
            detection_results["detections"].append({
                "id": int(track_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "centroid": [int(x_center), int(y_center)],
                "is_loitering": is_loitering,
                "time_tracked": self.get_track_duration(track_id)
            })
            
            # Generate random color for this track if it doesn't exist
            if track_id not in self.track_colors:
                self.track_colors[track_id] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            
            # Visualize detection and tracking
            frame = self.visualize_track(frame, track_id)
        
        # Draw ROI if defined
        if self.roi_coords is not None:
            cv2.polylines(frame, [self.roi_coords], True, (0, 255, 0), 2)
        
        return frame, detection_results
    
    def get_track_duration(self, track_id):
        """Get the duration a person has been tracked in seconds."""
        track_history = self.person_tracks[track_id]
        if len(track_history) < 2:
            return 0.0
            
        first_detection_time = track_history[0][0]
        latest_detection_time = track_history[-1][0]
        return latest_detection_time - first_detection_time
        
    def check_loitering(self, track_id, frame, current_time):
        """Check if a tracked person is loitering. Returns True if loitering."""
        track_history = self.person_tracks[track_id]
        
        if len(track_history) < 2:
            return False
            
        first_detection_time = track_history[0][0]
        duration = current_time - first_detection_time
        
        # Clean old tracks if too many stored
        if len(track_history) > 1000:
            self.person_tracks[track_id] = track_history[-1000:]
        
        # Check if duration exceeds threshold
        if duration >= self.loitering_threshold:
            if track_id not in self.loitering_alerts:
                self.loitering_alerts.add(track_id)
                
                # Log alert
                print(f"⚠️ LOITERING ALERT: Person (ID: {track_id}) loitering for {duration:.2f} seconds")
                
                # Save alert image if output directory specified
                if self.output_dir:
                    timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S")
                    alert_filename = f"{self.output_dir}/loitering_alert_{track_id}_{timestamp_str}.jpg"
                    _, bbox, _ = track_history[-1]
                    x1, y1, x2, y2 = bbox
                    
                    # Draw red box around loitering person
                    alert_frame = frame.copy()
                    cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(alert_frame, f"LOITERING: {duration:.1f}s", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.imwrite(alert_filename, alert_frame)
                    print(f"Alert image saved to {alert_filename}")
            return True
        return False
    
    def visualize_track(self, frame, track_id):
        """Visualize tracking and loitering information on the frame."""
        track_history = self.person_tracks[track_id]
        color = self.track_colors[track_id]
        
        # Get latest detection
        _, latest_bbox, latest_centroid = track_history[-1]
        x1, y1, x2, y2 = latest_bbox
        
        # Draw bounding box
        box_color = (0, 0, 255) if track_id in self.loitering_alerts else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Calculate time since first detection
        first_detection_time = track_history[0][0]
        latest_time = track_history[-1][0]
        duration = latest_time - first_detection_time
        
        # Add ID and duration text
        status_text = f"ID:{track_id} Time:{duration:.1f}s"
        if track_id in self.loitering_alerts:
            status_text = f"LOITERING: {status_text}"
            
        cv2.putText(frame, status_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw track line showing movement history
        if self.debug and len(track_history) >= 2:
            points = [pt[2] for pt in track_history[-20:]]  # Get last 20 centroid points
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources if needed."""
        # Remove old tracks that haven't been updated
        current_time = time.time()
        remove_tracks = []
        
        for track_id, history in self.person_tracks.items():
            last_update_time = history[-1][0]
            if current_time - last_update_time > 10:  # 10 seconds timeout
                remove_tracks.append(track_id)
                if track_id in self.loitering_alerts:
                    self.loitering_alerts.remove(track_id)
        
        for track_id in remove_tracks:
            del self.person_tracks[track_id]
            if track_id in self.track_colors:
                del self.track_colors[track_id]
                
    def process_image(self, image_data):
        """Process a single image for loitering detection.
        
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
            
        # Process the frame
        processed_frame, results = self.process_frame(frame)
        
        return processed_frame, results
        
    def encode_image_to_base64(self, image):
        """Convert an OpenCV image to base64 encoding."""
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

# # For testing purposes
# def test_loitering_detector():
#     config = {
#         "model": "yolov8n.pt",
#         "loitering_threshold": 5.0,
#         "confidence": 0.3,
#         "output_dir": "output",
#         "debug": True
#     }
    
#     detector = LoiteringDetectionSystem(config)
    
#     # Test with a single image
#     test_image = "path/to/test/image.jpg"
#     if os.path.exists(test_image):
#         processed_image, results = detector.process_image(test_image)
#         print(f"Detection results: {results}")
        
#         # Save processed image
#         output_path = "output/test_processed.jpg"
#         cv2.imwrite(output_path, processed_image)
#         print(f"Processed image saved to {output_path}")

# if __name__ == "__main__":
#     test_loitering_detector()




