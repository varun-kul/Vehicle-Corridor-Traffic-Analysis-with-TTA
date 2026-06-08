# Create test_detection.py
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('outputs/detection/yolov8n_visdrone3/weights/best.pt')

# Test on a VisDrone image
results = model('data/raw/visdrone/VisDrone2019-DET-val/images/0000026_01000_d_0000026.jpg')

# Display results
results[0].show()

# Or save
results[0].save('outputs/test_detection.jpg')

print(f"Detected {len(results[0].boxes)} vehicles")