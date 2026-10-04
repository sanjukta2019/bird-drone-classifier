from ultralytics import YOLO

# Initialize YOLOv8n (Nano), or use YOLOv8s/m/l/x for larger models
model = YOLO('yolov8n.pt')  # You can substitute with 'yolov8s.pt' etc.

# Train with your config file
results = model.train(data='data.yaml', epochs=50, imgsz=640)
# Validate on the validation set after training
val_results = model.val()





