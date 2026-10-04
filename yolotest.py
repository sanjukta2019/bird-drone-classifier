from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')

# 1) Save annotated detections for all test images
results = model.predict('data/test/images', save=True)



