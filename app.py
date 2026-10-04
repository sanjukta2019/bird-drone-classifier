import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile


st.title("Bird & Drone Detection (YOLOv8)")

model = YOLO('runs/detect/train2/weights/best.pt')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", width="stretch")  # use_container_width instead of use_column_width

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        results = model.predict(tmp.name, save=False, show=False)

    # Show YOLO detection result
    det_img = results[0].plot()  # numpy array with boxes drawn
    st.image(det_img, caption="Detections", width="stretch")
    
    
    # After st.image(det_img, caption="Detections", use_container_width=True)
    boxes = results[0].boxes
    names = model.model.names  # {0: 'bird', 1: 'drone'}  

    st.subheader("Detected objects (conf ≥ 0.5)")
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        if conf >= 0.5:   # show only strong detections
            st.write(f"{i+1}. {names[cls_id]} (confidence: {conf:.2f})")
































# st.title("Bird & Drone Detection (YOLOv8)")

# model = YOLO('runs/detect/train2/weights/best.pt')

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded image", use_column_width=True)

#     # Save to temp file for YOLO
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#         img.save(tmp.name)
#         results = model.predict(tmp.name, save=False, show=False)

#     st.image(results[0].plot(), caption="Detections", use_column_width=True)







