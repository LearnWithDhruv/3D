import streamlit as st
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import plotly.graph_objects as go
import asyncio
import sys
from datetime import datetime

# Fix event loop issue in Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Depth Estimation Functions ---
def load_midas_model():
    """Load the MiDaS model for depth estimation."""
    model_type = "DPT_Large"  # High-precision model
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.eval()
    return model

def estimate_depth(image_path):
    """Estimate depth using MiDaS and return a 2D depth map resized to original image dimensions."""
    model = load_midas_model()

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  # Resize to 512x512 for model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 512, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        depth_map = model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)

    # Resize depth map to original image dimensions
    depth_map = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    return depth_map

# --- 3D Model Generation Functions ---
def enhance_face_depth(depth_map):
    """Refine the depth map to enhance facial features."""
    # Ensure the input is uint8 for bilateralFilter
    depth_map = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    depth_map = clahe.apply(depth_map)
    return depth_map

def detect_face(image_path):
    """Detect face and return bounding box (x, y, w, h)."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, w, h
    return None

def generate_3d_model(depth_map_path, image_path):
    """Generate a refined 3D model focusing on the face area."""
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError("Error loading depth map!")

    # Get original image dimensions
    original_image = cv2.imread(image_path)
    orig_h, orig_w = original_image.shape[:2]

    # Depth map is 512x512
    depth_h, depth_w = 512, 512

    # Detect face in the original image
    face_box = detect_face(image_path)

    if face_box:
        x, y, w, h = face_box

        # Calculate scaling factors
        scale_x = depth_w / orig_w
        scale_y = depth_h / orig_h

        # Scale coordinates to depth map dimensions
        x_depth = max(0, min(depth_w - 1, int(x * scale_x)))
        y_depth = max(0, min(depth_h - 1, int(y * scale_y)))
        w_depth = max(1, int(w * scale_x))
        h_depth = max(1, int(h * scale_y))

        # Ensure the slice doesn’t exceed depth map bounds
        w_depth = min(w_depth, depth_w - x_depth)
        h_depth = min(h_depth, depth_h - y_depth)

        # Extract and enhance the face region
        face_region = depth_map[y_depth:y_depth+h_depth, x_depth:x_depth+w_depth]
        enhanced_face = enhance_face_depth(face_region)
        depth_map[y_depth:y_depth+h_depth, x_depth:x_depth+w_depth] = enhanced_face
    
    # Normalize depth values for consistency
    depth_map = depth_map / 255.0
    
    # Create mesh grid for 3D plotting
    h, w = depth_map.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    Z = cv2.GaussianBlur(depth_map, (5,5), 0)  # Additional smoothing

    return X, Y, Z

# --- Image Capture Function ---
def capture_image():
    """Capture an image from the webcam."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Could not access the camera")
        return None

    st.write("Press 'c' to capture, 'q' to exit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Press 'c' to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            os.makedirs("assets", exist_ok=True)  # Ensure folder exists
            filename = f"assets/captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Image saved: {filename}")
            break
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return filename

# --- Main Streamlit App ---
def main():
    st.title("📸 Image to 3D Face Reconstruction")

    # Ensure assets directory exists
    os.makedirs("assets", exist_ok=True)

    # Sidebar for options
    option = st.sidebar.selectbox(
        "Choose an option",
        ("Upload Image", "Capture Image from Webcam")
    )

    image_path = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image_path = os.path.join("assets", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(image_path, caption="📷 Uploaded Image", use_container_width=True)
    else:  # Capture Image from Webcam
        if st.button("📷 Start Webcam Capture"):
            image_path = capture_image()
            if image_path:
                st.image(image_path, caption="📷 Captured Image", use_container_width=True)

    # Process the image if available
    if image_path:
        if st.button("🔍 Generate Depth Map"):
            try:
                depth_map = estimate_depth(image_path)
                if depth_map is None or len(depth_map.shape) != 2:
                    st.error("❌ Error: Depth map has incorrect dimensions.")
                else:
                    depth_map_path = "assets/depth_map.png"
                    depth_map_8bit = (depth_map * 255).astype(np.uint8)
                    cv2.imwrite(depth_map_path, depth_map_8bit)
                    st.image(depth_map_path, caption="🗺️ Depth Map", use_container_width=True)
                    st.session_state["depth_map_path"] = depth_map_path
                    st.success("✅ Depth map generated!")
            except Exception as e:
                st.error(f"❌ Error generating depth map: {e}")

        if st.button("🎨 Generate 3D Model"):
            if "depth_map_path" in st.session_state:
                depth_map_path = st.session_state["depth_map_path"]
                try:
                    X, Y, Z = generate_3d_model(depth_map_path, image_path)

                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="gray")])
                    fig.update_layout(
                        title="🌀 3D Model",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Depth",
                        ),
                        margin=dict(l=0, r=0, t=30, b=0),
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ 3D model displayed!")
                except Exception as e:
                    st.error(f"❌ Error generating 3D model: {e}")
            else:
                st.warning("⚠️ Generate a depth map first.")

if __name__ == "__main__":
    main()
