import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO, RTDETR
import os
import torch
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from pathlib import Path

# -----------------------------
# Paths (robust against cwd)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_model.pt"
RTDETR_WEIGHTS = WEIGHTS_DIR / "rtdetr_model.pt"

# -----------------------------
# Device
# -----------------------------
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    return torch.device("cpu"), "CPU"

# -----------------------------
# Colors
# -----------------------------
# NOTE: model label strings can differ by dataset; provide both 'Person' and 'person' to be safe.
CLASS_COLORS = {
    "person": (255, 255, 255),      # White
    "Person": (255, 255, 255),

    "gloves": (0, 128, 0),          # Green
    "goggles": (0, 128, 0),
    "helmet": (0, 128, 0),

    "no_gloves": (0, 0, 255),       # Red
    "no_goggle": (0, 0, 255),
    "no_goggles": (0, 0, 255),
    "no_helmet": (0, 0, 255),
}

def get_color(label: str):
    return CLASS_COLORS.get(label, (0, 255, 0))  # default green

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    device, device_name = get_device()
    model = YOLO(model_path)
    model.to(device)
    return model, device, device_name

@st.cache_resource
def load_rtdetr_model(model_path: str):
    device, device_name = get_device()
    model = RTDETR(model_path)
    model.to(device)
    return model, device, device_name

# -----------------------------
# Drawing / composition
# -----------------------------
def draw_boxes_on_bgr(frame_bgr: np.ndarray, results, det_model) -> np.ndarray:
    out = frame_bgr.copy()

    if results is None or len(results) == 0:
        return out

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return out

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = det_model.names[cls] if hasattr(det_model, "names") else str(cls)

        color = get_color(label)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label_text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    return out

def add_top_banner(frame_bgr: np.ndarray, title: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    banner_h = max(34, h // 18)
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.putText(out, title, (10, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def infer_annotate_bgr(
    frame_bgr: np.ndarray,
    det_model,
    det_device,
    conf: float,
    imgsz: int | None,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        results = det_model(frame_rgb, device=det_device, conf=conf, imgsz=imgsz, verbose=False)
    return draw_boxes_on_bgr(frame_bgr, results, det_model)

def side_by_side_compare(
    frame_bgr: np.ndarray,
    yolo_model, yolo_device,
    rtdetr_model, rtdetr_device,
    conf: float,
    imgsz: int | None,
) -> np.ndarray:
    left = infer_annotate_bgr(frame_bgr, yolo_model, yolo_device, conf, imgsz)
    right = infer_annotate_bgr(frame_bgr, rtdetr_model, rtdetr_device, conf, imgsz)
    left = add_top_banner(left, "YOLOv11")
    right = add_top_banner(right, "RT-DETR v1")
    return cv2.hconcat([left, right])

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -----------------------------
# UI
# -----------------------------
st.title("PPE Detection System")
st.write("Detect whether people are wearing proper personal protective equipment")

st.sidebar.header("Model Selection")
model_type = st.sidebar.radio(
    "Choose Detection Model:",
    ["YOLOv11", "RT-DETR v1", "YOLOv11 vs RT-DETR v1 (Side-by-side)"],
    help="Side-by-side comparison works for Image / Video / Webcam (may be slower)."
)

# paths info
st.sidebar.markdown("**Weights Paths**")
st.sidebar.code(f"YOLO  : {YOLO_WEIGHTS}\nRTDETR : {RTDETR_WEIGHTS}", language="text")

# Load model(s)
comparison_mode = (model_type == "YOLOv11 vs RT-DETR v1 (Side-by-side)")

if comparison_mode:
    if not YOLO_WEIGHTS.exists() or not RTDETR_WEIGHTS.exists():
        st.error("weights 폴더에 yolo_model.pt / rtdetr_model.pt 가 필요합니다.")
        st.stop()
    yolo_model, yolo_device, yolo_device_name = load_yolo_model(str(YOLO_WEIGHTS))
    rtdetr_model, rtdetr_device, rtdetr_device_name = load_rtdetr_model(str(RTDETR_WEIGHTS))
    device_name = yolo_device_name  # just for display
    st.sidebar.info("📦 Loaded: YOLOv11 + RT-DETR v1")
else:
    if model_type == "YOLOv11":
        if not YOLO_WEIGHTS.exists():
            st.error("weights/yolo_model.pt 가 필요합니다.")
            st.stop()
        model, device, device_name = load_yolo_model(str(YOLO_WEIGHTS))
        st.sidebar.info("📦 Loaded: YOLOv11")
    else:
        if not RTDETR_WEIGHTS.exists():
            st.error("weights/rtdetr_model.pt 가 필요합니다.")
            st.stop()
        model, device, device_name = load_rtdetr_model(str(RTDETR_WEIGHTS))
        st.sidebar.info("📦 Loaded: RT-DETR v1")

st.sidebar.markdown(f"<p style='font-size:11px; color:#666666;'>{device_name} utilized</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Video Upload", "Webcam (Real-time)"])

st.sidebar.subheader("Inference Options")
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Input size (imgsz)", [None, 320, 416, 512, 640], index=0, help="Lower size = faster, may reduce accuracy.")
max_width = st.sidebar.selectbox("Resize (max width) for Video/Webcam", [None, 640, 960, 1280], index=0, help="Downscale for speed. Keeps aspect ratio.")

def maybe_resize(frame_bgr: np.ndarray, max_w: int | None) -> np.ndarray:
    if max_w is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -----------------------------
# Image Upload
# -----------------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if comparison_mode:
            st.subheader("Side-by-side Comparison (Image)")
            combo_bgr = side_by_side_compare(
                img_bgr, yolo_model, yolo_device, rtdetr_model, rtdetr_device, conf_thres, imgsz
            )
            st.image(bgr_to_rgb(combo_bgr), use_container_width=True)
        else:
            st.subheader(f"Detection (Image) — {model_type}")
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                results = model(frame_rgb, device=device, conf=conf_thres, imgsz=imgsz, verbose=False)
            out_bgr = draw_boxes_on_bgr(img_bgr, results, model)
            out_bgr = add_top_banner(out_bgr, model_type)
            st.image(bgr_to_rgb(out_bgr), use_container_width=True)

# -----------------------------
# Video Upload
# -----------------------------
elif mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    process_every_n = st.selectbox(
        "Process every N frames (1=all frames, higher=faster but less precise):",
        [1, 2, 3, 5, 10],
        index=0
    )

    if uploaded_video is not None:
        input_path = OUTPUTS_DIR / "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("✅ Video uploaded successfully!")

        if st.button("🎬 Process Video"):
            temp_output = OUTPUTS_DIR / "temp_output.avi"
            output_path = OUTPUTS_DIR / ("output_video_compare.mp4" if comparison_mode else "output_video_single.mp4")

            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps) if fps and fps > 0 else 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            last_out_bgr = None

            # Writer will be initialized after we know resized dims (first frame)
            out_writer = None

            st.info(f"🎥 Processing {total_frames} frames at {fps} FPS … (Comparison={comparison_mode})")

            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_bgr = maybe_resize(frame_bgr, max_width)

                if out_writer is None:
                    h, w = frame_bgr.shape[:2]
                    out_w = w * 2 if comparison_mode else w
                    out_h = h
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    out_writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (out_w, out_h))

                if frame_count % process_every_n == 0:
                    if comparison_mode:
                        out_bgr = side_by_side_compare(
                            frame_bgr, yolo_model, yolo_device, rtdetr_model, rtdetr_device, conf_thres, imgsz
                        )
                    else:
                        out_bgr = infer_annotate_bgr(frame_bgr, model, device, conf_thres, imgsz)
                        out_bgr = add_top_banner(out_bgr, model_type)

                    last_out_bgr = out_bgr
                else:
                    out_bgr = last_out_bgr if last_out_bgr is not None else frame_bgr
                    if comparison_mode and out_bgr.shape[1] == frame_bgr.shape[1]:
                        # if we have no last result yet in compare mode, show raw side-by-side
                        out_bgr = cv2.hconcat([add_top_banner(frame_bgr, "YOLOv11"), add_top_banner(frame_bgr, "RT-DETR v1")])
                    elif (not comparison_mode) and out_bgr.shape[1] == frame_bgr.shape[1]:
                        out_bgr = add_top_banner(frame_bgr, model_type)

                out_writer.write(out_bgr)

                frame_count += 1
                progress = frame_count / max(total_frames, 1)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")

            cap.release()
            if out_writer is not None:
                out_writer.release()

            # Convert to MP4 (H.264) for browser playback, if ffmpeg exists
            status_text.text("Converting to MP4 (H.264)…")
            try:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(temp_output), "-c:v", "libx264", "-preset", "fast", "-crf", "22", str(output_path)],
                    check=True,
                    capture_output=True,
                )
                try:
                    os.remove(temp_output)
                except OSError:
                    pass
            except Exception:
                # Fallback: keep AVI if conversion fails
                output_path = temp_output

            progress_bar.empty()
            status_text.empty()

            st.success("✅ Video processing complete!")
            st.session_state.processed_video_path = str(output_path)

    if "processed_video_path" in st.session_state and os.path.exists(st.session_state.processed_video_path):
        st.write("---")
        st.write("**Processed Video:**")
        st.video(st.session_state.processed_video_path)

        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name=os.path.basename(st.session_state.processed_video_path),
                mime="video/mp4" if st.session_state.processed_video_path.endswith(".mp4") else "video/x-msvideo"
            )

# -----------------------------
# Webcam (Real-time)
# -----------------------------
elif mode == "Webcam (Real-time)":
    st.write("Real-time webcam detection using WebRTC")
    if comparison_mode:
        st.info("⚠️ Side-by-side webcam uses TWO models per frame → FPS may drop. Try lowering imgsz / resize width.")

    class PPEDetector(VideoProcessorBase):
        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = maybe_resize(img_bgr, max_width)

            if comparison_mode:
                out_bgr = side_by_side_compare(
                    img_bgr, yolo_model, yolo_device, rtdetr_model, rtdetr_device, conf_thres, imgsz
                )
            else:
                out_bgr = infer_annotate_bgr(img_bgr, model, device, conf_thres, imgsz)
                out_bgr = add_top_banner(out_bgr, model_type)

            return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=PPEDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
