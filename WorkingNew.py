import gradio as gr
import cv2
import numpy as np
import imutils
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")
model.fuse()

# ---------------- HISTORY STORAGE ----------------
history = []

# ---------------- PROCESS FRAME ----------------
def process_frame(frame, conf_thresh, dist_thresh):

    frame = imutils.resize(frame, width=480)
    h, w, _ = frame.shape

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=conf_thresh,
        verbose=False
    )[0]

    boxes = []
    centroids = []

    if results.boxes is not None:
        for r in results.boxes.data.tolist():

            if len(r) == 7:
                x1, y1, x2, y2, track_id, conf, cls = r
            else:
                x1, y1, x2, y2, conf, cls = r

            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                boxes.append((x1, y1, x2, y2))
                centroids.append((cx, cy))

    violate = set()

    if len(centroids) >= 2:
        D = dist.cdist(np.array(centroids), np.array(centroids), metric="euclidean")

        for i in range(len(D)):
            for j in range(i + 1, len(D)):
                if D[i, j] < dist_thresh:
                    violate.add(i)
                    violate.add(j)

    overlay = frame.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 0, 255) if i in violate else (0, 255, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    total = len(boxes)
    viol = len(violate)
    safe = total - viol

    safety_score = (safe / total * 100) if total > 0 else 100
    density = (total / (h * w)) * 100000

    analytics = {
        "Status Metrics": {
            "Total Pedestrians": total,
            "Safety Score": f"{int(safety_score)}%",
            "Crowd Density Index": round(density, 2)
        },
        "Safety Breakdown": {
            "Safe People": safe,
            "Violations": viol,
            "Risk Level": "CRITICAL" if safety_score < 50 else "HIGH" if viol > 3 else "LOW"
        },
        "System Info": {
            "Last Update": time.strftime("%H:%M:%S"),
            "Frame Resolution": f"{w}x{h}"
        }
    }

    return frame, analytics


# ---------------- VIDEO STREAM ----------------
def video_stream(video, conf_thresh, dist_thresh):

    if video is None:
        yield None, {}

    cap = cv2.VideoCapture(video)
    last_analytics = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, analytics = process_frame(frame, conf_thresh, dist_thresh)
            last_analytics = analytics

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), analytics
            time.sleep(0.01)

    finally:
        cap.release()

    # store history after video ends
    if last_analytics:
        history.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analytics": last_analytics
        })

    yield None, {}


# ---------------- WEBCAM STREAM ----------------
def webcam_stream(_, conf_thresh, dist_thresh):

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, analytics = process_frame(frame, conf_thresh, dist_thresh)

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), analytics

    finally:
        cap.release()

    yield None, {}


# ---------------- HISTORY FUNCTIONS ----------------
def get_history():
    return history[::-1]


def delete_last_history():
    if history:
        history.pop()
    return history[::-1]


def clear_history():
    history.clear()
    return []


# ---------------- UI ----------------
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }

    img, video {
        max-height: 420px !important;
        object-fit: contain !important;
        border-radius: 8px;
    }
    """
) as demo:

    gr.Markdown("# 🦠 Social Distancing AI System")

    with gr.Tabs():

        # ================= VIDEO =================
        with gr.Tab("📹 Video Upload"):

            video_input = gr.Video()

            conf_v = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="Confidence")
            dist_v = gr.Slider(20, 200, value=80, step=5, label="Distance Threshold")

            video_btn = gr.Button("Analyze Video")

            with gr.Row():
                with gr.Column(scale=2):
                    video_output = gr.Image(label="Live Feed")

                with gr.Column(scale=1):
                    analytics_box = gr.JSON(label="Analytics Dashboard")

            video_btn.click(
                video_stream,
                inputs=[video_input, conf_v, dist_v],
                outputs=[video_output, analytics_box]
            )

        # ================= WEBCAM =================
        with gr.Tab("📷 Webcam"):

            webcam_btn = gr.Button("Start Webcam")

            conf_w = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="Confidence")
            dist_w = gr.Slider(20, 200, value=80, step=5, label="Distance Threshold")

            with gr.Row():
                with gr.Column(scale=2):
                    webcam_output = gr.Image(label="Live Feed")

                with gr.Column(scale=1):
                    webcam_stats = gr.JSON(label="Analytics Dashboard")

            webcam_btn.click(
                webcam_stream,
                inputs=[gr.State(None), conf_w, dist_w],
                outputs=[webcam_output, webcam_stats]
            )

        # ================= HISTORY =================
        with gr.Tab("🕘 History"):

            history_btn = gr.Button("🔄 Refresh History")
            delete_btn = gr.Button("🗑️ Delete Last Entry", variant="stop")
            clear_btn = gr.Button("🧹 Clear All History", variant="secondary")

            history_box = gr.JSON(label="Past Video Analytics")

            history_btn.click(get_history, outputs=history_box)
            delete_btn.click(delete_last_history, outputs=history_box)
            clear_btn.click(clear_history, outputs=history_box)


demo.launch(share=True)