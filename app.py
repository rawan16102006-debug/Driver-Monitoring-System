import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🚗 Driver Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO11s Object Detection for Driver Behavior Monitoring'
    '</div>',
    unsafe_allow_html=True
)

MODEL_PATH = Path("/content/drive/MyDrive/YOLO11_Project/baseline/weights/best.pt")

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best.pt was not found in the application directory."
        )

    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

except Exception as e:

    st.error(
        f"❌ Failed to load the model.\n\n{e}"
    )

    st.stop()


st.success("✅ YOLO11s model loaded successfully")

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.write("### Model Information")

st.sidebar.write("**Model:** YOLO11s")
st.sidebar.write("**Task:** Object Detection")
st.sidebar.write("**Input:** Driver Monitoring Video")

st.sidebar.markdown("---")

st.sidebar.write("### Classes")

for class_id, class_name in model.names.items():
    st.sidebar.write(f"• {class_name}")

st.header("🎥 Upload Driver Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload a driver monitoring video for detection."
)

if uploaded_file is not None:

    st.success(
        f"Uploaded: **{uploaded_file.name}**"
    )

    input_suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=input_suffix
    ) as temp_input:

        temp_input.write(uploaded_file.getbuffer())
        input_video_path = temp_input.name

    st.subheader("🎬 Original Video")

    st.video(uploaded_file)

    if st.button(
        "🚀 Start Detection",
        type="primary",
        use_container_width=True
    ):
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            os.unlink(input_video_path)

            st.stop()


        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0


        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_video_path = output_file.name

        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )


        if not writer.isOpened():

            cap.release()

            os.unlink(input_video_path)
            os.unlink(output_video_path)

            st.error(
                "❌ Could not create the output video."
            )

            st.stop()

        st.subheader("🔍 Detection in Progress")

        progress_bar = st.progress(0)

        status_text = st.empty()

        frame_placeholder = st.empty()

        total_detections = Counter()

        processed_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break
            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=640,
                verbose=False
            )


            result = results[0]
            if result.boxes is not None:

                for cls_id in result.boxes.cls:

                    class_id = int(
                        cls_id.item()
                    )

                    class_name = model.names[
                        class_id
                    ]

                    total_detections[
                        class_name
                    ] += 1

            annotated_frame = result.plot()

            writer.write(
                annotated_frame
            )


            processed_frames += 1

            if total_frames > 0:

                progress = (
                    processed_frames /
                    total_frames
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )

                status_text.write(
                    f"Processing frame "
                    f"{processed_frames:,} / "
                    f"{total_frames:,}"
                )

            if processed_frames % 10 == 0:

                frame_rgb = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

        cap.release()
        writer.release()

        progress_bar.progress(1.0)

        status_text.success(
            "✅ Video processing completed!"
        )

        frame_placeholder.empty()
        st.subheader("🎯 Detection Results")

        if total_detections:

            st.write(
                "### Detected Objects"
            )

            cols = st.columns(
                len(model.names)
            )

            for index, class_name in enumerate(
                model.names.values()
            ):

                count = total_detections.get(
                    class_name,
                    0
                )

                with cols[index]:

                    st.metric(
                        label=class_name,
                        value=count
                    )

        else:

            st.warning(
                "⚠️ No objects were detected "
                "using the selected confidence threshold."
            )

        st.subheader(
            "🎬 Processed Video"
        )

        with open(
            output_video_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()


        st.video(
            video_bytes
        )

        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name=(
                f"detected_{uploaded_file.name}"
            ),
            mime="video/mp4",
            use_container_width=True
        )

        try:
            os.unlink(input_video_path)
        except Exception:
            pass

        try:
            os.unlink(output_video_path)
        except Exception:
            pass

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">

    **Driver Monitoring System** 🚗
    Powered by **YOLO11s + Streamlit**

    </div>
    """,
    unsafe_allow_html=True
)
