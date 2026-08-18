
import streamlit as st
import tempfile
import os
from pathlib import Path
from collections import Counter

import av
import numpy as np
from ultralytics import YOLO


st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide"
)


# =========================
# Styling
# =========================

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


# =========================
# Model
# =========================

MODEL_PATH = Path("/content/best.pt")


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


# =========================
# Sidebar
# =========================

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


# =========================
# Upload Video
# =========================

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

    input_suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=input_suffix
    ) as temp_input:

        temp_input.write(
            uploaded_file.getbuffer()
        )

        input_video_path = temp_input.name


    st.subheader("🎬 Original Video")

    st.video(uploaded_file)


    # =========================
    # Start Detection
    # =========================

    if st.button(
        "🚀 Start Detection",
        type="primary",
        use_container_width=True
    ):

        try:

            # Open input video using PyAV
            input_container = av.open(
                input_video_path
            )

            video_stream = input_container.streams.video[0]

            fps = float(
                video_stream.average_rate
            ) if video_stream.average_rate else 30.0

            width = video_stream.width
            height = video_stream.height

            total_frames = (
                video_stream.frames
                if video_stream.frames
                else 0
            )


            # =========================
            # Output video
            # =========================

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_video_path = output_file.name

            output_file.close()


            output_container = av.open(
                output_video_path,
                mode="w"
            )

            output_stream = output_container.add_stream(
                "libx264",
                rate=fps
            )

            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"


            # =========================
            # UI
            # =========================

            st.subheader("🔍 Detection in Progress")

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_placeholder = st.empty()

            total_detections = Counter()

            processed_frames = 0


            # =========================
            # Process frames
            # =========================

            for frame in input_container.decode(
                video=0
            ):

                # Convert PyAV frame to RGB numpy array
                frame_rgb = frame.to_ndarray(
                    format="rgb24"
                )


                # YOLO prediction
                results = model.predict(
                    source=frame_rgb,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                # =========================
                # Count detections
                # =========================

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


                # =========================
                # Draw bounding boxes
                # =========================

                annotated_frame = result.plot()


                # result.plot() returns BGR
                # Convert BGR -> RGB
                annotated_rgb = annotated_frame[
                    :, :, ::-1
                ]


                # =========================
                # Write output frame
                # =========================

                output_frame = av.VideoFrame.from_ndarray(
                    annotated_rgb,
                    format="rgb24"
                )

                output_frame = output_frame.reformat(
                    width=width,
                    height=height,
                    format="yuv420p"
                )


                for packet in output_stream.encode(
                    output_frame
                ):

                    output_container.mux(
                        packet
                    )


                # =========================
                # Progress
                # =========================

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
                    f"{processed_frames:,}"
                    + (
                        f" / {total_frames:,}"
                        if total_frames > 0
                        else ""
                    )
                )


                # Show preview every 10 frames
                if processed_frames % 10 == 0:

                    frame_placeholder.image(
                        annotated_rgb,
                        channels="RGB",
                        use_container_width=True
                    )


            # =========================
            # Finish encoding
            # =========================

            for packet in output_stream.encode():

                output_container.mux(
                    packet
                )


            input_container.close()

            output_container.close()


            progress_bar.progress(1.0)

            status_text.success(
                "✅ Video processing completed!"
            )

            frame_placeholder.empty()


            # =========================
            # Detection Results
            # =========================

            st.subheader(
                "🎯 Detection Results"
            )


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


            # =========================
            # Processed Video
            # =========================

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


        except Exception as e:

            st.error(
                f"❌ Error during video processing:\n\n{e}"
            )


        finally:

            try:
                os.unlink(input_video_path)
            except Exception:
                pass


            try:
                if "output_video_path" in locals():
                    os.unlink(output_video_path)
            except Exception:
                pass


# =========================
# Footer
# =========================

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
