import streamlit as st
import os
import sys
import tempfile
import zipfile
import time
from typing import List, Dict
import pandas as pd
import numpy as np
import cv2
import torch

# Import custom style
from style import apply_custom_style, section_header

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_inference import YOLOInferenceEngine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'best.pt')
CLASS_NAMES = ['Healthy', 'Benign', 'OPMD', 'OCA']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45


st.set_page_config(
    page_title='AI Assisted Disease Screening for Oral Cancer ',
    page_icon='🦷',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Apply custom style
apply_custom_style()


# --- Logo + Title ---
col1, col2 = st.columns([0.20, 0.80])
with col1:
    st.image("logo.png", width=120)  # Make sure logo.png is in the app folder
with col2:
    st.markdown(
        "<h1 style='margin:0; color:#1D3557;'>AI Assisted Disease Screening for Oral Cancer</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='margin:0; color:#1D3557;'>एआईको सहयोगमा मुखको क्यान्सर पत्ता लगाउने सुविधा</h3>",
        unsafe_allow_html=True
    )





@st.cache_resource(show_spinner=False)
def load_engine():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place best.pt in the app folder.")
    engine = YOLOInferenceEngine(model_path=MODEL_PATH, class_names=CLASS_NAMES, device=DEVICE)
    return engine


def draw_detections(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated = image_rgb.copy()
    colors = [
        (0, 255, 0),      # Healthy → Green
        (0, 204, 204),    # Benign → Teal
        (255, 165, 0),    # OPMD → Orange
        (255, 0, 0),      # OCA → Red
    ]
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cid = det['class_id']
        cname = det['class_name']
        color = colors[cid % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 10)
        label = f"{cname}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(annotated, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, y1), color, 10)
        cv2.putText(annotated, label, (x1 + 3, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return annotated


def show_single_image(engine: YOLOInferenceEngine):
    section_header("Single Image Detection", "🖼")
    uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    if uploaded is not None:
        if st.button('Run Detection', type='primary'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                result = engine.predict_single_image(
                    image_path=tmp_path,
                    conf_threshold=DEFAULT_CONF,
                    iou_threshold=DEFAULT_IOU
                )
                col1, col2 = st.columns([2, 1])
                with col1:
                    annotated = draw_detections(result['original_image'], result['detections'])
                    st.image(annotated, caption='Detections', use_container_width=True)
                with col2:
                    st.metric('Detections', result['num_detections'])
                    st.metric('Inference Time', f"{result['inference_time']:.3f}s")
                    if result['detections']:
                        rows = []
                        for i, d in enumerate(result['detections'], 1):
                            rows.append({
                                'Detection #': i,
                                'Class': d['class_name'],
                                'Confidence': f"{d['confidence']:.3f}"
                            })
                        df_single = pd.DataFrame(rows)

                        # 🎨 Apply row color coding for single image results
                        def color_single_row(row):
                            cls = row['Class']
                            if cls in CLASS_COLORS:
                                return [f'background-color: {CLASS_COLORS[cls]}; color: white'] * len(row)
                            return [""] * len(row)

                        styled_single = df_single.style.apply(color_single_row, axis=1)
                        st.dataframe(styled_single, use_container_width=True)
                    else:
                        st.info('No detections found.')
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass




def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if 'error' in r:
            rows.append({
                'Image': os.path.basename(r.get('image_path', '')),
                'Detections': 0,
                'Classes': '',
                'Confidences': '',
            })
        else:
            classes = ', '.join([d['class_name'] for d in r.get('detections', [])])
            confs = ', '.join([f"{d['confidence']:.3f}" for d in r.get('detections', [])])
            rows.append({
                'Image': os.path.basename(r['image_path']),
                'Detections': r.get('num_detections', 0),
                'Classes': classes,
                'Confidences': confs,
            })
    return pd.DataFrame(rows)



def show_batch(engine: YOLOInferenceEngine):
    section_header("Batch Detection", "📂")
    
    # --- File Upload ---
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        files = st.file_uploader(
            'Upload multiple images', 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], 
            accept_multiple_files=True
        )
    with col_up2:
        zip_file = st.file_uploader('Or upload a ZIP folder of images', type=['zip'])

    # --- Run Batch Detection ---
    if st.button('Run Batch Detection', type='primary'):
        results: List[Dict] = []
        start = time.time()

        # Temporary directory to save images
        with tempfile.TemporaryDirectory() as tdir:
            paths = []

            # Individual image files
            if files:
                for f in files:
                    p = os.path.join(tdir, f.name)
                    with open(p, 'wb') as out:
                        out.write(f.getbuffer())
                    paths.append(p)

            # ZIP file
            elif zip_file is not None:
                zip_path = os.path.join(tdir, 'upload.zip')
                with open(zip_path, 'wb') as out:
                    out.write(zip_file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tdir)
                paths = [os.path.join(tdir, f) for f in os.listdir(tdir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]

            else:
                st.warning('Please upload images or a ZIP file.')
                return

            prog = st.progress(0)
            for i, p in enumerate(paths):
                try:
                    res = engine.predict_single_image(p, conf_threshold=DEFAULT_CONF, iou_threshold=DEFAULT_IOU)
                except Exception as e:
                    res = {'image_path': p, 'error': str(e), 'detections': [], 'num_detections': 0}
                results.append(res)
                prog.progress((i + 1) / len(paths))

        elapsed = time.time() - start
        df = results_to_dataframe(results)

        # Save full batch results in session_state
        st.session_state["batch_df"] = df
        st.metric('Processed Images', len(results))
        st.metric('Total Detections', int(df['Detections'].fillna(0).sum()))
        st.metric('Elapsed Time', f"{elapsed:.2f}s")

    # --- Filter and Display ---
    if "batch_df" in st.session_state:
        show_filtered_colored_table(st.session_state["batch_df"])
        

        # --- Download Filtered CSV ---
        if "filtered_df" in st.session_state and not st.session_state.filtered_df.empty:
            filtered_csv = st.session_state.filtered_df.to_csv(index=False)
            st.download_button(
                'Download Filtered CSV', 
                data=filtered_csv, 
                file_name=f"yolo_filtered_results_{int(time.time())}.csv", 
                mime='text/csv'
            )




# Mapping classes to colors
CLASS_COLORS = {
    "Healthy": "#2ECC71",   # Green
    "Benign": "#1ABC9C",    # Teal
    "OPMD": "#F39C12",      # Orange/Amber
    "OCA": "#E74C3C"        # Red
}

CLASS_SEVERITY = ["Healthy", "Benign", "OPMD", "OCA"]

def get_row_color(row):
    """Assign row color based on highest-severity detected class."""
    classes = row['Classes']
    n_cols = len(row)
    if not classes:
        return [""] * n_cols
    detected = classes.split(", ")
    for cls in reversed(CLASS_SEVERITY):
        if cls in detected:
            return [f'background-color: {CLASS_COLORS[cls]}; color: white'] * n_cols
    return [""] * n_cols

def show_filtered_colored_table(df: pd.DataFrame):
    st.subheader("🧾 Filtered Results")

    def max_conf_in_row(conf_str):
        if not conf_str:
            return 0.0
        return max([float(c) for c in conf_str.split(", ")])

    df['Max_Confidence'] = df['Confidences'].apply(max_conf_in_row)

    all_classes = sorted({cls for row in df['Classes'] for cls in row.split(', ') if cls})
    all_classes.insert(0, "All")

    # Default session state for filtered dataframe
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = df.copy()

    # --- Compact layout for filters ---
    with st.form("filters_form"):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            selected_class = st.selectbox("Lesion Type", all_classes, key="class_filter")

        with col2:
            min_conf = st.slider("Min Confidence (%)", 0, 100, 0, 5, key="conf_filter") / 100.0

        with col3:
            min_det = st.number_input("Min Lesions", min_value=0, value=0, step=1, key="det_filter")

        with col4:
            # Center the Apply button vertically
            st.markdown("<div style='display:flex; align-items:flex-end; height:100%;'>", unsafe_allow_html=True)
            apply_filters = st.form_submit_button("Apply")
            st.markdown("</div>", unsafe_allow_html=True)


    # Apply filters only when button clicked
    if apply_filters:
        filtered_df = df.copy()
        if selected_class != "All":
            filtered_df = filtered_df[filtered_df['Classes'].str.contains(selected_class, na=False)]
        filtered_df = filtered_df[filtered_df['Detections'] >= min_det]
        filtered_df = filtered_df[filtered_df['Max_Confidence'] >= min_conf]
        st.session_state.filtered_df = filtered_df

    # Drop helper col for display
    display_df = st.session_state.filtered_df.drop(columns=['Max_Confidence'])
    styled_df = display_df.style.apply(get_row_color, axis=1)
    st.dataframe(styled_df, use_container_width=True)



def main():
    try:
        engine = load_engine()
        # st.success(f"Model loaded: best.pt | Device: {DEVICE.upper()}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    tab1, tab2 = st.tabs(['Single Image', 'Batch'])
    with tab1:
        show_single_image(engine)
    with tab2:
        show_batch(engine)

    st.markdown("---")
    st.caption("This application is intended to assist clinical screening and is not a substitute for professional diagnosis.")


if __name__ == '__main__':
    main()