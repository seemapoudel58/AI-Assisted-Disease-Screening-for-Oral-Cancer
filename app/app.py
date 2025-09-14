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
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_inference import YOLOInferenceEngine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'best.pt')
CLASS_NAMES = ['Healthy', 'Benign', 'OPMD', 'OCA']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45

st.set_page_config(
    page_title='AI Assisted Disease Screening for Oral Cancer',
    page_icon='🦷',
    layout='wide',
    initial_sidebar_state='collapsed'
)

col1, col2 = st.columns([0.20, 0.80])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.markdown(
        "<h1 style='text-align:center;margin-top:0;'>AI Assisted Disease Screening for Oral Cancer</h1>",
        unsafe_allow_html=True
    )
st.markdown(
    "<h3 style='text-align:center;margin-top:0;'> मुखको क्यान्सर पहिचान</h3>",
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def load_engine():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place best.pt in the app folder.")
    engine = YOLOInferenceEngine(model_path=MODEL_PATH, class_names=CLASS_NAMES, device=DEVICE)
    return engine


def extract_patient_id(filename: str) -> str:
    # Extract patient ID from filename like K-014-05.jpg -> K-014
    parts = filename.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else filename


def draw_detections(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated = image_rgb.copy()
    colors = [
        (0, 255, 0),    # Healthy
        (0, 180, 0),    # Benign
        (0, 0, 255),    # OPMD
        (255, 255, 0),  # OCA
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
        text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)  # black for green
        cv2.putText(annotated, label, (x1 + 3, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    return annotated


def referral_rule(classes: List[str], confidences: List[float]) -> str:
    if 'OCA' in classes and any(c > 0.5 for c in confidences):
        return "Urgent Referral"
    elif any(cls in ['OPMD', 'Benign'] for cls in classes):
        return "Uncertain"
    else:
        return "No Referral"


# Unified PDF generator
def generate_pdf_report(df: pd.DataFrame, title="Report"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))

    elements = [Paragraph(title, styles["Title"]), table]
    doc.build(elements)
    buffer.seek(0)
    return buffer


def show_single_image(engine: YOLOInferenceEngine):
    st.subheader('Single Image')
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

                # Show detections only if present
                annotated = draw_detections(result['original_image'], result['detections']) \
                    if result['detections'] else result['original_image']
                st.image(annotated, caption='Detections', width='stretch')

                # Extract patient ID
                patient_id = extract_patient_id(uploaded.name)

                # Prepare results table
                if result['detections']:
                    classes = [d['class_name'] for d in result['detections']]
                    confs = [d['confidence'] for d in result['detections']]
                    referral = referral_rule(classes, confs)
                else:
                    classes, confs, referral = [], [], "No Referral"

                df = pd.DataFrame([{
                    'Patient ID': patient_id,
                    'Prediction': ', '.join(classes) if classes else "No detection",
                    'Confidence': ', '.join([f"{c:.3f}" for c in confs]) if confs else "-",
                    'Referral Suggestion': referral,
                    'Notes': ''
                }])

                st.markdown("**Referral Legend:** 🟥 Urgent Referral | 🟨 Uncertain | 🟩 No Referral")

                def highlight_row(row):
                    if row['Prediction'] == "No detection":
                        return ['']*len(row)
                    if row['Referral Suggestion'] == "Urgent Referral":
                        return ['background-color: red; color: white']*len(row)
                    elif row['Referral Suggestion'] == "Uncertain":
                        return ['background-color: yellow; color: black']*len(row)
                    else:
                        return ['background-color: lightgreen; color: black']*len(row)

                st.dataframe(df.style.apply(highlight_row, axis=1))

                # 🔥 Add PDF export
                pdf_file = generate_pdf_report(df, title=f"Patient Report - {patient_id}")
                st.download_button("Download PDF", pdf_file, f"{patient_id}_report.pdf", "application/pdf")

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
                'Status': 'Error',
                'Detections': 0,
                'Classes': '',
                'Confidences': '',
                'Inference Time (s)': '',
                'Error': r['error']
            })
        else:
            classes = [d['class_name'] for d in r.get('detections', [])]
            confs = [d['confidence'] for d in r.get('detections', [])]
            rows.append({
                'Patient ID': extract_patient_id(os.path.basename(r['image_path'])),
                'Prediction': ', '.join(classes) if classes else "No detection",
                'Confidence': ', '.join([f"{c:.3f}" for c in confs]) if confs else "-",
                'Referral Suggestion': referral_rule(classes, confs),
                'Notes': ''
            })
    return pd.DataFrame(rows)


def show_batch(engine: YOLOInferenceEngine):
    st.subheader('Batch (Multiple Images or ZIP)')
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        files = st.file_uploader('Upload multiple images', type=['jpg','jpeg','png','bmp','tiff'], accept_multiple_files=True)
    with col_up2:
        zip_file = st.file_uploader('Or upload a ZIP folder of images', type=['zip'])

    if st.button('Run Batch Detection', type='primary'):
        results: List[Dict] = []
        start = time.time()

        if files:
            with tempfile.TemporaryDirectory() as tdir:
                paths = []
                for f in files:
                    p = os.path.join(tdir, f.name)
                    with open(p, 'wb') as out:
                        out.write(f.getbuffer())
                    paths.append(p)
                prog = st.progress(0)
                for i, p in enumerate(paths):
                    try:
                        res = engine.predict_single_image(p, conf_threshold=DEFAULT_CONF, iou_threshold=DEFAULT_IOU)
                    except Exception as e:
                        res = {'image_path': p, 'error': str(e), 'detections': [], 'num_detections': 0}
                    results.append(res)
                    prog.progress((i + 1)/len(paths))
        elif zip_file is not None:
            with tempfile.TemporaryDirectory() as tdir:
                zip_path = os.path.join(tdir, 'upload.zip')
                with open(zip_path, 'wb') as out:
                    out.write(zip_file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tdir)
                results = engine.predict_directory(
                    input_dir=tdir,
                    conf_threshold=DEFAULT_CONF,
                    iou_threshold=DEFAULT_IOU,
                    save_results=False
                )
        else:
            st.warning('Please upload images or a ZIP file.')
            return

        elapsed = time.time() - start
        df = results_to_dataframe(results)

        st.markdown("**Referral Legend:** 🟥 Urgent Referral | 🟨 Uncertain | 🟩 No Referral")

        def highlight_row(row):
            if row['Prediction'] == "No detection":
                return [''] * len(row)
            if row['Referral Suggestion'] == "Urgent Referral":
                return ['background-color: red; color: white'] * len(row)
            elif row['Referral Suggestion'] == "Uncertain":
                return ['background-color: yellow; color: black'] * len(row)
            else:
                return ['background-color: lightgreen; color: black'] * len(row)

        st.dataframe(df.style.apply(highlight_row, axis=1))

        # st.metric('Processed Images', len(results))
        # st.metric('Elapsed Time', f"{elapsed:.2f}s")

        # CSV export
        csv = df.to_csv(index=False)
        st.download_button('Download CSV', data=csv,
                           file_name=f"yolo_results_{int(time.time())}.csv", mime='text/csv')

        # 🔥 Use unified PDF generator
        pdf_file = generate_pdf_report(df, title="Batch Report")
        st.download_button("Download PDF", pdf_file, "results.pdf", "application/pdf")


def main():
    try:
        engine = load_engine()
    except Exception as e:
        st.error(str(e))
        st.stop()

    tab1, tab2 = st.tabs(['Single Image', 'Batch'])
    with tab1:
        show_single_image(engine)
    with tab2:
        show_batch(engine)

    st.markdown("---")
    st.caption(
    "<div style='text-align: center;'>This application is intended to assist clinical screening and is not a substitute for professional diagnosis.</div>",
    unsafe_allow_html=True
)

if __name__ == '__main__':
    main()
