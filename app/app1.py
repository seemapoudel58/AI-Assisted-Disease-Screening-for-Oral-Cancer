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

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Spacer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_inference import YOLOInferenceEngine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'best.pt')
LOGO_PATH = os.path.join(APP_DIR, 'logo.png')
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
    st.image(LOGO_PATH, width=120)
with col2:
    st.markdown("<h1 style='margin:0;'>AI Assisted Disease Screening for Oral Cancer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;margin-top:0;'> एआईको सहयोगमा मुखको क्यान्सर पत्ता लगाउने सुविधा</h3>", unsafe_allow_html=True)

# ------------------- Load Model -------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place best.pt in the app folder.")
    engine = YOLOInferenceEngine(model_path=MODEL_PATH, class_names=CLASS_NAMES, device=DEVICE)
    return engine

# ------------------- Draw Detections -------------------
def draw_detections_rgb(image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated = image_rgb.copy()
    colors_list = [
        (34, 139, 34),   # Forest Green for Healthy
        (255, 165, 0),   # Orange for Benign
        (255, 69, 0),    # Red-Orange for OPMD
        (220, 20, 60),   # Crimson for OCA
    ]
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cid = det['class_id']
        cname = det['class_name']
        color = colors_list[cid % len(colors_list)]
        thickness = max(2, min(image_rgb.shape[:2]) // 300)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{cname}: {conf*100:.1f}%"
        font_scale = max(0.6, min(image_rgb.shape[:2]) / 1200)
        font_thickness = max(1, thickness // 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, max(y1 - text_height - 8, 0)), (x1 + text_width + 8, y1), color, -1)
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
        cv2.putText(annotated, label, (x1 + 4, max(y1 - 4, text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    return annotated

# ------------------- Referral Mapping & color map -------------------
referral_mapping = {
    'Healthy':'No Referral',
    'Benign':'No Referral',
    'OPMD':'Monitor / Specialist Referral',
    'OCA':'Urgent Referral'
}

risk_color_mapping = {
    'No Risk':'#90ee90',            # green
    'Low Risk':'#ffff99',           # yellow
    'Medium Risk - Requires Monitoring':'#ffd966',  # orange
    'High Risk - Immediate Consultation Required':'#ff9999' # red
}

# ------------------- Helpers -------------------
def get_patient_id_from_filename(filename: str) -> str:
    """Extract patient id from filename like K-025-03.jpg -> K-025"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('-')
    if len(parts) >= 2:
        pid = '-'.join(parts[:-1])
    else:
        pid = base
    return pid

# ------------------- PDF Generation (Landscape, plain table) -------------------
def generate_pdf_single(patient_info: dict, report_df: pd.DataFrame, annotated_image: np.ndarray, file_path: str):
    doc = SimpleDocTemplate(file_path, pagesize=landscape(A4))
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Clinical Assessment Report", styles['Title']))
    elements.append(Spacer(1, 5))  # adds 5 points vertical gap
    tmp_img_path = None
    if annotated_image is not None:
        tmp_img_path = file_path.replace('.pdf','_temp.png')
        cv2.imwrite(tmp_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        elements.append(RLImage(tmp_img_path, width=300, height=200))
        elements.append(Spacer(1,12))

    # ensure referral column
    if 'Referral' not in report_df.columns and 'AI Diagnosis' in report_df.columns:
        report_df['Referral'] = report_df['AI Diagnosis'].map(referral_mapping)

    col_count = len(report_df.columns)
    page_width = landscape(A4)[0] - 2*30
    default_width = page_width / max(1, col_count)
    col_widths = [default_width]*col_count

    data = [[Paragraph(str(c), styles['Normal']) for c in report_df.columns]]
    for row in report_df.values.tolist():
        data.append([Paragraph(str(cell), styles['Normal']) for cell in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')
    ])
    t.setStyle(ts)
    elements.append(t)
    doc.build(elements)
    if tmp_img_path and os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

def generate_pdf_batch(report_df: pd.DataFrame, file_path: str):
    doc = SimpleDocTemplate(file_path, pagesize=landscape(A4))
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Batch Analysis Report", styles['Title']))
    elements.append(Spacer(1,12))

    # drop Status and Inference Time if present (we keep unified schema)
    pdf_df = report_df.drop(columns=[c for c in ['Status','Inference Time (s)'] if c in report_df.columns])

    if 'Referral' not in pdf_df.columns and 'AI Diagnosis' in pdf_df.columns:
        pdf_df['Referral'] = pdf_df['AI Diagnosis'].map(referral_mapping)

    col_count = len(pdf_df.columns)
    page_width = landscape(A4)[0] - 2*30
    default_width = page_width / max(1, col_count)
    col_widths = [default_width]*col_count

    data = [[Paragraph(str(c), styles['Normal']) for c in pdf_df.columns]]
    for row in pdf_df.values.tolist():
        data.append([Paragraph(str(cell), styles['Normal']) for cell in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')
    ])
    t.setStyle(ts)
    elements.append(t)
    doc.build(elements)

# ------------------- Results -> standardized report rows -------------------
def results_to_report_rows(results: List[Dict], include_patient_fields: bool = False, patient_name: str = '', patient_age: int = 0, patient_id: str = '') -> List[Dict]:
    """
    Convert results (from single or batch) into standardized report rows.
    include_patient_fields: if True, include Patient Name/Age/Patient ID (used for single mode).
    For batch mode, patient_id must be extracted from filename and passed in patient_id param.
    """
    rows = []
    finding_counter = 1
    # mappings
    class_mapping = {'Healthy':'Normal Tissue','Benign':'Benign Lesion','OPMD':'Oral Potentially Malignant Disorder','OCA':'Oral Cavity Carcinoma'}
    risk_mapping = {'Healthy':'No Risk','Benign':'Low Risk','OPMD':'Medium Risk - Requires Monitoring','OCA':'High Risk - Immediate Consultation Required'}
    recommendation_mapping = {'Healthy':'Regular oral hygiene maintenance','Benign':'Routine follow-up recommended','OPMD':'Specialist referral and biopsy consideration','OCA':'Immediate oncology consultation required'}

    for r in results:
        img_path = r.get('image_path', '')
        base_name = os.path.basename(img_path) if img_path else ''
        extracted_pid = get_patient_id_from_filename(base_name) if base_name else ''
        # if result contains an explicit error
        if 'error' in r and r['error']:
            # create a single row marking analysis error (do not color)
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'AI Diagnosis': 'Analysis Error',
                'Confidence Level': '',
                'Risk Assessment': '',
                'Clinical Recommendation': r.get('error', ''),
                'Referral': '',
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            rows.append(row)
            finding_counter += 1
            continue

        detections = r.get('detections', []) or []
        if len(detections) == 0:
            # no detections -> normal tissue row
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'AI Diagnosis': 'Normal Oral Tissue',
                'Confidence Level': 'N/A',
                'Risk Assessment': 'No Risk Detected',
                'Clinical Recommendation': 'Continue routine oral care and regular dental check-ups',
                'Referral': 'No Referral',
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            rows.append(row)
            finding_counter += 1
            continue

        # one row per detection
        for det in detections:
            det_name = det.get('class_name', '')
            conf = det.get('confidence', 0.0)
            row = {
                'S.No': finding_counter,
                'Patient Name': patient_name if include_patient_fields else '',
                'Age': patient_age if include_patient_fields else '',
                'Patient ID': patient_id if include_patient_fields else (extracted_pid or ''),
                'AI Diagnosis': class_mapping.get(det_name, det_name),
                'Confidence Level': f"{conf*100:.1f}%",
                'Risk Assessment': risk_mapping.get(det_name, 'Unknown'),
                'Clinical Recommendation': recommendation_mapping.get(det_name, 'Consult specialist'),
                'Referral': referral_mapping.get(det_name, 'Consult specialist'),
                'Date': time.strftime('%Y-%m-%d'),
                'Time': time.strftime('%H:%M:%S')
            }
            rows.append(row)
            finding_counter += 1

    return rows

# ------------------- Highlight function for Streamlit -------------------
def highlight_risk(row):
    """
    Returns a list of CSS strings for the given row.
    Skip highlighting when AI Diagnosis indicates error or Confidence is empty/N/A.
    """
    ai_diag = str(row.get('AI Diagnosis', '')).lower()
    conf = str(row.get('Confidence Level', '')).strip()
    # skip coloring for analysis errors or empty confidence
    if 'error' in ai_diag or conf == '' or conf.upper() == 'N/A':
        return [''] * len(row)
    risk = str(row.get('Risk Assessment', 'No Risk'))
    color = risk_color_mapping.get(risk, "#a1f09a")
    # return same style for all columns in row
    return [f'background-color: {color}; color: black'] * len(row)

# ------------------- Single Image Mode -------------------
def show_single_image(engine: YOLOInferenceEngine):
    st.subheader('Single Image Analysis')
    col_p1, col_p2, col_p3 = st.columns([2,1,1])
    with col_p1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name", key="patient_name")
    with col_p2:
        patient_age = st.number_input("Age", min_value=0, max_value=120, step=1, value=0, key="patient_age")
    with col_p3:
        patient_id = st.text_input("Patient ID", placeholder="Optional", key="patient_id")

    uploaded = st.file_uploader('Upload oral cavity image', type=['jpg','jpeg','png','bmp','tiff'])
    if uploaded is not None:
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            run_detection = st.button('🔍 Run AI Analysis', type='primary', use_container_width=True)

        if run_detection:
            if not patient_name:
                st.error("Please enter patient name before running analysis.")
                return

            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("🔬 Analyzing image..."):
                    # Run inference
                    result = engine.predict_single_image(tmp_path, conf_threshold=DEFAULT_CONF, iou_threshold=DEFAULT_IOU)

                    original_image = result.get('original_image', None)
                    if original_image is None:
                        st.error("Invalid image format")
                        return

                    annotated_image = draw_detections_rgb(original_image, result.get('detections', [])) if result.get('detections') else original_image

                    st.markdown("### Analysis Results")
                    col1, col2, col3 = st.columns([0.5,2,0.5])
                    with col2:
                        st.image(annotated_image, caption=f'AI Analysis - Patient: {patient_name}', width=500)

                # Heavy step: Prepare report + generate PDF + CSV
                with st.spinner("📄 Preparing report for download..."):
                    # Build standardized report rows
                    rows = results_to_report_rows([result], include_patient_fields=True, 
                                                  patient_name=patient_name, patient_age=patient_age, patient_id=patient_id)
                    df = pd.DataFrame(rows)

                    # Display dataframe with highlights
                    st.dataframe(df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

                    # CSV Download
                    csv_file_name = f"medical_report_{patient_name.replace(' ','_')}_{int(time.time())}.csv"
                    st.download_button(
                        label="📄 Download Medical Report (CSV)",
                        data=df.to_csv(index=False),
                        file_name=csv_file_name,
                        mime='text/csv',
                        use_container_width=True
                    )

                    # PDF Download
                    pdf_file_path = f"medical_report_{patient_name.replace(' ','_')}_{int(time.time())}.pdf"
                    generate_pdf_single(
                        patient_info={'name':patient_name,'age':patient_age,'id':patient_id},
                        report_df=df,
                        annotated_image=annotated_image,
                        file_path=pdf_file_path
                    )
                    with open(pdf_file_path,'rb') as f:
                        st.download_button(
                            label="📄 Download Medical Report (PDF)",
                            data=f.read(),
                            file_name=pdf_file_path,
                            mime='application/pdf',
                            use_container_width=True
                        )

            finally:
                try: os.unlink(tmp_path)
                except Exception: pass

# ------------------- Batch Mode -------------------
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
                        # keep same shape as engine result but with error key
                        res = {'image_path': p, 'error': str(e), 'detections': [], 'num_detections': 0}
                    results.append(res)
                    prog.progress((i + 1) / len(paths))
        elif zip_file is not None:
            with tempfile.TemporaryDirectory() as tdir:
                zip_path = os.path.join(tdir, 'upload.zip')
                with open(zip_path, 'wb') as out:
                    out.write(zip_file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tdir)
                results = engine.predict_directory(input_dir=tdir, conf_threshold=DEFAULT_CONF, iou_threshold=DEFAULT_IOU, save_results=False)
        else:
            st.warning('Please upload images or a ZIP file.')
            return

        elapsed = time.time() - start

        # Build standardized report rows across all results (batch)
        batch_rows = results_to_report_rows(results, include_patient_fields=False)
        df = pd.DataFrame(batch_rows)

        # After creating df in show_batch
        if 'Patient Name' in df.columns:
            df = df.drop(columns=['Patient Name'])
        if 'Age' in df.columns:
            df = df.drop(columns=['Age'])


        st.metric('Processed Images', len(results))
        st.metric('Total Findings', len(df))
        st.metric('Elapsed Time', f"{elapsed:.2f}s")

        # Display color-coded in Streamlit only
        st.dataframe(df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

        # CSV Download (same standardized table)
        st.download_button('Download CSV', data=df.to_csv(index=False), file_name=f"yolo_results_{int(time.time())}.csv", mime='text/csv')

        # PDF Download (plain)
        batch_pdf_path = f"yolo_batch_report_{int(time.time())}.pdf"
        generate_pdf_batch(df, batch_pdf_path)
        with open(batch_pdf_path,'rb') as f:
            st.download_button("📄 Download Batch Report (PDF)", data=f.read(), file_name=batch_pdf_path, mime='application/pdf')

# ------------------- Main -------------------
def main():
    try:
        engine = load_engine()
    except Exception as e:
        st.error(str(e))
        st.stop()
    tab1, tab2 = st.tabs(['Single Image','Batch'])
    with tab1: show_single_image(engine)
    with tab2: show_batch(engine)
    st.markdown("---")
    st.caption(
    "<div style='text-align: center;'>This application is intended to assist clinical screening and is not a substitute for professional diagnosis.</div>",
    unsafe_allow_html=True)

if __name__ == '__main__':
    main()
