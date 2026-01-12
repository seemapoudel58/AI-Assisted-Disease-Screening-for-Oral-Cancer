import os

# Get project root directory (parent of app directory)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Asset paths (relative to project root)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'assets', 'models', 'best.pt')
LOGO_PATH = os.path.join(PROJECT_ROOT, 'assets', 'images', 'logo.png')

NEPALI_IMAGES = {
    "Normal Tissue": os.path.join(PROJECT_ROOT, 'assets', 'images', 'nepali_healthy.png'),
    "Benign Lesion": os.path.join(PROJECT_ROOT, 'assets', 'images', 'nepali_benign.png'),
    "Oral Potentially Malignant Disorder": os.path.join(PROJECT_ROOT, 'assets', 'images', 'nepali_opmd.png'),
    "Oral Cavity Carcinoma": os.path.join(PROJECT_ROOT, 'assets', 'images', 'nepali_cancer.png')
}

CLASS_NAMES = ['Healthy', 'Benign', 'OPMD', 'OCA']

BBOX_COLORS = [
    (34, 139, 34),   # Forest Green for Healthy
    (255, 165, 0),   # Orange for Benign
    (255, 69, 0),    # Red-Orange for OPMD
    (220, 20, 60),   # Crimson for OCA
]

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

REFERRAL_MAPPING = {
    'Healthy': 'No Referral',
    'Benign': 'Not Immediate',
    'OPMD': 'Monitor / Specialist Referral',
    'OCA': 'Urgent Referral'
}

RISK_COLOR_MAPPING = {
    'No Risk': '#90ee90',            # green
    'Low Risk': '#ffff99',           # yellow
    'Medium Risk - Requires Monitoring': '#ffd966',  # orange
    'High Risk - Immediate Consultation Required': '#ff9999'  # red
}
