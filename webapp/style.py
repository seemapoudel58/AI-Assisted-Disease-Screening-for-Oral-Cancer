# style.py
import streamlit as st

def apply_custom_style():
    """Apply global custom CSS for medical-themed aesthetics"""
    st.markdown("""
        <style>
            /* General Typography */
            h1, h3 {
                text-align: center;
                margin-top: 0;
                color: #1D3557; /* navy */
            }

            /* Metrics */
            .stMetric {
                background: #F8F9FA; /* light gray */
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                border-left: 6px solid #2BBBAD; /* teal accent */
            }

            /* Buttons */
            .stButton > button { 
                border-radius: 8px; 
                background-color: #2BBBAD; /* teal */
                color: white; 
                font-weight: bold;
                padding: 8px 20px;
                border: none;
            }
            .stButton > button:hover {
                background-color: #1D3557; /* navy hover */
                color: white;
            }

            /* Divider */
            hr {
                margin: 2rem 0;
                border: 1px solid #FF6B6B; /* awareness red */
            }

            /* Tables / Dataframes */
            .stDataFrame, .stTable {
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid #eee;
            }

            /* Page container */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)


def section_header(title: str, emoji: str = "🔹"):
    """Reusable styled header"""
    st.markdown(f"<h3>{emoji} {title}</h3>", unsafe_allow_html=True)
