"""
Reusable UI Components and Styling for Insurance Streamlit Dashboard.
"""
import streamlit as st
from typing import Dict, Any, List


def apply_custom_css():
    """Inject custom CSS for modern, executive dashboard styling."""
    st.markdown("""
        <style>
        /* Main Container Styling */
        .main {
            background-color: #0E131F;
            color: #F3F4F6;
        }
        
        /* Metric Card Styling */
        .kpi-card {
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.85) 0%, rgba(17, 24, 39, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            margin-bottom: 16px;
            backdrop-filter: blur(8px);
        }
        
        .kpi-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF;
            margin-bottom: 6px;
            font-weight: 600;
        }
        
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }
        
        .kpi-sub {
            font-size: 0.78rem;
            color: #10B981;
            margin-top: 4px;
        }
        
        /* Result Card */
        .result-card {
            border-radius: 14px;
            padding: 24px;
            margin-top: 20px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border-left: 6px solid #EF4444;
        }
        
        .badge-low {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .badge-medium {
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .badge-high {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .badge-critical {
            background-color: rgba(127, 29, 29, 0.35);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.6);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }
        
        .factor-card {
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid rgba(75, 85, 99, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        
        .factor-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #F9FAFB;
        }
        
        .factor-desc {
            font-size: 0.82rem;
            color: #9CA3AF;
            margin-top: 4px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", delta_color: str = "#10B981"):
    """Render a modern KPI card component."""
    sub_html = f'<div class="kpi-sub" style="color: {delta_color}">{subtext}</div>' if subtext else ""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def render_prediction_badge(risk_level: str) -> str:
    """Return appropriate HTML badge based on risk tier."""
    lvl = risk_level.upper()
    if lvl == "LOW":
        return '<span class="badge-low">LOW RISK</span>'
    elif lvl == "MEDIUM":
        return '<span class="badge-medium">MEDIUM RISK</span>'
    elif lvl == "HIGH":
        return '<span class="badge-high">HIGH RISK</span>'
    else:
        return '<span class="badge-critical">CRITICAL RISK</span>'
