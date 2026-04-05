"""
AI Nurse Copilot — Pre-Visit Intelligence Briefing System
Streamlit Prototype for Apollo Homecare Case Study

Run: streamlit run app.py
"""

import streamlit as st
import chromadb
import json
import os
import time
from datetime import datetime

from data_layer import (
    get_all_patient_ids,
    get_patient_summary,
    load_all_patient_data,
    load_demographics,
)
from intelligence_layer import generate_briefing, validate_briefing

# ── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Nurse Copilot — Apollo Homecare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1rem; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    
    /* Patient card */
    .patient-card {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .patient-card h3 { margin: 0 0 0.3rem 0; color: #1a365d; }
    .patient-card p { margin: 0.2rem 0; font-size: 0.9rem; color: #4a5568; }
    
    /* Risk badges */
    .risk-critical {
        background: #fee2e2; border-left: 4px solid #dc2626;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .risk-moderate {
        background: #fef3c7; border-left: 4px solid #d97706;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .risk-low {
        background: #d1fae5; border-left: 4px solid #059669;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Feedback section */
    .feedback-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1.5rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    
    /* Data source tags */
    .source-tag {
        display: inline-block;
        background: #e2e8f0;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    
    /* Architecture note */
    .arch-note {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #92400e;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────

if "briefing" not in st.session_state:
    st.session_state.briefing = None
if "briefing_patient" not in st.session_state:
    st.session_state.briefing_patient = None
if "briefing_time" not in st.session_state:
    st.session_state.briefing_time = None
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key. Get one at console.anthropic.com",
    )

    st.divider()

    st.markdown("### 📊 System Architecture")
    st.markdown("""
    **Pipeline (deterministic workflow):**
    1. 📁 Load patient data from all sources
    2. 🗄️ Index chunks into ChromaDB
    3. 🔍 Retrieve via semantic similarity
    4. 📝 Assemble context for LLM
    5. 🤖 Generate briefing (Claude API)
    6. ✅ Validate output structure
    """)

    st.divider()

    st.markdown("### 📂 Data Sources (Mock)")
    st.markdown("""
    - **CRM/EHR** → `demographics.json`
    - **Nurse Notes** → `visit_notes.txt`
    - **WhatsApp** → `whatsapp_messages.txt`
    - **IoT Vitals** → `vitals.csv`
    - **Physician** → `physician_notes.txt`
    - **Lab Reports** → `lab_reports.json`
    """)

    st.divider()

    st.markdown("### 🛡️ Clinical Guardrails")
    st.markdown("""
    - No diagnosis — decision support only
    - All signals cite data source
    - Risk flags are severity-ranked
    - Multilingual content preserved
    - Output validated for completeness
    """)

    st.divider()

    # Feedback history
    if st.session_state.feedback_log:
        st.markdown("### 📋 Feedback Log")
        for fb in reversed(st.session_state.feedback_log[-5:]):
            emoji = "👍" if fb["rating"] == "helpful" else "👎"
            st.markdown(
                f"{emoji} **{fb['patient']}** — {fb['timestamp']}"
            )
            if fb.get("comment"):
                st.caption(f"_{fb['comment']}_")

# ── Main Content ───────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Nurse Copilot</h1>
    <p>Pre-Visit Intelligence Briefing System — Apollo Homecare</p>
</div>
""", unsafe_allow_html=True)

# Patient Selection
st.markdown("#### Select Patient")
patient_ids = get_all_patient_ids()
patient_options = {pid: get_patient_summary(pid) for pid in patient_ids}

selected_patient = st.selectbox(
    "Choose a patient for pre-visit briefing:",
    options=patient_ids,
    format_func=lambda x: patient_options[x],
    label_visibility="collapsed",
)

# ── Patient Quick Card ─────────────────────────────────────────────────────

if selected_patient:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    demo = load_demographics(os.path.join(DATA_DIR, selected_patient))

    col1, col2 = st.columns([2, 1])
    with col1:
        diagnoses_html = "".join([f"<p>• {d}</p>" for d in demo["primary_diagnoses"]])
        st.markdown(f"""
        <div class="patient-card">
            <h3>{demo['name']} — {demo['age']}{demo['gender'][0]}</h3>
            <p><strong>Preferred address:</strong> {demo.get('preferred_address', demo['name'])}</p>
            <p><strong>Diagnoses:</strong></p>
            {diagnoses_html}
            <p><strong>Allergies:</strong> {', '.join(demo['allergies'])}</p>
            <p><strong>Physician:</strong> {demo['supervising_physician']['name']} ({demo['supervising_physician']['specialization']})</p>
            <p><strong>Visit frequency:</strong> {demo['care_plan']['visit_frequency']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        med_count = len(demo["current_medications"])
        st.metric("Active Medications", med_count)
        st.metric("Family Contact", demo["primary_family_contact"]["name"])
        if "NRI" in demo["primary_family_contact"].get("notes", ""):
            st.info("🌍 NRI 360 Program", icon="✈️")

# ── Generate Briefing ─────────────────────────────────────────────────────

st.divider()

col_btn, col_info = st.columns([1, 2])
with col_btn:
    generate_clicked = st.button(
        "🧠 Generate Pre-Visit Briefing",
        type="primary",
        use_container_width=True,
        disabled=not api_key,
    )
with col_info:
    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to generate briefings.", icon="🔑")

if generate_clicked and api_key and selected_patient:
    with st.spinner(""):
        # Progress indicator showing the deterministic workflow steps
        progress = st.progress(0)
        status = st.empty()

        status.markdown("**Step 1/6:** Loading patient data from all sources...")
        progress.progress(10)
        time.sleep(0.3)

        status.markdown("**Step 2/6:** Chunking and indexing into ChromaDB (vector store)...")
        progress.progress(25)
        time.sleep(0.3)

        status.markdown("**Step 3/6:** Constructing retrieval query from patient context...")
        progress.progress(35)
        time.sleep(0.2)

        status.markdown("**Step 4/6:** Retrieving relevant chunks via semantic search (RAG)...")
        progress.progress(45)
        time.sleep(0.2)

        status.markdown("**Step 5/6:** Assembling context and generating briefing via Claude API...")
        progress.progress(60)

        try:
            briefing = generate_briefing(
                patient_id=selected_patient,
                api_key=api_key,
                chroma_client=st.session_state.chroma_client,
            )

            status.markdown("**Step 6/6:** Validating output structure...")
            progress.progress(90)

            validation = validate_briefing(briefing)
            progress.progress(100)
            time.sleep(0.3)

            # Clear progress indicators
            progress.empty()
            status.empty()

            # Store in session state
            st.session_state.briefing = briefing
            st.session_state.briefing_patient = selected_patient
            st.session_state.briefing_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.validation = validation

        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Error generating briefing: {str(e)}")
            st.info("Check that your API key is valid and has available credits.")

# ── Display Briefing ──────────────────────────────────────────────────────

if (
    st.session_state.briefing
    and st.session_state.briefing_patient == selected_patient
):
    st.divider()

    # Briefing header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"## 📄 Pre-Visit Briefing")
        st.caption(f"Generated at {st.session_state.briefing_time}")
    with col_h2:
        validation = st.session_state.get("validation", {})
        if validation.get("valid"):
            st.success("✅ All 5 sections present", icon="✅")
        else:
            missing = ", ".join(validation.get("missing_sections", []))
            st.warning(f"⚠️ Missing: {missing}")

    # Architecture transparency note
    st.markdown("""
    <div class="arch-note">
        <strong>🏗️ How this briefing was generated:</strong> 
        Patient data from 6 sources → chunked & embedded in ChromaDB → 
        top-15 relevant chunks retrieved via semantic search → 
        assembled into structured context → generated by Claude (claude-sonnet-4-20250514) 
        with clinical guardrails → output validated for section completeness.
    </div>
    """, unsafe_allow_html=True)

    # Display the briefing
    st.markdown(st.session_state.briefing)

    # ── Feedback Section ──────────────────────────────────────────────────

    st.divider()
    st.markdown("""
    <div class="feedback-box">
        <h4>💬 Nurse Feedback</h4>
        <p style="color: #64748b; font-size: 0.85rem;">
            Your feedback helps improve the AI Copilot. Rate this briefing and optionally share what was helpful or what was missed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])

    with fb_col1:
        if st.button("👍 Helpful", use_container_width=True):
            st.session_state.feedback_log.append({
                "patient": patient_options[selected_patient].split("—")[0].strip(),
                "rating": "helpful",
                "timestamp": datetime.now().strftime("%H:%M"),
                "comment": "",
            })
            st.toast("Thanks for the feedback! 👍", icon="✅")

    with fb_col2:
        if st.button("👎 Not Helpful", use_container_width=True):
            st.session_state.feedback_log.append({
                "patient": patient_options[selected_patient].split("—")[0].strip(),
                "rating": "not_helpful",
                "timestamp": datetime.now().strftime("%H:%M"),
                "comment": "",
            })
            st.toast("Thanks — we'll use this to improve.", icon="📝")

    with fb_col3:
        comment = st.text_input(
            "Optional: What was missing or inaccurate?",
            placeholder="e.g., 'Missed that patient recently fell' or 'Risk flag was too aggressive'",
            key="feedback_comment",
        )
        if comment and st.button("Submit Comment", key="submit_comment"):
            st.session_state.feedback_log.append({
                "patient": patient_options[selected_patient].split("—")[0].strip(),
                "rating": "comment",
                "timestamp": datetime.now().strftime("%H:%M"),
                "comment": comment,
            })
            st.toast("Comment recorded. Thank you!", icon="💬")

    # ── Feedback Dashboard (simple) ──────────────────────────────────────

    if st.session_state.feedback_log:
        with st.expander("📊 Feedback Dashboard"):
            total = len(st.session_state.feedback_log)
            helpful = sum(1 for f in st.session_state.feedback_log if f["rating"] == "helpful")
            not_helpful = sum(1 for f in st.session_state.feedback_log if f["rating"] == "not_helpful")
            comments = [f for f in st.session_state.feedback_log if f.get("comment")]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Ratings", total)
            m2.metric("Helpful", helpful)
            m3.metric("Not Helpful", not_helpful)
            m4.metric("Comments", len(comments))

            if comments:
                st.markdown("**Recent Comments:**")
                for c in reversed(comments[-5:]):
                    st.markdown(f"- _{c['comment']}_ ({c['timestamp']})")

# ── Footer ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem;">
    AI Nurse Copilot — Apollo Homecare Case Study Prototype<br>
    Built with Streamlit + ChromaDB + Claude API (Anthropic)<br>
    ⚠️ Prototype — not for clinical use. All patient data is simulated.
</div>
""", unsafe_allow_html=True)
