%%writefile app.py
import io
import json
import os
import sqlite3
import time
import bcrypt
import docx
from fpdf import FPDF
import google.generativeai as genai
from PyPDF2 import PdfReader
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List, Dict
from typing import List, Dict
from mistralai import Mistral
import requests
import pytesseract
from pdf2image import convert_from_bytes

# ─── Configuration ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get(
    "GOOGLE_API_KEY",
    "AIzaSyANbVVzZACnYnus00xwwRRE01n34yoAmcU"  # fallback for dev/testing
)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "DUW9f3t6nvZaNkEbxcrxYP4hLIrC3g7Y")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

BRAND_COLORS = {
    "primary": "#2E86AB",
    "secondary": "#F18F01",
    "background": "#F7F7F7",
    "text": "#121111"
}

genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ─── Database Helpers ──────────────────────────────────────────────────────────
def init_db(db_path: str = "users.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    # Create or migrate users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB NOT NULL,
            role TEXT NOT NULL,
            location_id TEXT,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("PRAGMA table_info(users)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    for col in ["location_id", "last_login", "created_at"]:
        if col not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
    # Interactions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            feature TEXT,
            input_text TEXT,
            output_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
        )
        """
    )
    conn.commit()
    return conn


def create_default_admin(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_pwd = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", admin_pwd, "admin"),
        )
        conn.commit()


def verify_password(hashed: bytes, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

# ─── Authentication UI ─────────────────────────────────────────────────────────
def login_ui(conn):
    """Plain‐style login UI with minimal styling and a welcome banner."""
    # ─── Sidebar Styling ───────────────────────────────────────
    st.sidebar.title("🔑 Login / Register")
    st.sidebar.markdown(
        """
        <style>
            .sidebar .sidebar-content {
                background-color: #FFFFFF !important;
                color: #000000 !important;
                box-shadow: none !important;
            }
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea {
                background-color: #FFFFFF !important;
                color: #000000 !important;
                border: 1px solid #CCCCCC !important;
            }
            .stButton>button {
                background-color: #FFFFFF !important;
                color: #000000 !important;
                border: 1px solid #CCCCCC !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])

    # ─── LOGIN TAB ────────────────────────────────────────────────
    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        location_id = st.text_input("Location ID (optional)", key="login_location")

        if st.button("Log In", key="login_button"):
            if not username or not password:
                st.sidebar.error("Enter both username and password")
            else:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(users)")
                cols = [c[1] for c in cursor.fetchall()]

                cursor.execute(
                    "SELECT password, role FROM users WHERE username = ?",
                    (username,)
                )
                row = cursor.fetchone()

                if row and bcrypt.checkpw(password.encode(), row[0]):
                    # set session
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = row[1]
                    st.session_state.location_id = location_id or None
                    # update last_login if exists
                    if "last_login" in cols:
                        cursor.execute(
                            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
                            (username,)
                        )
                        conn.commit()
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
                    time.sleep(1)

    # ─── REGISTER TAB ─────────────────────────────────────────────
    with register_tab:
        if st.session_state.get("role") == "admin":
            new_user = st.text_input("New Username", key="reg_username")
            new_pass = st.text_input("New Password", type="password", key="reg_password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
            user_role = st.selectbox("Role", ["user", "admin"], key="reg_role")

            if st.button("Create User", key="reg_button"):
                if not new_user or not new_pass:
                    st.error("Username and password are required")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA table_info(users)")
                        cols = [c[1] for c in cursor.fetchall()]

                        if "location_id" in cols:
                            cursor.execute(
                                "INSERT INTO users (username, password, role, location_id) VALUES (?, ?, ?, NULL)",
                                (new_user, hashed, user_role)
                            )
                        else:
                            cursor.execute(
                                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                                (new_user, hashed, user_role)
                            )
                        conn.commit()
                        st.success(f"User '{new_user}' created successfully.")
                        time.sleep(1)
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That username already exists")
        else:
            st.info("Only admins can register new users.")

    # ─── MAIN PANE WELCOME BANNER ────────────────────────────────
    if not st.session_state.get("logged_in"):
        st.markdown(
            """
            <div style="
                height: 80vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            ">
                <h1 style="color: #2E86AB; font-weight: normal; margin-bottom: 0.2em;">
                    Welcome to Finchat AI Bot
                </h1>
                <p style="color: #555555; font-size: 1.1em; margin-top: 0;">
                    🤖 Powered by Alphax  — crafting real estate insights in seconds!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

# ─── AI Helper Functions ───────────────────────────────────────────────────────
def call_gemini(feature: str, content: str) -> str:
    system_prompts = {
        "lease_summary": (
            "You are a real estate document expert. Analyze the provided lease agreement "
            "and provide a comprehensive summary, including key terms and potential risks."
        ),
        "deal_strategy": (
            "You are a creative real estate strategist. Based on the provided deal details, "
            "suggest structuring options with pros, cons, and negotiation tactics."
        ),
        "offer_generator": (
            "You are a real estate transaction specialist. Generate a professional purchase offer "
            "with all essential clauses formatted for the jurisdiction."
        ),
        "chatbot": (
            "You are a knowledgeable assistant that answers questions based on the user's past interactions."
        )
    }
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"SYSTEM: {system_prompts.get(feature)}\n\nUSER: {content}"
    response = model.generate_content(prompt)
    return response.text


def call_mistral(messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 1.0, max_tokens: int = None) -> str:
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def save_interaction(conn, feature: str, input_text: str, output_text: str):
    if st.session_state.get("username"):
        conn.execute(
            "INSERT INTO interactions (username, feature, input_text, output_text) VALUES (?, ?, ?, ?)",
            (st.session_state.username, feature, input_text, output_text),
        )
        conn.commit()
# ─── Summarization Utility ─────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into manageable chunks for AI summarization."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ─── OCR Extraction ─────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file, ocr_threshold=100):
    # Read raw bytes
    raw = uploaded_file.read()
    text = ""
    # 1) Try PyPDF2
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""
    # 2) If still too short, try pdfplumber
    if len(text.strip()) < ocr_threshold:
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = []
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
                alt_text = "\n".join(pages)
                if len(alt_text.strip()) > len(text.strip()):
                    text = alt_text
        except Exception:
            pass
    # 3) If still below threshold, use OCR
    if len(text.strip()) < ocr_threshold:
        images = convert_from_bytes(raw)
        ocr_text = []
        for img in images:
            ocr_text.append(pytesseract.image_to_string(img))
        text = "\n".join(ocr_text)
    return text

# Main UI function

def lease_summarization_ui(conn):
    """Enhanced lease summarization with multi-stage AI processing and persistent output"""
    st.header("📄 Advanced Lease Summarization")
    st.markdown("Upload lease documents (PDF/DOCX) for comprehensive AI-powered analysis.")

    # Initialize persistent outputs
    if "lease_docs" not in st.session_state:
        st.session_state["lease_docs"] = None
        st.session_state["lease_results"] = None
        st.session_state["lease_comparison"] = None

    uploaded_files = st.file_uploader(
        "Upload Lease Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="lease_upload"
    )
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Basic", "Standard", "Detailed", "Comprehensive"],
                value="Standard",
                key="lease_depth"
            )
            legal_jurisdiction = st.selectbox(
                "Legal Jurisdiction",
                ["General", "UK", "US", "EU", "Other"],
                index=0,
                key="lease_jurisdiction"
            )
        with col2:
            highlight_risks = st.checkbox("Highlight Potential Risks", value=True, key="lease_risks")
            compare_clauses = st.checkbox("Compare Similar Clauses", value=True, key="lease_compare")
    ai_model = st.radio("AI Model", ["Gemini", "Mistral"], horizontal=True, key="lease_model")

    # Analyze button
    if st.button("Analyze Documents", key="lease_analyze"):
        if not uploaded_files:
            st.error("Please upload at least one document.")
            return
        st.session_state["lease_docs"] = uploaded_files
        results = []
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            if uploaded_file.type == "application/pdf":
                text = extract_pdf_text(uploaded_file)
            else:
                text = ".join(p.text for p in docx.Document(uploaded_file).paragraphs)"

            preview = text[:5000] + ("..." if len(text) > 5000 else "")
            analysis_steps = []

            steps = [
                ("Initial Document Structure Analysis", "Identify and summarize the overall structure of the document."),
                ("Key Clause Extraction", "Extract and categorize all major clauses and provisions."),
                ("Obligations Mapping", "Map out all parties' obligations and responsibilities."),
                ("Financial Terms Analysis", "Identify all payment terms, amounts, and schedules."),
                ("Termination Conditions", "Summarize conditions for termination or renewal.")
            ]
            doc_out = {"name": uploaded_file.name, "sections": []}
            for title, instr in steps:
                if ai_model == "Gemini":
                    prompt = (f"Document Analysis Task: {instr}"
                        f"Legal Context: {legal_jurisdiction}"
                        f"Analysis Level: {analysis_depth}"
                        f"Document Excerpt:{text[:15000]}"
                    )
                    content = call_gemini("lease_summary", prompt)
                else:
                    msgs = [
                        {"role": "system", "content": f"You are a legal document expert specializing in {legal_jurisdiction} law. {instr}"},
                        {"role": "user", "content": f"Analyze this with {analysis_depth} detail:{text[:15000]}"}
                    ]
                    content = call_mistral(msgs)
                doc_out["sections"].append({"title": title, "content": content})
            if highlight_risks:
                if ai_model == "Gemini":
                    rp = f"Analyze risks in the document:{text[:15000]}"
                    risk = call_gemini("lease_summary", rp)
                else:
                    msgs = [
                        {"role": "system", "content": "You are a legal risk assessment specialist."},
                        {"role": "user", "content": text[:15000]}
                    ]
                    risk = call_mistral(msgs)
                doc_out["sections"].append({"title": "Risk Assessment", "content": risk})
            results.append(doc_out)
            save_interaction(conn, "lease_summary", text, str(doc_out))
        comp = None
        if compare_clauses and len(results) > 1:
            comp_text = "".join(f"DOCUMENT: {d['name']}{d['sections'][0]['content']}" for d in results)
            if ai_model == "Gemini":
                comp = call_gemini("lease_summary", f"Compare clauses:{comp_text}")
            else:
                msgs = [
                    {"role": "system", "content": "You are a legal comparison expert."},
                    {"role": "user", "content": comp_text}
                ]
                comp = call_mistral(msgs)
            save_interaction(conn, "lease_comparison", "multiple", comp)
        st.session_state["lease_results"] = results
        st.session_state["lease_comparison"] = comp

    # Display persistent results
    if st.session_state.get("lease_results"):
        st.success("Analysis Complete!")
        for doc in st.session_state["lease_results"]:
            with st.expander(f"Analysis: {doc['name']}"):
                for sec in doc["sections"]:
                    st.subheader(sec["title"])
                    st.markdown(sec["content"])
        if st.session_state.get("lease_comparison"):
            st.subheader("Cross-Document Comparison")
            st.markdown(st.session_state["lease_comparison"])
        # Export UI
        fmt = st.selectbox("Export Format", ["PDF", "Word", "JSON", "HTML"], key="lease_export_fmt")
        name = st.text_input("Export File Name", "lease_analysis", key="lease_export_name")
        if st.button("Export Analysis", key="lease_export"):
            data = {"documents": st.session_state["lease_results"], "comparison": st.session_state["lease_comparison"]}
            if fmt == "PDF":
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
                for d in st.session_state["lease_results"]:
                    pdf.set_font("Arial","B",16); pdf.cell(0,10,d['name'],0,1)
                    pdf.set_font("Arial",size=12)
                    for sec in d['sections']:
                        pdf.set_font("Arial","B",14); pdf.cell(0,10,sec['title'],0,1)
                        pdf.multi_cell(0,10,sec['content']); pdf.ln(5)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button("Download PDF", pdf_bytes, f"{name}.pdf","application/pdf")
            elif fmt == "Word":
                docx_out = docx.Document()
                for d in st.session_state["lease_results"]:
                    docx_out.add_heading(d['name'],level=1)
                    for sec in d['sections']:
                        docx_out.add_heading(sec['title'],level=2)
                        docx_out.add_paragraph(sec['content'])
                buf = io.BytesIO(); docx_out.save(buf)
                st.download_button("Download Word", buf.getvalue(), f"{name}.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            elif fmt == "JSON":
                st.download_button("Download JSON", json.dumps(data,indent=2).encode(), f"{name}.json","application/json")
            else:
                html = "<html><body>"
                for d in st.session_state["lease_results"]:
                    html += f"<h1>{d['name']}</h1>"
                    for sec in d['sections']:
                        html += f"<h2>{sec['title']}</h2><p>{sec['content']}</p>"
                html += "</body></html>"
                st.download_button("Download HTML", html.encode(), f"{name}.html","text/html")

#////////////////////////////////////////deal//////////////////////////////////////////////////
import re
import streamlit as st

def deal_structuring_ui(conn):
    """Enhanced deal structuring with persistent strategy chat until cleared."""
    st.header("💡 Creative Deal Structuring Bot")
    st.markdown("Get AI-powered strategies for your property deals")

    # Initialize memory
    if "deal_strategy_memory" not in st.session_state:
        st.session_state.deal_strategy_memory = []
        st.session_state.last_strategies = None

    # Clear chat
    if st.button("Clear Strategies", key="clear_strategies"):
        st.session_state.deal_strategy_memory.clear()
        st.session_state.last_strategies = None

    # Replay chat history
    for role, msg in st.session_state.deal_strategy_memory:
        st.chat_message(role).write(msg)

    # Input form
    with st.expander("Deal Details", expanded=True):
        property_type = st.selectbox("Property Type", ["Residential", "Commercial", "Mixed-Use", "Land"])
        deal_stage = st.selectbox("Deal Stage", ["Pre-offer", "Under Contract", "Rehab Planning", "Exit Strategy"])
        financials = st.text_area("Financial Parameters")
        market_conditions = st.text_area("Market Conditions")
        special_considerations = st.text_area("Special Considerations")

    with st.expander("Strategy Preferences"):
        col1, col2 = st.columns(2)
        with col1:
            risk_tolerance = st.select_slider("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            creativity_level = st.select_slider("Creativity Level", ["Standard", "Creative", "Outside-the-box"])
        with col2:
            timeframe = st.selectbox(
                "Investment Horizon",
                ["Short-term (0-2 years)", "Medium-term (2-5 years)", "Long-term (5+ years)"]
            )
            capital_available = st.selectbox("Capital Availability", ["Limited", "Moderate", "Substantial"])

    ai_model = st.radio("AI Model", ["Gemini", "Mistral"], horizontal=True)

    # Generate strategies
    if st.button("Generate Strategies", type="primary", key="gen_strat"):
        prompt = (
            f"Property Type: {property_type}\n"
            f"Deal Stage: {deal_stage}\n"
            f"Financial Parameters: {financials}\n"
            f"Market Conditions: {market_conditions}\n"
            f"Special Considerations: {special_considerations}\n\n"
            f"Generate {risk_tolerance.lower()} strategies with {creativity_level.lower()} approaches "
            f"for a {timeframe} investment using {capital_available.lower()} capital."
        )
        with st.spinner("Developing strategies…"):
            if ai_model == "Gemini":
                strategies = call_gemini("deal_strategy", prompt)
            else:
                messages = [
                    {"role": "system", "content": "You are a real estate investment strategist. Provide creative deal structuring options."},
                    {"role": "user",   "content": prompt}
                ]
                strategies = call_mistral(messages=messages)

        # Record and display
        st.session_state.deal_strategy_memory.append(("assistant", strategies))
        st.session_state.last_strategies = strategies
        st.chat_message("assistant").write(strategies)
        st.subheader("Recommended Strategies")
        st.markdown(strategies)

    # Strategy evaluation & refinement
    strategies = st.session_state.get("last_strategies")
    if strategies:
        # Parse individual strategies by number
        matches = re.findall(
            r"Strategy\s+(\d+):\s*(.*?)(?=(?:Strategy\s+\d+:)|\Z)",
            strategies,
            flags=re.S
        )
        if matches:
            strategy_dict = {f"Strategy {num}": text.strip() for num, text in matches}
        else:
            # Fallback if no numbered sections found
            strategy_dict = {"Strategy 1": strategies.strip()}

        labels = list(strategy_dict.keys())
        selected_label = st.selectbox("Which strategy do you prefer?", labels, key="eval_choice")
        selected_text = strategy_dict[selected_label]

        # Show the selected content
        st.markdown(f"**{selected_label}**")
        st.markdown(selected_text)

        # Use a cleaned key for per-strategy confidence
        conf_key = f"conf_{selected_label.replace(' ', '_')}"
        if conf_key not in st.session_state:
            st.session_state[conf_key] = 7
        # Slider without explicit default value argument
        confidence = st.slider(
            "Confidence in this strategy", 1, 10, key=conf_key
        )

        if st.button("Refine Strategy", key="refine_strat"):
            feedback = f"{selected_label} with confidence {confidence}/10"
            st.session_state.deal_strategy_memory.append(("user", feedback))

            refinement_prompt = (
                f"Refine this single strategy based on user feedback:\n\n"
                f"{selected_text}\n\n"
                f"Feedback: {feedback}"
            )
            if ai_model == "Gemini":
                refinement = call_gemini("deal_strategy", refinement_prompt)
            else:
                messages = [
                    {"role": "system", "content": "Refine the selected strategy based on user feedback."},
                    {"role": "user",   "content": refinement_prompt}
                ]
                refinement = call_mistral(messages=messages)

            st.session_state.deal_strategy_memory.append(("assistant", refinement))
            st.chat_message("assistant").write(refinement)
            save_interaction(conn, "deal_strategy_refinement", selected_text, refinement)

# ─── Offer Generator ───────────────────────────────────────────────────────────
def offer_generator_ui(conn):
    """Enhanced offer generator with templates"""
    st.header("✍️ Offer Generator Bot")
    st.markdown("Create professional property purchase offers")

    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Guided Form", "Free Text", "Upload Existing Offer"],
        horizontal=True
    )

    ai_model = st.radio("AI Model", ["Gemini", "Mistral"], horizontal=True)
    offer = None  # Initialize offer variable to track offer content

    if input_method == "Guided Form":
        with st.form("offer_details"):
            col1, col2 = st.columns(2)
            with col1:
                buyer_name = st.text_input("Buyer Name")
                property_address = st.text_input("Property Address")
                purchase_price = st.number_input("Purchase Price", min_value=1000)
            with col2:
                seller_name = st.text_input("Seller Name")
                closing_date = st.date_input("Proposed Closing Date")
                earnest_money = st.number_input("Earnest Money Deposit", min_value=0)

            # Additional terms
            financing = st.selectbox(
                "Financing Type",
                ["Cash", "Conventional Loan", "FHA", "VA", "Seller Financing", "Other"]
            )
            contingencies = st.multiselect(
                "Contingencies",
                ["Inspection", "Appraisal", "Financing", "Title Review", "Other"]
            )
            special_terms = st.text_area("Special Terms")

            if st.form_submit_button("Generate Offer"):
                offer_data = {
                    "buyer": buyer_name,
                    "seller": seller_name,
                    "property": property_address,
                    "price": f"${purchase_price:,}",
                    "closing_date": closing_date.strftime("%B %d, %Y"),
                    "earnest_money": f"${earnest_money:,}",
                    "financing": financing,
                    "contingencies": contingencies,
                    "special_terms": special_terms
                }
                prompt = (
                    "Generate a professional real estate purchase agreement based on these details:\n\n"
                    f"{json.dumps(offer_data, indent=2)}"
                )

                with st.spinner("Drafting offer..."):
                    if ai_model == "Gemini":
                        offer = call_gemini("offer_generator", prompt)
                    else:
                        messages = [
                            {"role": "system", "content": "You are a real estate attorney. Draft a professional purchase agreement."},
                            {"role": "user", "content": prompt}
                        ]
                        offer = call_mistral(messages=messages)
                    save_interaction(conn, "offer_generator", prompt, offer)

                st.subheader("Generated Offer")
                st.markdown(offer)

    elif input_method == "Free Text":
        free_text = st.text_area("Enter all relevant deal details")
        if st.button("Generate from Text"):
            with st.spinner("Creating offer..."):
                if ai_model == "Gemini":
                    offer = call_gemini("offer_generator", free_text)
                else:
                    messages = [
                        {"role": "system", "content": "Create a property purchase offer from these details."},
                        {"role": "user", "content": free_text}
                    ]
                    offer = call_mistral(messages=messages)
                save_interaction(conn, "offer_generator", free_text, offer)
            st.subheader("Generated Offer")
            st.markdown(offer)

    else:  # Upload Existing Offer
        uploaded = st.file_uploader("Upload Existing Offer", type=["pdf", "docx"])
        if uploaded and st.button("Analyze & Improve"):
            with st.spinner("Processing document..."):
                if uploaded.type == "application/pdf":
                    reader = PdfReader(uploaded)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    document = docx.Document(uploaded)
                    text = "\n".join(p.text for p in document.paragraphs)

                if ai_model == "Gemini":
                    analysis = call_gemini(
                        "offer_generator",
                        f"Analyze this existing offer and suggest improvements:\n\n{text}"
                    )
                else:
                    messages = [
                        {"role": "system", "content": "Review this property offer and suggest improvements."},
                        {"role": "user", "content": text}
                    ]
                    analysis = call_mistral(messages=messages)
                save_interaction(conn, "offer_analysis", text, analysis)

            st.subheader("Suggested Improvements")
            st.markdown(analysis)

    # Export options (Make sure offer is generated before this step)
    if offer:
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            export_format = st.selectbox("Export As", ["PDF", "Word", "Text"])
        with export_col2:
            export_name = st.text_input("File Name", "property_offer")

        if st.button("Export Offer"):
            if export_format == "PDF":
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in offer.split("\n"):
                    pdf.multi_cell(0, 10, line)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{export_name}.pdf",
                    mime="application/pdf",
                )
            elif export_format == "Word":
                doc = docx.Document()
                doc.add_heading("Property Purchase Offer", level=1)
                doc.add_paragraph(offer)
                bio = io.BytesIO()
                doc.save(bio)
                st.download_button(
                    "Download Word",
                    data=bio.getvalue(),
                    file_name=f"{export_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.download_button(
                    "Download Text",
                    data=offer.encode("utf-8"),
                    file_name=f"{export_name}.txt",
                    mime="text/plain",
                )

# ─── Admin Portal ──────────────────────────────────────────────────────────────
def admin_portal_ui(conn):
    """Enhanced admin portal with usage analytics"""
    st.header("🔒 Admin Portal")

    tab1, tab2, tab3 = st.tabs(["User Management", "Content Management", "Usage Analytics"])

    with tab1:
        st.subheader("User Accounts")

        # First check what columns exist in the users table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        # Build the SELECT query based on available columns
        select_columns = ["username", "role"]
        if "last_login" in columns:
            select_columns.append("last_login")
        if "location_id" in columns:
            select_columns.append("location_id")
        if "created_at" in columns:
            select_columns.append("created_at")

        query = f"SELECT {', '.join(select_columns)} FROM users"
        users = conn.execute(query).fetchall()

        # Format datetime columns for display
        formatted_users = []
        for user in users:
            formatted_user = list(user)
            for i, col in enumerate(select_columns):
                if isinstance(formatted_user[i], str) and col in ['last_login', 'created_at']:
                    try:
                        formatted_user[i] = datetime.strptime(formatted_user[i], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
            formatted_users.append(formatted_user)

        # Create DataFrame with available columns
        user_df = pd.DataFrame(formatted_users, columns=select_columns)
        st.dataframe(user_df)

        with st.expander("Create New User"):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            user_role = st.selectbox("Role", ["user", "admin"])
            location_id = st.text_input("Location ID")

            if st.button("Add User"):
                if not new_user or not new_pass:
                    st.error("Username and password are required")
                elif len(new_pass) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
                    try:
                        # Use the correct columns based on what exists
                        if "location_id" in columns:
                            conn.execute(
                                "INSERT INTO users (username, password, role, location_id) VALUES (?, ?, ?, ?)",
                                (new_user, hashed, user_role, location_id)
                            )
                        else:
                            conn.execute(
                                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                                (new_user, hashed, user_role)
                            )
                        conn.commit()
                        st.success("User created successfully!")
                        time.sleep(1)
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Username already exists")

    with tab2:
        st.subheader("Training Content")

        # Document upload
        with st.expander("Upload Training Materials"):
            file_type = st.selectbox("Content Type", ["Document", "Video"])
            uploaded = st.file_uploader(
                f"Upload {file_type}",
                type=["pdf", "docx", "mp4"] if file_type == "Document" else ["mp4", "mov"]
            )
            description = st.text_area("Content Description")

            if uploaded and st.button("Upload"):
                save_dir = "training_content"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, uploaded.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # Store metadata
                meta_path = os.path.join(save_dir, f"{uploaded.name}.meta")
                with open(meta_path, "w") as f:
                    json.dump({
                        "uploaded_by": st.session_state.username,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "description": description,
                        "type": file_type.lower()
                    }, f)

                st.success(f"{file_type} uploaded successfully!")

        # Content library
        st.subheader("Content Library")
        if os.path.exists("training_content"):
            files = os.listdir("training_content")
            content_files = [f for f in files if not f.endswith(".meta")]

            for file in content_files:
                meta_file = f"{file}.meta"
                if meta_file in files:
                    with open(os.path.join("training_content", meta_file)) as f:
                        meta = json.load(f)
                    st.markdown(f"**{file}**")
                    st.caption(f"Type: {meta['type']} | Uploaded by: {meta['uploaded_by']}")
                    st.caption(f"Description: {meta['description']}")
                    st.download_button(
                        f"Download {file}",
                        data=open(os.path.join("training_content", file), "rb").read(),
                        file_name=file
                    )
                    st.divider()

    with tab3:
        st.subheader("Usage Analytics")

        # Feature usage
        st.write("### Feature Usage")
        usage = conn.execute(
            "SELECT feature, COUNT(*) as count FROM interactions GROUP BY feature"
        ).fetchall()
        if usage:
            fig = px.pie(
                names=[u[0] for u in usage],
                values=[u[1] for u in usage],
                title="Feature Usage Distribution"
            )
            st.plotly_chart(fig)
        else:
            st.warning("No usage data available yet")

        # User activity
        st.write("### User Activity")
        activity = conn.execute(
            "SELECT username, COUNT(*) as interactions "
            "FROM interactions GROUP BY username ORDER BY interactions DESC LIMIT 10"
        ).fetchall()
        if activity:
            fig = px.bar(
                x=[a[0] for a in activity],
                y=[a[1] for a in activity],
                labels={"x": "User", "y": "Interactions"},
                title="Top Users by Activity"
            )
            st.plotly_chart(fig)
        else:
            st.warning("No user activity data available")

# ─── History View ──────────────────────────────────────────────────────────────
def history_ui(conn):
    """Show user's interaction history"""
    st.header("🕒 Your History")

    if "username" not in st.session_state:
        st.warning("Please log in to view your history")
        return

    # If user has requested a full view, show it and bail out immediately
    if "current_interaction" in st.session_state:
        interaction = st.session_state.current_interaction
        st.subheader(f"Full Interaction – {interaction['timestamp']}")
        st.write(f"**Feature:** {interaction['feature']}")
        tabs = st.tabs(["Input", "Output"])
        with tabs[0]:
            st.text(interaction["input"])
        with tabs[1]:
            st.markdown(interaction["output"])

        if st.button("← Back to History"):
            del st.session_state.current_interaction
            st.rerun()
        return  # don’t render the list below

    # Otherwise: render the list of past interactions
    history = conn.execute(
        "SELECT timestamp, feature, input_text, output_text "
        "FROM interactions WHERE username = ? ORDER BY timestamp DESC",
        (st.session_state.username,)
    ).fetchall()

    if not history:
        st.info("No history found – your interactions will appear here")
        return

    for i, (ts, feature, inp, out) in enumerate(history):
        with st.expander(f"{ts} • {feature}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Input**")
                st.text(inp[:500] + ("…" if len(inp) > 500 else ""))
            with col2:
                st.write("**Output**")
                st.text(out[:500] + ("…" if len(out) > 500 else ""))

            # single button per interaction
            if st.button(f"View Full Interaction #{i+1}", key=f"view_full_{i}"):
                st.session_state.current_interaction = {
                    "timestamp": ts,
                    "feature": feature,
                    "input": inp,
                    "output": out
                }
                st.rerun()

# ─── Chatbot Helper (Conversational) ───────────────────────────────────────────
def chatbot_ui(conn):
    """Persistent conversational chatbot beneath features"""
    if not st.session_state.get("username"):
        st.warning("Please log in to use the chatbot.")
        return
    # Retrieve conversation for this feature
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = []
    st.header("🤖 AI Chatbot")
        # Clear chat button
    if st.button("Clear Chat", key="clear_chat_button"):
        st.session_state["chat_memory"] = []
    st.markdown("Chat with the assistant based on your recent output.")
    # Display past messages
    for role, message in st.session_state["chat_memory"]:
        st.chat_message(role).write(message)
    # New user message
    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state["chat_memory"].append(("user", user_input))
        # Build context from last 10 interactions
        rows = conn.execute(
            "SELECT feature, input_text, output_text FROM interactions WHERE username=? ORDER BY timestamp DESC LIMIT 10",
            (st.session_state.username,)
        ).fetchall()
        context = "\n\n".join([f"Feature: {r[0]}\nInput: {r[1]}\nOutput: {r[2]}" for r in rows])
        prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"
        # Call AI
        if st.session_state.get("chat_model_choice", "Gemini") == "Gemini":
            answer = call_gemini("chatbot", prompt)
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant using past interactions."},
                {"role": "user", "content": prompt}
            ]
            answer = call_mistral(messages)
        # Append and display bot response
        st.session_state["chat_memory"].append(("assistant", answer))
        st.chat_message("assistant").write(answer)
        # Save interaction
        save_interaction(conn, "chatbot", user_input, answer)

def main():
    """Main application function with comprehensive error handling and persistent outputs"""
    # Configure page
    st.set_page_config(
        page_title="Property Deals AI",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply brand styling
    st.markdown(
        f"""
        <style>
            .main {{
                background-color: {BRAND_COLORS['background']};
            }}
            .sidebar .sidebar-content {{
                background-color: {BRAND_COLORS['primary']} !important;
                color: white;
            }}
            h1, h2, h3 {{
                color: {BRAND_COLORS['primary']};
            }}
            .stButton>button {{
                background-color: {BRAND_COLORS['secondary']};
                color: white;
            }}
            /* Input styling: black background, white text */
            .stTextInput>div>div>input,
            .stTextArea>div>div>textarea {{
                background-color: black !important;
                color: white !important;
            }}
            /* Placeholder text styling: white */
            .stTextInput>div>div>input::placeholder,
            .stTextArea>div>div>textarea::placeholder {{
                color: white !important;
                opacity: 1 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize database and session
    try:
        conn = init_db()
        create_default_admin(conn)
    except sqlite3.Error as e:
        st.error(f"Failed to initialize database: {e}")
        return

    # Ensure login state
    if "logged_in" not in st.session_state:
        st.session_state.update({
            "logged_in": False,
            "username": None,
            "role": None
        })

    # Authentication flow
    if not st.session_state.logged_in:
        login_ui(conn)
        return

    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    st.sidebar.markdown(f"**Location ID:** {st.session_state.get('location_id', 'Not specified')}")
    features = ["Lease Summarization", "Deal Structuring", "Offer Generator", "History"]
    if st.session_state.role == "admin":
        features.insert(-1, "Admin Portal")
    selected = st.sidebar.radio("Navigation", features)

    # Main content
    try:
        if selected == "Lease Summarization":
            lease_summarization_ui(conn)
        elif selected == "Deal Structuring":
            deal_structuring_ui(conn)
        elif selected == "Offer Generator":
            offer_generator_ui(conn)
        elif selected == "History":
            history_ui(conn)
        elif selected == "Admin Portal" and st.session_state.role == "admin":
            admin_portal_ui(conn)
        else:
            st.error("Access Denied")
    except Exception as e:
        st.error(f"Error in {selected} feature: {e}")

    # Divider and chatbot helper
    st.divider()
    chatbot_ui(conn)

    # Logout
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
