import io
import json
import os
import sqlite3
import time
import bcrypt
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
import re
# try:
#     from docx import Document
# except ImportError:
#     Document = None  # python-docx not installed or conflicting package



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
    """Plain‐style login UI with minimal styling, welcome banner, and open registration."""
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
                cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()

                if row and bcrypt.checkpw(password.encode(), row[0]):
                    # set session
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = row[1]
                    st.session_state.location_id = location_id or None
                    # update last_login if column exists
                    cursor.execute("PRAGMA table_info(users)")
                    cols = [c[1] for c in cursor.fetchall()]
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

    # ─── REGISTER TAB ────────────────────────────────────────────
    with register_tab:
        new_user = st.text_input("New Username", key="reg_username")
        new_pass = st.text_input("New Password", type="password", key="reg_password")
        confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
        # Default all new registrations to 'user'
        user_role = "user"

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

    # ─── MAIN PANE WELCOME BANNER ───────────────────────────────
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
    """
    Try PyPDF2 → pdfplumber → OCR via pdf2image.
    If Poppler is missing, we skip OCR gracefully.
    """
    import io
    from PyPDF2 import PdfReader

    raw = uploaded_file.read()
    text = ""

    # 1) PyPDF2
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = ""

    # 2) pdfplumber if still too short
    if len(text.strip()) < ocr_threshold:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            alt = "\n".join(pages)
            if len(alt.strip()) > len(text.strip()):
                text = alt
        except Exception:
            # pdfplumber failed or PDF malformed
            pass

    # 3) OCR fallback if still below threshold and pdf2image/Poppler available
    if len(text.strip()) < ocr_threshold:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(raw)  # may raise if poppler missing
            ocr_pages = [pytesseract.image_to_string(img) for img in images]
            text = "\n".join(ocr_pages)
        except Exception:
            # Could be Poppler missing or pytesseract error; skip OCR
            st.warning("OCR step skipped (Poppler or Tesseract not available).")
    return text


def lease_summarization_ui(conn):
    """Enhanced lease summarization UI with styled PDF export and Poppler-safe OCR."""
    st.header("📄 Advanced Lease Summarization")
    st.markdown("Upload lease documents (PDF/DOCX) for comprehensive AI-powered analysis.")

    # Initialize session state
    if "lease_results" not in st.session_state:
        st.session_state["lease_results"] = []
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

    if st.button("Analyze Documents", key="lease_analyze"):
        if not uploaded_files:
            st.error("Please upload at least one document.")
            return

        results = []
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            # Extract text safely
            if uploaded_file.type == "application/pdf":
                text = extract_pdf_text(uploaded_file)
            else:
                import docx
                text = "\n".join(p.text for p in docx.Document(uploaded_file).paragraphs)

            # Build AI analysis
            doc_out = {"name": uploaded_file.name, "sections": []}
            steps = [
                ("Document Structure", "Summarize the overall structure."),
                ("Key Clauses", "Extract and categorize major clauses."),
                ("Obligations", "Map parties' obligations."),
                ("Financial Terms", "Identify payments and schedules."),
                ("Termination", "Summarize termination/renewal conditions.")
            ]
            for title, instr in steps:
                if ai_model == "Gemini":
                    prompt = (
                        f"{instr}\nContext: {legal_jurisdiction}\nLevel: {analysis_depth}\n\n"
                        f"{text[:15000]}"
                    )
                    content = call_gemini("lease_summary", prompt)
                else:
                    msgs = [
                        {"role": "system", "content": f"You are a legal expert. {instr}"},
                        {"role": "user", "content": f"{analysis_depth} detail:\n{text[:15000]}"}
                    ]
                    content = call_mistral(msgs)
                doc_out["sections"].append({"title": title, "content": content})

            # Optional risk assessment
            if highlight_risks:
                if ai_model == "Gemini":
                    content = call_gemini("lease_summary", f"Assess risks:\n{text[:15000]}")
                else:
                    msgs = [
                        {"role": "system", "content": "You are a risk assessor."},
                        {"role": "user", "content": text[:15000]}
                    ]
                    content = call_mistral(msgs)
                doc_out["sections"].append({"title": "Risk Assessment", "content": content})

            results.append(doc_out)
            save_interaction(conn, "lease_summary", text, str(doc_out))

        # Cross‐doc comparison
        comp = None
        if compare_clauses and len(results) > 1:
            comp_text = "".join(f"{d['name']}:\n{d['sections'][0]['content']}\n\n" for d in results)
            if ai_model == "Gemini":
                comp = call_gemini("lease_summary", f"Compare clauses:\n{comp_text}")
            else:
                msgs = [
                    {"role": "system", "content": "You are a comparison expert."},
                    {"role": "user", "content": comp_text}
                ]
                comp = call_mistral(msgs)
            save_interaction(conn, "lease_comparison", "multiple", comp)

        st.session_state["lease_results"] = results
        st.session_state["lease_comparison"] = comp

    # Display results
    if st.session_state.get("lease_results"):
        st.success("Analysis Complete!")
        for doc in st.session_state["lease_results"]:
            with st.expander(doc["name"]):
                for sec in doc["sections"]:
                    st.subheader(sec["title"])
                    st.markdown(sec["content"])
        if st.session_state.get("lease_comparison"):
            st.subheader("Comparison")
            st.markdown(st.session_state["lease_comparison"])

        # Export controls
        fmt = st.selectbox("Export As", ["PDF", "Word", "JSON", "HTML"], key="lease_export_fmt")
        name = st.text_input("File Name", "lease_analysis", key="lease_export_name")
        if st.button("Export Analysis", key="lease_export"):
            data = {
                "documents": st.session_state["lease_results"],
                "comparison": st.session_state["lease_comparison"]
            }


            if fmt == "PDF":
                pdf = FPDF("P", "mm", "A4")
                pdf.set_auto_page_break(True, 15)

                primary = (46, 134, 171)
                accent  = (241, 143,   1)
                text_col= (18,  17,   17)

                for idx, d in enumerate(st.session_state["lease_results"]):
                    pdf.add_page()

                    # Title
                    pdf.set_text_color(*primary)
                    pdf.set_font("Arial", "B", 20)
                    pdf.cell(0, 10, d["name"], ln=True, align="C")
                    y = pdf.get_y() + 2
                    pdf.set_draw_color(*accent)
                    pdf.set_line_width(0.5)
                    pdf.line(15, y, 195, y)
                    pdf.ln(5)

                    # Sections
                    for sec in d["sections"]:
                        # Clean out any asterisks
                        title = re.sub(r"\*", "", sec["title"])
                        pdf.set_text_color(*primary)
                        pdf.set_font("Arial", "B", 14)
                        pdf.multi_cell(0, 8, title)
                        pdf.ln(1)

                        pdf.set_text_color(*text_col)
                        pdf.set_font("Arial", "", 11)
                        for line in sec["content"].split("\n"):
                            # remove stray asterisks and then latin-1–safe encode
                            clean = re.sub(r"\*", "", line)
                            safe  = clean.encode("latin-1", "replace").decode("latin-1")
                            pdf.multi_cell(0, 6, safe)
                        pdf.ln(4)

                    # Footer
                    pdf.set_y(-15)
                    pdf.set_draw_color(*primary)
                    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                    pdf.set_font("Arial", "I", 8)
                    pdf.set_text_color(*text_col)
                    pdf.cell(0, 10, f"Page {idx+1}", align="R")

                pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
                st.download_button("Download PDF", pdf_bytes, f"{name}.pdf", "application/pdf")


            elif fmt == "Word":
                import io, docx
                docx_out = docx.Document()
                for d in st.session_state["lease_results"]:
                    docx_out.add_heading(d["name"], 1)
                    for sec in d["sections"]:
                        docx_out.add_heading(sec["title"], 2)
                        docx_out.add_paragraph(sec["content"])
                buf = io.BytesIO()
                docx_out.save(buf)
                st.download_button("Download Word", buf.getvalue(), f"{name}.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            elif fmt == "JSON":
                st.download_button("Download JSON", json.dumps(data, indent=2).encode(),
                                   f"{name}.json", "application/json")

            else:  # HTML
                html = "<html><body>"
                for d in st.session_state["lease_results"]:
                    html += f"<h1>{d['name']}</h1>"
                    for sec in d["sections"]:
                        html += f"<h2>{sec['title']}</h2><p>{sec['content']}</p>"
                html += "</body></html>"
                st.download_button("Download HTML", html.encode(), f"{name}.html", "text/html")

def deal_structuring_ui(conn):
    """Enhanced deal structuring with persistent strategy chat until cleared."""
    st.header("💡 Creative Deal Structuring Bot")
    st.markdown("Get AI-powered strategies for your property deals")

    # Initialize session state
    if "deal_strategy_memory" not in st.session_state:
        st.session_state.deal_strategy_memory = []
        st.session_state.last_strategies = None
        st.session_state.strategy_confidences = {}  # Track confidences per strategy

    # Clear chat
    if st.button("Clear Strategies", key="clear_strategies"):
        st.session_state.deal_strategy_memory.clear()
        st.session_state.last_strategies = None
        st.session_state.strategy_confidences = {}
        st.rerun()

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
        with st.spinner("Developing strategies..."):
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

        # Initialize confidences for each strategy
        matches = re.findall(
            r"Strategy\s+(\d+):\s*(.*?)(?=(?:Strategy\s+\d+:)|\Z)",
            strategies,
            flags=re.S
        )
        if matches:
            for num, _ in matches:
                strategy_key = f"Strategy {num}"
                if strategy_key not in st.session_state.strategy_confidences:
                    st.session_state.strategy_confidences[strategy_key] = 7  # Default confidence
        else:
            # Fallback if no numbered sections found
            if "Strategy 1" not in st.session_state.strategy_confidences:
                st.session_state.strategy_confidences["Strategy 1"] = 7

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

        # Confidence slider - gets/sets value from session state
        confidence = st.slider(
            "Confidence in this strategy", 
            1, 10, 
            value=st.session_state.strategy_confidences.get(selected_label, 7),
            key=f"conf_{selected_label.replace(' ', '_')}"
        )
        
        # Update confidence in session state
        st.session_state.strategy_confidences[selected_label] = confidence

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


# -------------------------------------------------offer generator----------------------------------------------------------------------------------

def build_guided_prompt(details: dict, detail_level: str) -> str:
    """
    Construct a detailed prompt from guided form data to generate a real estate purchase agreement.
    """
    buyer = details['parties']['buyer']
    buyer_rep = details['parties'].get('buyer_rep', '')
    seller = details['parties']['seller']
    seller_rep = details['parties'].get('seller_rep', '')
    buyer_line = f"- Buyer: {buyer}{f' (Represented by: {buyer_rep})' if buyer_rep else ''}"
    seller_line = f"- Seller: {seller}{f' (Represented by: {seller_rep})' if seller_rep else ''}"

    address = details['property']['address']
    county = details['property'].get('county', '')
    address_line = f"- Property Address: {address}"
    county_line = f"- County: {county}" if county else ''

    price = details['financial']['price_fmt']
    earnest = details['financial']['earnest_fmt']
    price_line = f"- Purchase Price: {price}"
    earnest_line = f"- Earnest Money Deposit: {earnest}"

    closing = details['dates']['closing']
    expiry = details['dates']['expiry']
    closing_line = f"- Proposed Closing Date: {closing}"
    expiry_line = f"- Offer Expiration: {expiry} hours from signing"

    financing = details['terms'].get('financing', '')
    contingencies = details['terms'].get('contingencies', [])
    contingencies_str = ', '.join(contingencies) if contingencies else 'None'
    special_terms = details['terms'].get('special', '')
    financing_line = f"- Financing Type: {financing}"
    contingencies_line = f"- Contingencies: {contingencies_str}"
    special_line = f"- Special Terms: {special_terms}" if special_terms else ''

    jurisdiction = details['terms'].get('jurisdiction', '')
    jurisdiction_line = f"- Governing Law: {jurisdiction}" if jurisdiction else ''

    sections = [
        "Generate a professional real estate purchase agreement with the following details:",
        buyer_line, seller_line, address_line, county_line,
        price_line, earnest_line, closing_line, expiry_line,
        financing_line, contingencies_line, special_line, jurisdiction_line,
        f"Level of Detail: {detail_level}."
    ]
    return "\n".join([s for s in sections if s])


def offer_generator_ui(conn):
    st.header("✍️ Advanced Offer Generator")
    st.markdown(
        """
        <style>
        .offer-section { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 20px; }
        .offer-highlight { background-color: #fffacd; padding: 2px 5px; border-radius: 3px; }
        </style>
        """, unsafe_allow_html=True
    )

    if 'offer_stage' not in st.session_state:
        st.session_state.update({
            'offer_stage': 'input_method',
            'offer_data': {},
            'generated_offer': None,
            'edited_offer': None,
            'review_comments': []
        })

    stages = ["input_method", "details_entry", "offer_generation", "review_edit", "export"]
    labels = ["Input Method", "Details Entry", "Offer Generation", "Review & Edit", "Export"]
    idx = stages.index(st.session_state.offer_stage)
    cols = st.columns(len(stages))
    for i, label in enumerate(labels):
        with cols[i]:
            if i < idx:
                st.success(f"✓ {label}")
            elif i == idx:
                st.info(f"→ {label}")
            else:
                st.caption(label)

    # Stage 1: Input Method
    if st.session_state.offer_stage == 'input_method':
        st.markdown("### 1. Select Input Method")
        method = st.radio(
            "How would you like to create your offer?",
            ["Guided Form", "Free Text", "Upload Existing", "Template Library"],
            horizontal=True,
            key="offer_input_method"
        )
        st.session_state.offer_data['input_method'] = method
        with st.expander("AI Configuration"):
            ai_model = st.radio(
                "AI Model Preference", ["Gemini", "Mistral"], horizontal=True, key="offer_ai_model"
            )
            creativity = st.slider("Creativity Level", 0.0, 1.0, 0.3, key="offer_creativity")
            detail_level = st.select_slider(
                "Detail Level", options=["Minimal","Standard","Comprehensive"],
                value="Standard", key="offer_detail_level"
            )
            st.session_state.offer_data.update({
                'ai_model': ai_model,
                'creativity': creativity,
                'detail_level': detail_level
            })
        if st.button("Continue to Details", key="btn_continue_details"): 
            st.session_state.offer_stage = 'details_entry'
            st.rerun()

    # Stage 2: Details Entry
    elif st.session_state.offer_stage == 'details_entry':
        st.markdown("### 2. Enter Offer Details")
        method = st.session_state.offer_data['input_method']
        if method == 'Guided Form':
            with st.form("offer_details_form"):
                st.markdown('<div class="offer-section">', unsafe_allow_html=True)
                st.markdown('#### Basic Information')
                c1, c2, c3 = st.columns(3)
                with c1:
                    buyer = st.text_input("Buyer Full Name*", key="offer_buyer")
                    buyer_rep = st.text_input("Buyer\'s Representative", key="offer_buyer_rep")
                with c2:
                    seller = st.text_input("Seller Full Name*", key="offer_seller")
                    seller_rep = st.text_input("Seller\'s Representative", key="offer_seller_rep")
                with c3:
                    address = st.text_input("Property Address*", key="offer_address")
                    county = st.text_input("County", key="offer_county")
                st.markdown('#### Financial Terms')
                c1, c2, c3 = st.columns(3)
                with c1:
                    price = st.number_input("Purchase Price*", min_value=1000, step=1000, key="offer_price")
                with c2:
                    earnest = st.number_input("Earnest Money Deposit*", min_value=0, step=1000, key="offer_earnest")
                with c3:
                    closing = st.date_input("Proposed Closing Date*", min_value=datetime.now().date(), key="offer_closing")
                st.markdown('#### Terms & Conditions')
                c1, c2 = st.columns(2)
                with c1:
                    financing = st.selectbox("Financing Type*", ["Cash","Conventional Loan","FHA","VA","Seller Financing","Other"], key="offer_financing")
                    if financing == "Other": st.text_input("Specify Financing Type", key="offer_financing_other")
                with c2:
                    cont = st.multiselect("Contingencies", ["Inspection","Appraisal","Financing","Title Review","HOA Approval","Other"], key="offer_contingencies")
                    if "Other" in cont: st.text_input("Specify Other Contingency", key="offer_contingencies_other")
                st.markdown('#### Additional Provisions')
                terms = st.text_area("Special Terms/Conditions", key="offer_special_terms")
                st.markdown('#### Jurisdiction & Expiry')
                c1, c2 = st.columns(2)
                with c1:
                    jurisdiction = st.selectbox("Governing Law", ["State Default","California","Texas","New York","Florida","Other"], key="offer_jurisdiction")
                with c2:
                    expiry = st.number_input("Offer Expiration (hours)", min_value=1, max_value=168, value=48, key="offer_expiry")
                st.markdown('</div>', unsafe_allow_html=True)

                submitted = st.form_submit_button("Generate Offer Draft")
                if submitted:
                    missing = []
                    for field, msg in {
                        "offer_buyer": "Buyer required",
                        "offer_seller": "Seller required",
                        "offer_address": "Address required",
                        "offer_price": "Price >=1000",
                        "offer_earnest": "Earnest required",
                        "offer_closing": "Closing date required"
                    }.items():
                        if not st.session_state.get(field):
                            missing.append(msg)
                    if missing:
                        for m in missing:
                            st.error(m)
                    else:
                        st.session_state.offer_data['details'] = {
                            'parties': {
                                'buyer': buyer,
                                'buyer_rep': buyer_rep,
                                'seller': seller,
                                'seller_rep': seller_rep
                            },
                            'property': {
                                'address': address,
                                'county': county
                            },
                            'financial': {
                                'price': price,
                                'earnest': earnest,
                                'price_fmt': f"${price:,}",
                                'earnest_fmt': f"${earnest:,}"
                            },
                            'dates': {
                                'closing': closing.strftime("%B %d, %Y"),
                                'expiry': expiry
                            },
                            'terms': {
                                'financing': financing,
                                'contingencies': cont,
                                'special': terms,
                                'jurisdiction': jurisdiction
                            }
                        }
                        st.session_state.offer_stage = 'offer_generation'
                        st.rerun()

        elif method == 'Free Text':
            st.markdown("Enter deal details (min 50 chars):")
            text = st.text_area("Deal Details", key="offer_free_text", height=200)
            if st.button("Generate Offer Draft Free Text", key="btn_ft_draft"):
                if len(text) < 50:
                    st.error("Please add more detail.")
                else:
                    st.session_state.offer_data['details'] = {'free_text': text}
                    st.session_state.offer_stage = 'offer_generation'
                    st.rerun()

        elif method == 'Upload Existing':
            uploaded = st.file_uploader("Upload Document", type=["pdf","docx","txt"], key="offer_upload")
            if uploaded and st.button("Analyze & Improve Upload", key="btn_upload_analyze"):
                if uploaded.type == "application/pdf":
                    reader = PdfReader(uploaded)
                    doc_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                elif uploaded.type == "text/plain":
                    doc_text = uploaded.read().decode("utf-8")
                else:
                    # doc = Document(uploaded)
                    doc_text = "\n".join(p.text for p in doc.paragraphs)
                st.session_state.offer_data['details'] = {'uploaded': doc_text}
                st.session_state.offer_stage = 'offer_generation'
                st.rerun()

        else:  # Template Library
            st.markdown("### Template Library")
            templates = {
                "Residential": "templates/standard_residential.json",
                "Commercial": "templates/commercial_lease_purchase.json",
                "Seller Financing": "templates/seller_financing.json",
                "1031 Exchange": "templates/1031_exchange.json"
            }
            choice = st.selectbox("Select Template", list(templates.keys()), key="offer_template")
            with st.expander("Preview"):
                try:
                    st.json(json.load(open(templates[choice])))
                except Exception:
                    st.warning("Preview unavailable")
            if st.button("Use Template", key="btn_use_template"):
                try:
                    data = json.load(open(templates[choice]))
                    st.session_state.offer_data['details'] = data
                    st.session_state.offer_stage = 'offer_generation'
                    st.rerun()
                except Exception:
                    st.error("Failed to load")

        if st.button("← Back", key="btn_back_to_input"):
            st.session_state.offer_stage = 'input_method'
            st.rerun()


    # Stage 3: Offer Generation
    if st.session_state.offer_stage == 'offer_generation':
        d = st.session_state.offer_data
        if d['input_method'] == 'Guided Form':
            prompt = build_guided_prompt(d['details'], d['detail_level'])
        elif d['input_method'] == 'Free Text':
            prompt = f"Draft a purchase agreement:\n\n{d['details']['free_text']}"
        elif d['input_method'] == 'Upload Existing':
            prompt = f"Improve this draft:\n\n{d['details']['uploaded']}"
        else:
            prompt = f"Generate from template:\n\n{json.dumps(d['details'], indent=2)}"
        prompt += f"\n\nDetail Level: {d['detail_level']}"

        with st.spinner("Generating..."):
            if d['ai_model'] == 'Gemini':
                offer = call_gemini('offer_generator', prompt)
            else:
                offer = call_mistral([
                    {'role': 'system', 'content': 'You are a real estate attorney.'},
                    {'role': 'user', 'content': prompt}
                ], temperature=d['creativity'])
            st.session_state.generated_offer = offer
            save_interaction(conn, 'offer_generator', prompt, offer)

        st.subheader("Generated Offer")
        st.markdown(offer, unsafe_allow_html=True)
        if st.button("Proceed to Review"): st.session_state.offer_stage = 'review_edit'; st.rerun()
        if st.button("← Back"): st.session_state.offer_stage = 'details_entry'; st.rerun()


    # Stage 3: Offer Generation
    if st.session_state.offer_stage == 'offer_generation':
        d = st.session_state.offer_data
        if d['input_method'] == 'Guided Form':
            prompt = build_guided_prompt(d['details'], d['detail_level'])
        elif d['input_method'] == 'Free Text':
            prompt = f"Draft a purchase agreement:\n\n{d['details']['free_text']}"
        elif d['input_method'] == 'Upload Existing':
            prompt = f"Improve this draft:\n\n{d['details']['uploaded']}"
        else:
            prompt = f"Generate from template:\n\n{json.dumps(d['details'], indent=2)}"
        prompt += f"\n\nDetail Level: {d['detail_level']}"

        with st.spinner("Generating..."):
            if d['ai_model'] == 'Gemini':
                offer = call_gemini('offer_generator', prompt)
            else:
                offer = call_mistral([
                    {'role': 'system', 'content': 'You are a real estate attorney.'},
                    {'role': 'user', 'content': prompt}
                ], temperature=d['creativity'])
            st.session_state.generated_offer = offer
            save_interaction(conn, 'offer_generator', prompt, offer)

        st.subheader("Generated Offer")
        st.markdown(offer, unsafe_allow_html=True)
        if st.button("Proceed to Review"):
            st.session_state.offer_stage = 'review_edit'
            st.rerun()
        if st.button("← Back"):
            st.session_state.offer_stage = 'details_entry'
            st.rerun()

    # Stage 4: Review & Edit
    if st.session_state.offer_stage == 'review_edit':
        edited = st.text_area(
            "Edit draft", value=st.session_state.generated_offer,
            height=300, key='offer_edit'
        )
        if edited != st.session_state.edited_offer:
            st.session_state.edited_offer = edited

        st.markdown("#### Comments")
        new_c = st.text_input("Add comment", key='offer_new_comment')
        if st.button("Add Comment") and new_c:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.review_comments.append({
                'ts': ts, 'text': new_c, 'resolved': False
            })
            st.rerun()

        for i, c in enumerate(st.session_state.review_comments):
            cols = st.columns([1, 8, 1])
            with cols[0]:
                st.markdown(f"**{c['ts']}**")
            with cols[1]:
                st.markdown(f"{'✓' if c['resolved'] else '◯'} {c['text']}")
            with cols[2]:
                if not c['resolved'] and st.button('Resolve', key=f'res_{i}'):
                    c['resolved'] = True
                    st.rerun()

        if st.button('← Back'):
            st.session_state.offer_stage = 'offer_generation'
            st.rerun()
        if st.button('Proceed to Export'):
            st.session_state.offer_stage = 'export'
            st.rerun()

    # Stage 5: Export
    if st.session_state.offer_stage == 'export':
        content = st.session_state.edited_offer or st.session_state.generated_offer
        if st.checkbox('Include Comments', value=True):
            content += (
                "\n\n---\n## Comments\n" +
                "\n".join([f"- [{c['ts']}] {c['text']}" for c in st.session_state.review_comments])
            )

        fmt = st.selectbox('Format', ['PDF', 'Word', 'Text', 'HTML'], key='offer_export_format')
        name = st.text_input('File Name', 'property_offer', key='offer_export_name')

        if st.button('Download'):
            if fmt == 'PDF':
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', size=12)
                for line in content.split('\n'):
                    pdf.multi_cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'))
                st.download_button(
                    'Download PDF', pdf.output(dest='S').encode('latin-1'),
                    f"{name}.pdf", "application/pdf"
                )
            elif fmt == 'Word':
                doc = docx.Document()
                for line in content.split('\n'):
                    doc.add_paragraph(line)
                buf = io.BytesIO()
                doc.save(buf)
                st.download_button(
                    'Download Word', buf.getvalue(),
                    f"{name}.docx",
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
            elif fmt == 'HTML':
                html = f"<pre>{content}</pre>"
                st.download_button(
                    'Download HTML', html.encode(),
                    f"{name}.html", 'text/html'
                )
            else:
                st.download_button(
                    'Download Text', content.encode(),
                    f"{name}.txt", 'text/plain'
                )

        if st.button('Start New'):
            for k in list(st.session_state.keys()):
                if k.startswith('offer_'):
                    del st.session_state[k]
            st.rerun()


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
