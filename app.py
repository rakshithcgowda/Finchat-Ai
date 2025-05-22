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
from mistralai import Mistral
import requests
import pytesseract
from pdf2image import convert_from_bytes
import re
from openai import OpenAI
from PIL import Image
import pdfplumber
import torch
from torchvision import transforms

# ─── Configuration ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get(
    "GOOGLE_API_KEY",
    "AIzaSyANbVVzZACnYnus00xwwRRE01n34yoAmcU"  # fallback for dev/testing
)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "DUW9f3t6nvZaNkEbxcrxYP4hLIrC3g7Y")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-61f7f17d33bd4598b4dd61edd13af337")
DEEPSEEK_CLIENT = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"  # ensure /v1 is included
)

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
def call_gemini(
    feature: str,
    content: str,
    temperature: float = 0.7
) -> str:
    """Call Google's Gemini model with proper temperature handling."""
    system_prompts = {
        "lease_analysis": (
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

    try:
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"SYSTEM: {system_prompts.get(feature, '')}\n\nUSER: {content}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return f"Error: {e}"


def call_mistral(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop: List[str] = None,
    stream: bool = False,
    user: str = None,
    logit_bias: Dict[int, float] = None,
) -> str:
    """
    Call Mistral API with extended parameter support.
    """
    payload = {
        "model": "mistral-small-latest",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    if user:
        payload["user"] = user
    if logit_bias:
        payload["logit_bias"] = logit_bias

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers, stream=stream)
    resp.raise_for_status()
    data = resp.json()

    if stream:
        # for streaming, yield chunks or join
        return "".join(chunk.get("content", "") for chunk in data.get("choices", []))
    return data["choices"][0]["message"]["content"]


def call_deepseek(
    messages: List[Dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = False,
) -> str:
    """
    Call DeepSeek via the OpenAI-compatible SDK.
    """
    try:
        resp = DEEPSEEK_CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream
        )
        if stream:
            # stream is a generator of chunks
            return "".join(chunk.choices[0].delta.content for chunk in resp)
        return resp.choices[0].message.content
    except Exception as e:
        # more informative error
        return f"Error processing request with DeepSeek: {str(e)}"


def save_interaction(conn, feature: str, input_text: str, output_text: str):
    if st.session_state.get("username"):
        conn.execute(
            "INSERT INTO interactions (username, feature, input_text, output_text) VALUES (?, ?, ?, ?)",
            (st.session_state.username, feature, input_text, output_text),
        )
        conn.commit()


def lease_summarization_ui(conn):
    """Lease Summarization: upload PDF and get either full-document or page-by-page summaries with model selection, persisting results for chatbot usage."""
    import streamlit as st
    from PyPDF2 import PdfReader
    import json

    st.header("📄 Lease Summary")
    st.markdown(
        "Upload your lease PDF and receive a concise summary—choose to process the entire document at once or summarize each page individually."
    )

    # Initialize or clear previous summary when file changes
    uploaded_file = st.file_uploader(
        "Upload Lease Document (PDF)", type=["pdf"], key="lease_file_uploader"
    )
    if 'last_file' in st.session_state and uploaded_file:
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.pop('last_summary', None)
            st.session_state.pop('last_mode', None)
            st.session_state.pop('last_pages', None)
            st.session_state.pop('last_engine', None)
    
    if not uploaded_file:
        return

    # Model selection
    ai_engine = st.radio(
        "Select AI Model",
        ["DeepSeek", "Gemini Pro", "Mistral Large"],
        index=0,
        horizontal=True,
        key="lease_ai_engine"
    )
    # Summary mode selection
    summary_mode = st.radio(
        "Summary Mode",
        ["Full Document", "Page-by-Page"],
        index=1,
        horizontal=True,
        key="lease_summary_mode"
    )

    # Display existing summary if available
    if 'last_summary' in st.session_state and st.session_state.get('last_file') == uploaded_file.name:
        st.subheader(f"{st.session_state['last_mode']} Summary ({st.session_state['last_engine']})")
        if st.session_state['last_mode'] == 'Full Document':
            st.write(st.session_state['last_summary'])
        else:
            for idx, part in enumerate(st.session_state['last_summary'], start=1):
                st.markdown(f"**Page {idx} Summary:**")
                st.write(part)
        st.divider()

    # Generate new summary
    if st.button("Generate Summary", key="lease_generate_button"):
        try:
            reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            st.error("Failed to extract text from the PDF.")
            return

        if not any(pages):
            st.error("No readable text found in the PDF.")
            return

        # Prepare storage
        st.session_state['last_file'] = uploaded_file.name
        st.session_state['last_mode'] = summary_mode
        st.session_state['last_engine'] = ai_engine

        # Full document summary
        if summary_mode == "Full Document":
            raw_text = "\n".join(pages)
            chunks = [raw_text[i:i+15000] for i in range(0, len(raw_text), 15000)] if len(raw_text) > 15000 else [raw_text]
            parts = []
            with st.spinner("Summarizing full document..."):
                for chunk in chunks:
                    prompt = (
                        "Summarize this portion of the lease agreement in clear, concise language, "
                        "preserving all key details:\n\n" + chunk
                    )
                    if ai_engine == "DeepSeek":
                        part = call_deepseek(
                            messages=[{"role":"user","content":prompt}],
                            model="deepseek-chat",
                            temperature=0.3,
                            max_tokens=1024
                        )
                    elif ai_engine == "Gemini Pro":
                        part = call_gemini(
                            feature="lease_analysis",
                            content=prompt,
                            temperature=0.3
                        )
                    else:
                        part = call_mistral(
                            messages=[{"role":"user","content":prompt}],
                            temperature=0.3,
                            max_tokens=1024
                        )
                    parts.append(part)
                final_output = parts[0] if len(parts) == 1 else "\n\n".join(parts)
            st.subheader("Full Document Summary")
            st.write(final_output)
            save_interaction(conn, "lease_summary_full", uploaded_file.name, final_output)
            st.session_state['last_summary'] = final_output

        # Page-by-page summary
        else:
            summaries = []
            st.subheader("Page-by-Page Summaries")
            for idx, page_text in enumerate(pages, start=1):
                if not page_text.strip():
                    summaries.append("(no text detected)")
                    st.markdown(f"**Page {idx}:** _(no text detected)_")
                    continue
                with st.spinner(f"Summarizing page {idx}..."):
                    prompt = (
                        f"Summarize page {idx} of this lease agreement in clear, concise language, "
                        f"covering all information:\n\n{page_text}"
                    )
                    if ai_engine == "DeepSeek":
                        summary = call_deepseek(
                            messages=[{"role":"user","content":prompt}],
                            model="deepseek-chat",
                            temperature=0.3,
                            max_tokens=512
                        )
                    elif ai_engine == "Gemini Pro":
                        summary = call_gemini(
                            feature="lease_analysis",
                            content=prompt,
                            temperature=0.3
                        )
                    else:
                        summary = call_mistral(
                            messages=[{"role":"user","content":prompt}],
                            temperature=0.3,
                            max_tokens=512
                        )
                st.markdown(f"**Page {idx} Summary:**")
                st.write(summary)
                summaries.append(summary)
            save_interaction(
                conn,
                "lease_summary_pagewise",
                uploaded_file.name,
                json.dumps({f"page_{i}": pages[i-1] for i in range(1, len(pages)+1)})
            )
            st.session_state['last_summary'] = summaries

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

    ai_model = st.radio("AI Model", ["Gemini", "Mistral", "DeepSeek"], horizontal=True)

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
            elif ai_model == "Mistral":
                messages = [
                    {"role": "system", "content": "You are a real estate investment strategist. Provide creative deal structuring options."},
                    {"role": "user",   "content": prompt}
                ]
                strategies = call_mistral(messages=messages)
            else:  # DeepSeek
                messages = [
                    {"role": "system", "content": "You are an expert real estate strategist. Suggest creative deal structures with pros/cons."},
                    {"role": "user",   "content": prompt}
                ]
                strategies = call_deepseek(messages)

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
            elif ai_model == "Mistral":
                messages = [
                    {"role": "system", "content": "Refine the selected strategy based on user feedback."},
                    {"role": "user",   "content": refinement_prompt}
                ]
                refinement = call_mistral(messages=messages)
            else:  # DeepSeek
                messages = [
                    {"role": "system", "content": "Refine this real estate strategy based on the provided feedback."},
                    {"role": "user",   "content": refinement_prompt}
                ]
                refinement = call_deepseek(messages)

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
                "AI Model Preference", ["Gemini", "Mistral", "DeepSeek"], horizontal=True, key="offer_ai_model"
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
            elif d['ai_model'] == 'Mistral':
                messages = [
                    {'role': 'system', 'content': 'You are a real estate attorney.'},
                    {'role': 'user', 'content': prompt}
                ]
                offer = call_mistral(messages, temperature=d['creativity'])
            else:  # DeepSeek
                messages = [
                    {'role': 'system', 'content': 'You are a legal expert drafting a real estate purchase agreement.'},
                    {'role': 'user', 'content': prompt}
                ]
                offer = call_deepseek(messages, temperature=d['creativity'])

            st.session_state.generated_offer = offer
            save_interaction(conn, 'offer_generator', prompt, offer)

        st.subheader("Generated Offer")
        st.markdown(offer, unsafe_allow_html=True)
        if st.button("Proceed to Review"): st.session_state.offer_stage = 'review_edit'; st.rerun()
        if st.button("← Back"): st.session_state.offer_stage = 'details_entry'; st.rerun()


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
        return  # don't render the list below

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
        elif st.session_state.get("chat_model_choice") == "Mistral":
            messages = [
                {"role": "system", "content": "You are a helpful assistant using past interactions."},
                {"role": "user", "content": prompt}
            ]
            answer = call_mistral(messages)
        else:  # DeepSeek
            messages = [
                {"role": "system", "content": "You are an AI assistant answering questions based on the user's context."},
                {"role": "user", "content": prompt}
            ]
            answer = call_deepseek(messages)
        # Append and display bot response
        st.session_state["chat_memory"].append(("assistant", answer))
        st.chat_message("assistant").write(answer)
        # Save interaction
        save_interaction(conn, "chatbot", user_input, answer)





def ocr_pdf_to_searchable(input_pdf_bytes, ocr_model=None):
    """
    Convert a non-selectable PDF (scanned document) into a searchable PDF using OCR.

    Args:
        input_pdf_bytes: Bytes of the input PDF file
        ocr_model: Optional OCR model tuple (processor, model)

    Returns:
        Bytes of the searchable PDF
    """
    from fpdf import FPDF
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_bytes

    try:
        # Convert PDF pages to images
        images = convert_from_bytes(input_pdf_bytes)

        # Create a new PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for img in images:
            # Perform OCR on each image
            if ocr_model:
                text = predict_text_with_model(img, ocr_model)
            else:
                text = pytesseract.image_to_string(img)

            # Create a new page
            pdf.add_page()

            # Add the original image
            img_path = "temp_img.jpg"
            img.save(img_path)
            pdf.image(img_path, x=10, y=8, w=190)

            # Add invisible text layer
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(0, 0, 0, 0)  # Transparent text
            pdf.multi_cell(0, 5, text)

            # Clean up temp file
            os.remove(img_path)

        # Return the PDF bytes
        return pdf.output(dest='S').encode('latin-1')

    except Exception as e:
        st.error(f"OCR PDF conversion failed: {str(e)}")
        return None

# ─── OCR PDF Converter UI Function ─────────────────────────────────────────────
import io
import os
import time
from fpdf import FPDF
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import streamlit as st

# ─── OCR PDF Converter UI Function ─────────────────────────────────────────────
def ocr_pdf_ui(conn):
    """Convert non-selectable PDFs to searchable PDFs using OCR"""
    st.header("🔍 OCR PDF Converter")
    st.markdown("Convert scanned/non-selectable PDFs into searchable PDF documents with text layers.")

    # Settings
    with st.expander("⚙️ OCR Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            ocr_engine = st.radio("OCR Engine", ["Tesseract", "AI Model"], index=0)
            dpi = st.slider("Scan Resolution (DPI)", 150, 600, 300)
        with col2:
            preserve_layout = st.checkbox("Preserve Original Layout", True)
            language = st.selectbox(
                "Document Language",
                ["eng", "fra", "deu", "spa", "por", "chi_sim", "jpn", "kor"]
            )
            force_ocr = st.checkbox("Force OCR Processing", False)

    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
    if not uploaded_file:
        return

    # If using AI model, load it once
    ocr_model = None
    if ocr_engine == "AI Model":
        with st.spinner("Loading AI OCR model..."):
            try:
                ocr_model = load_ocr_model()
                if ocr_model is None:
                    raise ValueError("Model load returned None")
            except Exception as e:
                st.error(f"Failed to load AI OCR model: {e}")
                st.info("Falling back to Tesseract engine.")
                ocr_engine = "Tesseract"

    if st.button("Convert to Searchable PDF"):
        file_bytes = uploaded_file.read()
        if not file_bytes.startswith(b"%PDF"):
            st.error("ⓘ That doesn’t look like a valid PDF.")
            return

        status = st.empty()
        # Pre-OCR text check
        if not force_ocr:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                sample_text = "".join(
                    page.extract_text() or "" for page in reader.pages[:2]
                )
                if len(sample_text) > 1000 or (sample_text.count(" ")/len(sample_text) > 0.15):
                    st.warning(
                        "This PDF appears to have selectable text. Use ‘Force OCR’ to override."
                    )
                    with st.expander("Extracted Text Sample"):
                        st.code(sample_text[:1000] + "…")
                    return
            except Exception:
                pass  # Proceed to OCR if check fails

        texts = []
        image_paths = []
        success_pages = 0

        try:
            # Convert PDF to images
            with st.spinner("Converting PDF → images…"):
                images = convert_from_bytes(
                    file_bytes,
                    dpi=dpi,
                    fmt="jpeg",
                    thread_count=4,
                    strict=False
                )
                if not images:
                    raise RuntimeError("No pages found in PDF")

            # OCR each page and save temp images
            for idx, img in enumerate(images):
                status.text(f"🔠 OCR page {idx+1}/{len(images)}…")
                img_path = f"temp_page_{idx}.jpg"
                img.save(img_path, "JPEG", quality=80)
                image_paths.append(img_path)

                if ocr_engine == "Tesseract":
                    page_text = pytesseract.image_to_string(
                        Image.open(img_path),
                        lang=language,
                        config="--oem 3 --psm 6"
                    )
                else:
                    page_text = predict_text_with_model(img, ocr_model) or ""

                texts.append(page_text)
                success_pages += 1

            # Build searchable PDF
            pdf = FPDF()
            pdf.set_auto_page_break(True, 15)
            pdf.set_creator("PropertyDealsAI OCR Converter")

            for img_path, page_text in zip(image_paths, texts):
                # Sanitize text to Latin-1 by ignoring unsupported characters
                safe_text = page_text.encode('latin-1', 'ignore').decode('latin-1')

                pdf.add_page()
                if preserve_layout:
                    pdf.image(img_path, x=10, y=8, w=190)
                pdf.set_font("Arial", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, safe_text)

            # Encode final PDF, ignoring any remaining non–Latin-1 chars
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')

        except Exception as e:
            st.error(f"Conversion failed: {e}")
            return

        finally:
            status.empty()
            # Clean up temp images
            for path in image_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Results & Download
        st.success(f"✅ Converted {success_pages}/{len(texts)} pages.")
        with st.expander("📝 OCR Results Preview", expanded=True):
            tab1, tab2 = st.tabs(["Extracted Text", "First-Page Preview"])
            with tab1:
                preview = "\n\n------\n\n".join(texts)
                st.text(preview[:2000] + ("…" if len(preview) > 2000 else ""))
            with tab2:
                st.image(Image.open(image_paths[0]), use_column_width=True)

        st.download_button(
            "💾 Download Searchable PDF",
            data=pdf_bytes,
            file_name=f"searchable_{uploaded_file.name}",
            mime="application/pdf"
        )

        save_interaction(
            conn,
            "ocr_pdf",
            f"{uploaded_file.name} → searchable PDF",
            f"Engine={ocr_engine}, DPI={dpi}, Lang={language}"
        )
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
        elif selected == "OCR PDF":
            ocr_pdf_ui(conn)
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