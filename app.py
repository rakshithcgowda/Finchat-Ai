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
# ─── Summarization Utility ─────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into manageable chunks for AI summarization."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ─── Helper: Load OCR Model ─────────────────────────────────────────────────

def load_ocr_model(model_path: str):
    """Attempt to load a custom OCR model; return None on failure"""
    if not os.path.exists(model_path):
        st.warning("OCR model not found; using Tesseract fallback.")
        return None
    try:
        model = torch.load(model_path, map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.eval()
            return model
    except:
        pass
    try:
        from models import OCRModel
        state = torch.load(model_path, map_location="cpu")
        m = OCRModel()
        m.load_state_dict(state)
        m.eval()
        return m
    except Exception as e:
        st.error(f"Failed to load OCR model: {e}")
    return None


# ─── Helper: Load OCR Model ─────────────────────────────────────────────────

def load_ocr_model(model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
    """Load a Hugging Face OCR model; return None on failure"""
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        return (processor, model)
    except Exception as e:
        st.error(f"Failed to load Hugging Face OCR model: {e}")

    # Fallback to TrOCR if Donut fails
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        return (processor, model)
    except Exception as e:
        st.error(f"Failed to load TrOCR model: {e}")

    return None


# ─── Helper: Predict Text With Model ─────────────────────────────────────────

def predict_text_with_model(img: Image.Image, model_tuple) -> str:
    """Run Hugging Face OCR model on an image; return empty string on failure"""
    if model_tuple is None:
        return ""

    processor, model = model_tuple

    try:
        # Convert image to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Process image based on model type
        if isinstance(processor, DonutProcessor):
            # Donut model processing
            pixel_values = processor(img, return_tensors="pt").pixel_values
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            return sequence

        else:
            # TrOCR model processing
            pixel_values = processor(img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    except Exception as e:
        st.warning(f"Hugging Face OCR prediction failed: {e}")
        return ""


# ─── Helper: Extract PDF/Text/Image ────────────────────────────────────────────

def extract_text_from_file(f, ocr_model=None, ocr_threshold: int = 100) -> str:
    """
    Extract text from PDF, image, or DOCX using Hugging Face models.
    """
    raw = f.read()
    text = ""

    # PDF processing
    if f.type == "application/pdf":
        try:
            # First try traditional text extraction
            reader = PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or '' for p in reader.pages]
            text = '\n'.join(pages)

            # If text extraction is insufficient, try OCR
            if len(text.strip()) < ocr_threshold and ocr_model:
                images = convert_from_bytes(raw)
                ocrs = []
                for img in images:
                    ocrs.append(predict_text_with_model(img, ocr_model))
                text = '\n'.join(ocrs)

        except Exception as e:
            st.warning(f"PDF processing error: {e}")
            text = ''

    # Image processing
    elif f.type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(raw))
            if ocr_model:
                text = predict_text_with_model(img, ocr_model)
            else:
                text = pytesseract.image_to_string(img)
        except Exception as e:
            st.warning(f"Image processing error: {e}")
            text = ''

    # DOCX processing (unchanged)
    elif f.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            text = '\n'.join(p.text for p in doc.paragraphs)
        except Exception:
            text = ''

    # Plain text fallback
    else:
        try:
            text = raw.decode('utf-8')
        except:
            text = ''

    return ' '.join(text.split())

def lease_summarization_ui(conn):
    """Advanced Lease Summarization with Multi-Stage Analysis"""
    st.header("📄 Advanced Lease Analysis Suite")
    st.markdown("""
    <style>
        .lease-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #2E86AB;
        }
        .risk-high { background-color: #ffdddd; }
        .risk-medium { background-color: #fff3cd; }
        .risk-low { background-color: #d4edda; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for multi-document analysis
    if "lease_docs" not in st.session_state:
        st.session_state.lease_docs = {}
        st.session_state.comparison_matrix = None
        st.session_state.analysis_stages = []

    # Configuration in expandable sections
    with st.expander("⚙️ Analysis Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            analysis_mode = st.selectbox(
                "Analysis Mode",
                ["Standard Review", "Due Diligence", "Risk Assessment", "Full Legal Audit"],
                index=0,
                help="Depth of analysis to perform"
            )
            jurisdiction = st.selectbox(
                "Governing Law",
                ["US (Default)", "UK", "EU", "Australia", "Canada", "Singapore", "UAE"],
                index=0
            )

        with col2:
            risk_profile = st.select_slider(
                "Risk Sensitivity",
                options=["Lenient", "Moderate", "Strict", "Aggressive"],
                value="Moderate"
            )
            compare_mode = st.checkbox(
                "Enable Cross-Document Comparison",
                value=True,
                help="Identify inconsistencies across multiple leases"
            )

        with col3:
            ai_engine = st.radio(
                "AI Engine",
                ["Gemini Pro", "Mistral Large", "DeepSeek", "Ensemble"],
                horizontal=True,
                index=0
            )
            ocr_fallback = st.checkbox(
                "Advanced OCR Fallback",
                value=True,
                help="Use AI-powered OCR when text extraction fails"
            )

    # Advanced settings
    with st.expander("🔧 Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Processing Chunk Size (chars)",
                min_value=500,
                max_value=4000,
                value=2000,
                step=500
            )
            temperature = st.slider(
                "AI Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        with col2:
            max_parallel = st.slider(
                "Parallel Processes",
                min_value=1,
                max_value=10,
                value=3,
                help="Higher values speed up processing but increase resource usage"
            )
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=50,
                max_value=95,
                value=75,
                help="Minimum confidence score for automatic clause classification"
            )

    # Document upload section with metadata
    st.markdown("---")
    st.subheader("📂 Document Upload & Metadata")

    uploaded_files = st.file_uploader(
        "Upload Lease Documents",
        type=["pdf", "docx", "jpg", "png", "tiff"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, and common image formats"
    )

    # Document metadata collection
    if uploaded_files:
        with st.expander("📝 Document Metadata", expanded=True):
            cols = st.columns(4)
            lease_types = ["Commercial", "Residential", "Industrial", "Retail", "Mixed-Use"]
            doc_metadata = {}

            for i, file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f"**{file.name}**")
                    doc_cols = st.columns([2,1,1])
                    with doc_cols[0]:
                        doc_metadata[file.name] = {
                            "type": st.selectbox(
                                "Lease Type",
                                lease_types,
                                index=0,
                                key=f"type_{i}"
                            )
                        }
                    with doc_cols[1]:
                        d = st.date_input(
                            "Effective Date",
                            key=f"date_{i}"
                        )
                        doc_metadata[file.name]["effective_date"] = d.strftime("%Y-%m-%d")
                    with doc_cols[2]:
                        doc_metadata[file.name]["parties"] = st.text_input(
                            "Parties (Optional)",
                            placeholder="Landlord/Tenant",
                            key=f"parties_{i}"
                        )

    # Analysis pipeline configuration
    st.markdown("---")
    st.subheader("🔍 Analysis Pipeline")

    analysis_options = st.multiselect(
        "Select Analysis Components",
        options=[
            "Key Term Extraction",
            "Financial Obligations",
            "Termination Clauses",
            "Renewal Options",
            "Assignment Provisions",
            "Maintenance Responsibilities",
            "Insurance Requirements",
            "Default Provisions",
            "Force Majeure",
            "Dispute Resolution",
            "Compliance Checklist",
            "Market Benchmarking"
        ],
        default=[
            "Key Term Extraction",
            "Financial Obligations",
            "Termination Clauses",
            "Renewal Options"
        ]
    )

    custom_instructions = st.text_area(
        "Custom Analysis Instructions",
        placeholder="E.g., Focus on early termination penalties, highlight any unusual clauses...",
        height=100
    )

    # Action buttons
    st.markdown("---")
    action_cols = st.columns([1,1,1,2])
    with action_cols[0]:
        analyze_btn = st.button("🚀 Analyze Documents", type="primary")
    with action_cols[1]:
        compare_btn = st.button("🔍 Compare Documents", disabled=len(uploaded_files) < 2)
    with action_cols[2]:
        template_btn = st.button("📋 Generate Summary Template")
    with action_cols[3]:
        export_format = st.selectbox(
            "Export Format",
            ["PDF Report", "Word Document", "Excel Matrix", "JSON Data"],
            index=0,
            label_visibility="collapsed"
        )

    # Processing logic
    if analyze_btn and uploaded_files:
        with st.status("🔍 Processing documents...", expanded=True) as status:
            progress_bar = st.progress(0)
            status_text = st.empty()
            doc_results = {}

            # Process each document
            for i, file in enumerate(uploaded_files):
                status_text.markdown(f"**Processing {file.name}** ({i+1}/{len(uploaded_files)})")

                try:
                    # Extract text
                    raw_text = extract_text_from_file(file)
                    if not raw_text.strip():
                        st.warning(f"Could not extract text from {file.name}")
                        continue

                    # Store document metadata
                    doc_results[file.name] = {
                        "metadata": doc_metadata.get(file.name, {}),
                        "raw_text": raw_text,
                        "analysis": {}
                    }

                    # Process each analysis component
                    for component in analysis_options:
                        prompt = build_analysis_prompt(
                            component,
                            raw_text,
                            analysis_mode,
                            jurisdiction,
                            risk_profile,
                            custom_instructions
                        )

                        # Call AI engine
                        if ai_engine == "Gemini Pro":
                            result = call_gemini("lease_analysis", prompt, temperature=temperature)
                        elif ai_engine == "Mistral Large":
                            messages = [{"role": "user", "content": prompt}]
                            result = call_mistral(messages, temperature=temperature)
                        elif ai_engine == "DeepSeek":
                            messages = [{"role": "user", "content": prompt}]
                            result = call_deepseek(messages, temperature=temperature)
                        else:  # Ensemble
                            results = []
                            for model in ["Gemini", "Mistral", "DeepSeek"]:
                                if model == "Gemini":
                                    results.append(call_gemini("lease_analysis", prompt))
                                elif model == "Mistral":
                                    messages = [{"role": "user", "content": prompt}]
                                    results.append(call_mistral(messages))
                                else:
                                    messages = [{"role": "user", "content": prompt}]
                                    results.append(call_deepseek(messages))
                            result = resolve_ensemble(results)

                        doc_results[file.name]["analysis"][component] = result

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue

            st.session_state.lease_docs = doc_results
            status.update(label="✅ Analysis complete!", state="complete")

        display_lease_results(doc_results, compare_mode)
        save_interaction(
            conn,
            "lease_analysis",
            f"Analyzed {len(uploaded_files)} documents with {analysis_mode} mode",
            json.dumps(doc_results, default=str, indent=2)
        )

    if compare_btn and len(st.session_state.lease_docs) >= 2:
        with st.spinner("Building comparison matrix..."):
            comparison_data = build_comparison_matrix(st.session_state.lease_docs)
            st.session_state.comparison_matrix = comparison_data
            display_comparison_results(comparison_data)

    if template_btn:
        with st.spinner("Generating summary template..."):
            template = generate_summary_template(st.session_state.lease_docs)
            st.subheader("Executive Summary Template")
            st.markdown(template)
            st.download_button(
                "Download Summary",
                data=template.encode('utf-8'),
                file_name="lease_summary_template.md",
                mime="text/markdown"
            )

def extract_risks(risk_text):
    """Extract risk items from analysis text"""
    risks = {}
    for line in risk_text.split('\n'):
        if "RISK:" in line:
            parts = line.split("RISK:")
            if len(parts) > 1:
                risk_level = parts[0].strip().upper()
                risk_desc = parts[1].strip()
                risks[risk_desc] = risk_level
    return risks

def extract_dates(date_text):
    """Extract key dates from analysis text"""
    dates = {}
    for line in date_text.split('\n'):
        if "-" in line and ("/" in line or any(month in line for month in [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ])):
            parts = line.split("-")
            if len(parts) > 1:
                date_desc = parts[0].strip()
                date_value = parts[1].strip()
                dates[date_desc] = date_value
    return dates

def extract_financial_terms(financial_text):
    """Extract financial terms from analysis text"""
    financials = {}
    for line in financial_text.split('\n'):
        if "$" in line or "£" in line or "€" in line or "amount" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                term = parts[0].strip()
                value = parts[1].strip()
                financials[term] = value
    return financials

def generate_redline(raw_text, analysis):
    """Generate redline version of lease with annotations"""
    redline_text = raw_text
    for component, content in analysis.items():
        if "Risk" in component:
            risks = extract_risks(content)
            for risk_desc, risk_level in risks.items():
                annotation = f"[{risk_level} RISK: {risk_desc}]"
                redline_text = redline_text.replace(risk_desc, annotation)
    return redline_text

def resolve_ensemble(results):
    """Resolve differences between multiple AI model outputs"""
    # Simple voting system - take the majority result
    from collections import Counter
    result_texts = [r for r in results if r.strip()]
    if not result_texts:
        return ""
    
    # For structured data, find most common elements
    if all("\n- " in r for r in result_texts):
        all_items = []
        for r in result_texts:
            items = [line.strip() for line in r.split('\n') if line.strip()]
            all_items.extend(items)
        counter = Counter(all_items)
        most_common = counter.most_common()
        return "\n".join([item for item, count in most_common if count > 1])
    
    # For free text, return the longest result (most detailed)
    return max(result_texts, key=len)

def build_analysis_prompt(component, text, mode, jurisdiction, risk, custom_instructions):
    """Construct detailed prompt for each analysis component"""
    component_prompts = {
        "Key Term Extraction": (
            f"Analyze this lease agreement and extract all key terms. "
            f"Organize into these categories:\n"
            f"1. Basic Terms (parties, property, term dates)\n"
            f"2. Financial Terms (rent, deposits, fees)\n"
            f"3. Operational Terms (use restrictions, maintenance)\n"
            f"4. Legal Terms (jurisdiction, dispute resolution)\n\n"
            f"Present in clear markdown tables with explanations. "
            f"Highlight any unusual terms. Jurisdiction: {jurisdiction}. "
            f"Risk sensitivity: {risk}. {custom_instructions}"
        ),
        "Financial Obligations": (
            f"Identify all financial obligations in this lease including:\n"
            f"- Base rent amount and payment schedule\n"
            f"- Security deposit and conditions for return\n"
            f"- Additional fees (CAM, utilities, taxes)\n"
            f"- Rent escalation clauses\n"
            f"- Penalties for late payment\n\n"
            f"Calculate effective annual costs considering all factors. "
            f"Highlight any unusual or onerous provisions. {custom_instructions}"
        ),
        "Termination Clauses": (
            f"Analyze the termination provisions in this lease. Identify:\n"
            f"- Early termination options and penalties\n"
            f"- Notice periods required\n"
            f"- Conditions for landlord/tenant termination\n"
            f"- Automatic renewal clauses\n"
            f"- Holdover provisions\n\n"
            f"Assess the balance of power in these clauses. "
            f"Flag any unusually restrictive terms. {custom_instructions}"
        )
    }
    
    base_prompt = (
        f"Perform {component} analysis on this lease agreement.\n"
        f"Analysis Mode: {mode}\n"
        f"Jurisdiction: {jurisdiction}\n"
        f"Risk Profile: {risk}\n\n"
        f"Document Excerpt:\n{text[:10000]}\n\n"
        f"Provide response in structured markdown with clear headings. "
        f"Include risk ratings where appropriate."
    )
    return component_prompts.get(component, base_prompt)

def display_lease_results(results, compare_mode):
    """Display analysis results with interactive elements"""
    st.subheader("Analysis Results")
    tab_names = [f"📄 {name}" for name in results.keys()]
    tabs = st.tabs(tab_names)

    for i, (doc_name, doc_data) in enumerate(results.items()):
        with tabs[i]:
            st.markdown(f"### {doc_name}")
            st.caption(f"Lease Type: {doc_data['metadata'].get('type', 'Unknown')}")

            # Document overview
            with st.expander("📋 Document Overview", expanded=True):
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Text Length", f"{len(doc_data['raw_text']):,} chars")
                with cols[1]:
                    st.metric("Key Terms", len(doc_data['analysis'].get('Key Term Extraction', '').split('\n')))
                with cols[2]:
                    risks = sum(1 for line in doc_data['analysis'].get('Risk Assessment', '').split('\n') if "HIGH" in line)
                    st.metric("High Risks", risks, delta_color="inverse")

            # Analysis sections
            for component, content in doc_data['analysis'].items():
                with st.expander(f"🔍 {component}"):
                    if "Risk" in component:
                        display_risk_content(content)
                    else:
                        st.markdown(content)

            # Quick actions
            st.markdown("### Quick Actions")
            action_cols = st.columns(3)
            with action_cols[0]:
                st.download_button(
                    "Download Analysis",
                    data=json.dumps(doc_data, indent=2).encode('utf-8'),
                    file_name=f"{doc_name}_analysis.json",
                    mime="application/json"
                )
            with action_cols[1]:
                if st.button("Generate Redline", key=f"redline_{i}"):
                    with st.spinner("Generating redline version..."):
                        redline = generate_redline(doc_data['raw_text'], doc_data['analysis'])
                        st.session_state[f"redline_{doc_name}"] = redline
                        st.rerun()
            with action_cols[2]:
                if compare_mode:
                    st.checkbox("Include in Comparison", value=True, key=f"compare_{i}")

def display_risk_content(content):
    """Format risk assessment content with color coding"""
    lines = content.split('\n')
    for line in lines:
        if "HIGH RISK" in line:
            st.markdown(f'<div class="risk-high">{line}</div>', unsafe_allow_html=True)
        elif "MEDIUM RISK" in line:
            st.markdown(f'<div class="risk-medium">{line}</div>', unsafe_allow_html=True)
        elif "LOW RISK" in line:
            st.markdown(f'<div class="risk-low">{line}</div>', unsafe_allow_html=True)
        else:
            st.write(line)

def build_comparison_matrix(docs):
    """Create a comparison matrix across multiple documents"""
    comparison = {
        "metadata": {},
        "financial_terms": {},
        "key_dates": {},
        "risk_factors": {}
    }

    for doc_name, doc_data in docs.items():
        # Metadata comparison
        comparison["metadata"][doc_name] = doc_data["metadata"]

        # Financial terms comparison
        financial_data = extract_financial_terms(doc_data["analysis"].get("Financial Obligations", ""))
        comparison["financial_terms"][doc_name] = financial_data

        # Key dates comparison
        dates_data = extract_dates(doc_data["analysis"].get("Key Term Extraction", ""))
        comparison["key_dates"][doc_name] = dates_data

        # Risk factors comparison
        risks_data = extract_risks(doc_data["analysis"].get("Risk Assessment", ""))
        comparison["risk_factors"][doc_name] = risks_data

    return comparison

def display_comparison_results(comparison):
    """Display interactive comparison results"""
    st.subheader("📊 Cross-Document Comparison")

    with st.expander("📋 Metadata Comparison", expanded=True):
        meta_df = pd.DataFrame(comparison["metadata"]).T
        st.dataframe(meta_df.style.highlight_max(axis=0, color='#d4edda'))

    with st.expander("💰 Financial Terms Comparison"):
        financial_df = pd.DataFrame(comparison["financial_terms"]).T
        st.dataframe(
            financial_df.style.format("${:,.2f}").highlight_max(axis=0, color='#ffdddd')
        )
        fig = px.bar(
            financial_df,
            barmode='group',
            title="Financial Terms Comparison"
        )
        st.plotly_chart(fig)

    with st.expander("📅 Key Dates Comparison"):
        dates_df = pd.DataFrame(comparison["key_dates"]).T
        st.dataframe(dates_df)

    with st.expander("⚠️ Risk Profile Comparison"):
        risks_df = pd.DataFrame(comparison["risk_factors"]).T
        st.dataframe(risks_df.style.highlight_max(axis=0, color='#ffdddd'))
        fig = px.line_polar(
            risks_df.reset_index(),
            r=risks_df.mean(axis=1),
            theta=risks_df.index,
            line_close=True,
            title="Comparative Risk Profiles"
        )
        st.plotly_chart(fig)

def generate_summary_template(docs):
    """Generate an executive summary template from analysis results"""
    template = "# Lease Portfolio Summary\n\n"
    template += "## Key Findings\n\n"

    for doc_name, doc_data in docs.items():
        template += f"### {doc_name}\n"
        template += f"- **Type**: {doc_data['metadata'].get('type', 'Unknown')}\n"

        # Financial highlights
        financial = doc_data['analysis'].get('Financial Obligations', '')
        template += "\n**Financial Highlights**:\n"
        for line in financial.split('\n')[:5]:
            if line.strip() and any(c in line for c in ['$', '£', '€']):
                template += f"- {line}\n"

        # Top risks
        risks = doc_data['analysis'].get('Risk Assessment', '')
        high_risks = [line for line in risks.split('\n') if "HIGH" in line]
        if high_risks:
            template += "\n**Critical Risks**:\n"
            for risk in high_risks[:3]:
                template += f"- {risk.split('HIGH RISK:')[-1].strip()}\n"

        template += "\n---\n"

    if len(docs) > 1:
        template += "\n## Comparative Analysis\n"
        template += "Key differences across documents:\n\n"
        template += "- [Add key differences here]\n"
        template += "- [Consider creating a comparison table]\n"
        template += "- [Highlight most favorable terms]\n"

    return template

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