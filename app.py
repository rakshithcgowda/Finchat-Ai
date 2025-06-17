import io
import json
import os
import sqlite3
import time
import bcrypt
import docx
from docx.shared import Pt
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
import re
from openai import OpenAI
from PIL import Image
import pdfplumber
import torch
from torchvision import transforms
import uuid
import base64
import tempfile
from docx import Document
import logging
import sqlite3
from contextlib import contextmanager

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
    base_url="https://api.deepseek.com/v1"
)

OCR_API_KEY = "PDF8V7778Y0TX"

BRAND_COLORS = {
    "primary": "#2E86AB",
    "secondary": "#F18F01",
    "background": "#F7F7F7",
    "text": "#121111"
}

genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ─── OCR.space API Functions ───────────────────────────────────────────────────
OCR_ENDPOINTS = [
    "https://apipro1.ocr.space",
    "https://apipro2.ocr.space",
]

def ocr_space_file_pro(
    filename: str,
    api_key: str = OCR_API_KEY,
    language: str = "eng",
    overlay: bool = False,
    engine: int = 2,
    create_searchable_pdf: bool = True,
    hide_text_layer: bool = True,
    retries: int = 3,
    timeout: int = 120,
) -> dict:
    """
    Send a PDF or image to OCR.space PRO endpoint.
    Tries each datacenter endpoint in turn, with exponential backoff.
    """
    mimetype = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png"
    }.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")

    payload = {
        "apikey": api_key,
        "language": language,
        "isOverlayRequired": overlay,
        "scale": True,
        "OCREngine": engine,
        "isCreateSearchablePdf": create_searchable_pdf,
        "isSearchablePdfHideTextLayer": hide_text_layer,
    }

    last_err = None
    for endpoint in OCR_ENDPOINTS:
        for attempt in range(1, retries + 1):
            try:
                with open(filename, "rb") as f:
                    files = {"file": (os.path.basename(filename), f, mimetype)}
                    resp = requests.post(
                        endpoint + "/parse/image",
                        data=payload,
                        files=files,
                        timeout=timeout
                    )
                resp.raise_for_status()
                result = resp.json()
                if result.get("IsErroredOnProcessing"):
                    raise RuntimeError(result.get("ErrorMessage") or "OCR processing error")
                return result
            except Exception as e:
                last_err = e
                sleep = 2 ** attempt
                time.sleep(sleep)
        # if this endpoint failed all retries, move to the next
    raise RuntimeError(f"All OCR endpoints failed: {last_err}")




def ocr_space_url(
    url: str,
    api_key: str = OCR_API_KEY,
    language: str = "eng",
    overlay: bool = False,
    engine: int = 2,
    retries: int = 3,
    timeout: int = 60,
) -> dict:
    """
    Send an image URL to OCR.space PRO endpoint.
    Rotates through endpoints if necessary.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    params = {
        "apikey": api_key,
        "url": url,
        "language": language,
        "isOverlayRequired": overlay,
        "detectOrientation": True,
        "scale": True,
        "OCREngine": engine
    }

    last_err = None
    for endpoint in OCR_ENDPOINTS:
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(
                    endpoint + "/parse/imageurl",
                    params=params,
                    timeout=timeout
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("IsErroredOnProcessing"):
                    raise RuntimeError(result.get("ErrorMessage") or "OCR processing error")
                return result
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        # try next endpoint
    raise RuntimeError(f"All OCR endpoints failed: {last_err}")


# ─── Database Helpers ──────────────────────────────────────────────────────────
DB_PATH = "users.db"  # Define a constant for the database path

# ─── Database Helpers ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database context manager
@contextmanager
def get_db_connection(db_path: str = DB_PATH):
    conn = None
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize database
def init_db(db_path: str = DB_PATH):
    """
    Initialize the database only if it doesn't exist.
    Creates necessary tables and ensures schema updates are non-destructive.
    Returns a database connection.
    """
    db_exists = os.path.exists(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password BLOB NOT NULL,
                role TEXT NOT NULL,
                location_id TEXT,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("PRAGMA table_info(users)")
        existing_cols = {col[1] for col in cursor.fetchall()}
        for col, col_type in [
            ("location_id", "TEXT"),
            ("last_login", "TIMESTAMP"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ]:
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
                    logging.info(f"Added {col} column to users table")
                except sqlite3.OperationalError as e:
                    logging.warning(f"Could not add {col} column: {e}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                username TEXT PRIMARY KEY,
                lease_analysis INTEGER DEFAULT 0,
                deal_structuring INTEGER DEFAULT 0,
                offer_generator INTEGER DEFAULT 0,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                feature TEXT,
                input_text TEXT,
                output_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        conn.commit()

        if not db_exists:
            create_default_admin(conn)

        return conn
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
        st.error(f"Failed to initialize database: {e}")
        conn.close()
        raise

# Verify password
def verify_password(hashed: bytes, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

def create_default_admin(conn):
    """Create a default admin user if none exists."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            admin_pwd = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", admin_pwd, "admin")
            )
            cursor.execute(
                "INSERT INTO subscriptions (username, lease_analysis, deal_structuring, offer_generator) VALUES (?, ?, ?, ?)",
                ("admin", 1, 1, 1)
            )
            conn.commit()
            logging.info("Default admin user created")
    except sqlite3.Error as e:
        logging.error(f"Failed to create default admin: {e}")
        raise

def verify_password(hashed: bytes, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

# ─── AI Helper Functions ───────────────────────────────────────────────────────
def call_gemini(
    feature: str,
    content: str,
    temperature: float = 0.7
) -> str:
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
        ),
        "lease_questions": (
            "You are a real estate expert answering questions about a lease agreement based on its summary. "
            "Provide accurate and concise answers using only the information in the summary."
        )
    }

    try:
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
            return "".join(chunk.choices[0].delta.content for chunk in resp)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error processing request with DeepSeek: {str(e)}"

def save_interaction(conn, feature: str, input_text: str, output_text: str):
    if st.session_state.get("username"):
        conn.execute(
            "INSERT INTO interactions (username, feature, input_text, output_text) VALUES (?, ?, ?, ?)",
            (st.session_state.username, feature, input_text, output_text),
        )
        conn.commit()

def convert_pdf_to_images(pdf_file, output_dir, dpi=300):
    """Convert PDF to high-quality images for OCR"""
    images = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                img = page.to_image(resolution=dpi).original
                img_path = os.path.join(output_dir, f"page_{i+1}.png")
                img.save(img_path, "PNG", quality=100)
                images.append(img_path)
    except Exception as e:
        st.error(f"PDF to image conversion failed: {e}")
    return images

def preprocess_image_for_ocr(img):
    """Enhance image quality for better OCR results"""
    try:
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Remove noise
        img = img.point(lambda x: 0 if x < 128 else 255, '1')

        return img
    except Exception as e:
        st.warning(f"Image preprocessing failed: {e}")
        return img


# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_text_with_ocr(uploaded_file=None, file_type: str = "pdf", url: str = None):
    """Enhanced text extraction with OCR.space API for PDFs and images.
    - Direct per-page extraction with fallback to per-page OCR
    - Merges direct and OCR results to maintain page order
    - Detailed logging for debugging
    """
    def validate_extracted_text(text: str) -> bool:
        text = text.strip()
        return bool(text) and len(text.split()) >= 2

    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()

    try:
        pages_text = []
        used_ocr = False

        # Handle uploaded file case
        if uploaded_file and file_type.lower() == "pdf":
            reader = PdfReader(uploaded_file)
            # ── Try direct per-page extraction ─────────────────────────
            for i, page in enumerate(reader.pages):
                raw = page.extract_text() or ""
                cleaned = clean_text(raw)
                if validate_extracted_text(cleaned):
                    pages_text.append(cleaned)
                else:
                    logging.info(f"Page {i+1} empty; falling back to per-page OCR.")
                    # Create single-page PDF
                    from PyPDF2 import PdfWriter
                    writer = PdfWriter()
                    writer.add_page(page)
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as onepagetmp:
                        writer.write(onepagetmp)
                        tmp_pdf = onepagetmp.name

                    # Convert to image
                    images = convert_pdf_to_images(tmp_pdf, tempfile.gettempdir(), dpi=300)
                    os.unlink(tmp_pdf)

                    # OCR the first image
                    page_text = ""
                    if images:
                        img = Image.open(images[0])
                        img = preprocess_image_for_ocr(img)
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgf:
                            img.save(imgf.name, "PNG", quality=100)
                            resp = ocr_space_file_pro(
                                filename=imgf.name,
                                api_key=OCR_API_KEY,
                                language='eng',
                                retries=2,
                                timeout=60
                            )
                        os.unlink(imgf.name)
                        page_text = clean_text(
                            resp.get("ParsedResults", [{}])[0].get("ParsedText", "")
                        )
                        used_ocr = True
                    pages_text.append(page_text)

                    # Clean up intermediate images
                    for img_path in images:
                        try:
                            os.unlink(img_path)
                        except:
                            pass

            return pages_text

        # Handle uploaded file case
        # Handle uploaded file
        if uploaded_file:
            if file_type.lower() == "pdf":
                # 1. Try direct text extraction first
                try:
                    reader = PdfReader(uploaded_file)
                    direct_pages = [clean_text(p.extract_text() or "") for p in reader.pages]
                except Exception as e:
                    logging.warning(f"Direct PDF extraction failed: {e}")
                    direct_pages = []
                    # ————— Updated boilerplate detection —————
                    bp_phrase = "incomplete without the preceding notes"
                    bp_count = sum(1 for p in direct_pages if bp_phrase in p.lower())
                    if bp_count >= 0.8 * len(direct_pages):
                        logging.info(f"Detected {bp_count}/{len(direct_pages)} boilerplate pages; forcing OCR.")
                        direct_pages = []
                    # ————————————————————————————————

                # ————————— Patch starts here —————————
                unique_pages = set(direct_pages)
                if len(unique_pages) == 1 and 'official copy' in next(iter(unique_pages)).lower():
                    logging.info("Detected boilerplate repetition; forcing OCR extraction.")
                    direct_pages = []
                # ————————— Patch ends here —————————

                # 2. If any page has real text, trust direct extraction
                if any(len(p.split()) >= 2 for p in direct_pages):
                    logging.info("Using direct PDF text extraction, skipping OCR.")
                    return direct_pages

                # 3. Otherwise fall back to image-based OCR
                with tempfile.TemporaryDirectory() as temp_dir:
                    uploaded_file.seek(0)
                    images = convert_pdf_to_images(uploaded_file, temp_dir, dpi=300)
                    if not images:
                        logging.error("Failed to convert PDF to images for OCR.")
                        return []

                    progress = st.progress(0)
                    ocr_pages = []
                    for i, img_path in enumerate(images):
                        try:
                            img = Image.open(img_path)
                            img = preprocess_image_for_ocr(img)
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                                img.save(tmpf.name, "PNG", quality=100)
                                resp = ocr_space_file_pro(
                                    filename=tmpf.name,
                                    api_key=OCR_API_KEY,
                                    language='eng',
                                    retries=2,
                                    timeout=60
                                )
                            parsed = resp.get("ParsedResults", [{}])[0].get("ParsedText", "")
                            ocr_pages.append(clean_text(parsed))
                            used_ocr = True
                            os.unlink(tmpf.name)
                        except Exception as e:
                            logging.error(f"OCR failed for page {i+1}: {e}")
                            ocr_pages.append("")
                        progress.progress((i + 1) / len(images))

                    # 4. Merge direct vs OCR
                    for direct, ocr in zip(direct_pages, ocr_pages):
                        if validate_extracted_text(direct):
                            pages_text.append(direct)
                        elif validate_extracted_text(ocr):
                            pages_text.append(ocr)
                        else:
                            pages_text.append(clean_text(direct + " " + ocr))

            # Image processing (JPG)
            else:
                st.info("Processing uploaded JPG with OCR.space API...")
                logging.info("Processing uploaded JPG")
                uploaded_file.seek(0)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    img = Image.open(uploaded_file)
                    img = preprocess_image_for_ocr(img)
                    img.save(temp_file.name, quality=95)
                    response = ocr_space_file_pro(
                        filename=temp_file.name,
                        api_key=OCR_API_KEY,
                        language='eng',
                        retries=2,
                        timeout=60
                    )
                    os.unlink(temp_file.name)
                parsed_text = response.get("ParsedResults", [{}])[0].get("ParsedText", "")
                cleaned_text = clean_text(parsed_text)
                if validate_extracted_text(cleaned_text):
                    pages_text.append(cleaned_text)
                    used_ocr = True
                    logging.info(f"OCR for JPG: {len(cleaned_text)} chars")
                else:
                    pages_text.append("")
                    st.error("OCR failed to extract meaningful text from JPG.")
                    logging.error("No meaningful text from JPG")

        # Validate final results
        if not any(validate_extracted_text(p) for p in pages_text):
            st.error("Failed to extract meaningful text from the document.")
            logging.error("No meaningful text extracted from document")
            return []

        st.session_state['used_ocr'] = used_ocr
        logging.info(f"Extraction complete: {len(pages_text)} pages, used_ocr={used_ocr}")
        return pages_text

    except Exception as e:
        logging.error(f"extract_text_with_ocr failed: {e}", exc_info=True)
        return []


def lease_summarization_ui(conn):
    """Lease Summarization: upload PDF or JPG, or provide a JPG URL, and get a full-document summary after OCR extraction. Allows uploading a Word document with any content based on the summary and generates relevant output."""
    st.header("📄 Lease Summary")

    # Clear previous summary and related data
    if st.button("Clear Summary", key="clear_lease_summary"):
        for k in ['last_file', 'last_url', 'last_summary', 'last_mode', 'last_engine', 'last_docx_content', 'last_docx_output', 'extracted_pages', 'used_ocr', 'last_selected_page_index']:
            st.session_state.pop(k, None)
        st.success("Cleared previous summary and related content.")
        st.rerun()

    st.markdown(
        "Upload your lease PDF or JPG image, or provide a public JPG URL. Text will be extracted using OCR and summarized as a full document."
    )

    # File uploader for PDF or JPG
    uploaded_file = st.file_uploader(
        "Upload Lease Document (PDF or JPG)", type=["pdf", "jpg", "jpeg"], key="lease_file_uploader"
    )

    # URL input for remote JPG
    image_url = st.text_input("Or Enter Public JPG URL", key="lease_image_url", placeholder="e.g., https://example.com/image.jpg")

    # Validate inputs
    if uploaded_file and image_url:
        st.error("Please provide either a file or a URL, not both.")
        return
    elif not uploaded_file and not image_url:
        return

    # Determine input type
    if uploaded_file:
        file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "jpg"
        input_identifier = uploaded_file.name
    else:
        file_type = "jpg"
        input_identifier = image_url

    # Check if input has changed
    if 'last_file' in st.session_state and uploaded_file and st.session_state.last_file != uploaded_file.name:
        for k in ['last_summary', 'last_mode', 'last_engine', 'last_docx_content', 'last_docx_output', 'extracted_pages', 'used_ocr', 'last_selected_page_index', 'last_url']:
            st.session_state.pop(k, None)
    elif 'last_url' in st.session_state and image_url and st.session_state.last_url != image_url:
        for k in ['last_summary', 'last_mode', 'last_engine', 'last_docx_content', 'last_docx_output', 'extracted_pages', 'used_ocr', 'last_selected_page_index', 'last_file']:
            st.session_state.pop(k, None)

    # AI engine selection
    ai_engine = st.radio(
        "Select AI Model",
        ["in-depth"],
        index=0,
        horizontal=True,
        key="lease_ai_engine"
    )

    # Extract text if not already done or if input has changed
    if 'extracted_pages' not in st.session_state or \
       (uploaded_file and st.session_state.get('last_file') != uploaded_file.name) or \
       (image_url and st.session_state.get('last_url') != image_url):
        if st.button("Extract Text", key="lease_extract_button"):
            pages = extract_text_with_ocr(uploaded_file=uploaded_file, file_type=file_type, url=image_url)
            if not pages:
                st.error(f"No readable text found in the {'PDF' if file_type == 'pdf' else 'JPG'} after OCR.")
                return

            # Store extracted pages and input identifier
            st.session_state['extracted_pages'] = pages
            st.session_state['used_ocr'] = True
            if uploaded_file:
                st.session_state['last_file'] = uploaded_file.name
            else:
                st.session_state['last_url'] = image_url
            st.rerun()

    # Display extracted text preview if available
    if 'extracted_pages' in st.session_state:
        st.subheader("Extracted Text Preview")
        page_options = [f"Page {i+1}" for i in range(len(st.session_state['extracted_pages']))] if file_type == "pdf" else ["Image"]
        selected_page = st.selectbox(
            "Select Page to View Extracted Text",
            page_options,
            key="extracted_text_dropdown",
            index=st.session_state.get('last_selected_page_index', 0)
        )
        st.session_state['last_selected_page_index'] = page_options.index(selected_page)
        page_index = page_options.index(selected_page)
        extracted_text = st.session_state['extracted_pages'][page_index]
        if extracted_text.strip():
            st.text_area(
                f"Extracted Text for {selected_page} (via OCR)",
                value=extracted_text,
                height=300,
                key=f"extracted_text_page_{page_index}"
            )
        else:
            st.warning(f"No text extracted for {selected_page}. Try a different file or check the document quality.")

        # Generate full document summary
        if st.button("Generate Summary", key="lease_generate_button"):
            pages = st.session_state['extracted_pages']
            if not any(p.strip() for p in pages):
                st.error("No readable text available for summarization.")
                return

            st.session_state['last_engine'] = ai_engine

            text = "\n".join(pages)
            with st.spinner("Summarizing full document..."):
                summaries = []
                chunks = [text[i:i+15000] for i in range(0, len(text), 15000)] if len(text) > 15000 else [text]
                for chunk in chunks:
                    prompt = (
                        "Summarize this portion of the lease agreement in clear, concise language, "
                        "preserving all key details:\n\n" + chunk
                    )
                    summaries.append(call_mistral(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1024
                    ))
                final = "\n\n".join(summaries)
            st.subheader("Full Document Summary")
            st.markdown(final)
            save_interaction(conn, "lease_summary_full", input_identifier, final)
            st.session_state['last_summary'] = final
            st.rerun()

    # Display existing summary if available
    if 'last_summary' in st.session_state and \
       ((uploaded_file and st.session_state.get('last_file') == uploaded_file.name) or \
        (image_url and st.session_state.get('last_url') == image_url)):
        engine = st.session_state['last_engine']
        summary_content = st.session_state['last_summary']
        st.subheader(f"Full Document Summary ({engine})")
        st.markdown(summary_content)
        st.divider()

        # Export section
        st.markdown("### 📥 Export Styled Summary")

        # Derive default filename
        file_base = input_identifier.rsplit(".", 1)[0]
        file_name = st.text_input("Filename (no extension):", value=file_base, key="lease_export_name")

        # Sanitize content
        safe_content = summary_content.encode('latin-1', 'replace').decode('latin-1')
        paragraphs_pdf = [p.strip() for p in safe_content.split("\n\n") if p.strip()]
        paragraphs_word = [p.strip() for p in summary_content.split("\n\n") if p.strip()]

        # PDF Export
        class LeasePDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 20)
                self.set_text_color(30, 90, 140)
                self.cell(0, 12, "Lease Agreement Summary", ln=1, align='C')
                self.set_font('Arial', 'I', 10)
                self.set_text_color(120, 120, 120)
                date_str = datetime.now().strftime("%B %d, %Y")
                self.cell(0, 6, date_str, ln=1, align='C')
                self.set_draw_color(200, 200, 200)
                self.set_line_width(0.4)
                y = self.get_y() + 2
                self.line(15, y, 195, y)
                self.ln(8)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.set_text_color(130, 130, 130)
                page_info = f"Page {self.page_no()} / {{nb}}"
                self.cell(0, 10, page_info, align='C')

            def section_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.set_fill_color(245, 245, 245)
                self.set_text_color(30, 90, 140)
                self.cell(0, 8, title, ln=1, fill=True, border=0)
                x1, x2 = self.l_margin, self.w - self.r_margin
                y = self.get_y() - 1
                self.set_draw_color(30, 90, 140)
                self.set_line_width(0.3)
                self.line(x1, y, x2, y)
                self.ln(4)

            def paragraph(self, text):
                self.set_font('Arial', '', 12)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 7, text.encode('latin-1', 'replace').decode('latin-1'))
                self.ln(4)

            def render_summary_md(self, md_text):
                lines = md_text.splitlines()
                list_level = 0
                for raw in lines:
                    line = raw.rstrip()
                    if not line:
                        self.ln(4)
                        continue
                    if line.startswith("### "):
                        self.section_title(line[4:].strip())
                        continue
                    m = re.match(r"\*\*(.+?)\*\*:", line)
                    if m:
                        self.set_font('Arial', 'B', 12)
                        self.set_text_color(30, 90, 140)
                        self.cell(0, 6, m.group(1) + ":", ln=1)
                        self.ln(2)
                        continue
                    m = re.match(r"\*(.+?)\*", line)
                    if m:
                        self.set_font('Arial', 'I', 12)
                        self.set_text_color(80, 80, 80)
                        self.multi_cell(0, 7, m.group(1))
                        self.ln(2)
                        self.set_font('Arial', '', 12)
                        self.set_text_color(50, 50, 50)
                        continue
                    m = re.match(r"^(\d+)\.\s+(.*)", line)
                    if m:
                        num, text = m.groups()
                        indent = 10 * list_level
                        self.set_x(self.l_margin + indent)
                        self.set_font('Arial', 'B', 11)
                        self.cell(8, 7, f"{num}.", ln=0)
                        self.set_font('Arial', '', 12)
                        self.multi_cell(0, 7, text)
                        list_level = 1
                        continue
                    m = re.match(r"^\s*-\s+(.*)", line)
                    if m:
                        text = m.group(1)
                        indent = 10 * list_level
                        self.set_x(self.l_margin + indent)
                        self.set_font('Arial', '', 12)
                        self.cell(5, 7, "*")
                        self.multi_cell(0, 7, text)
                        list_level = 1
                        continue
                    if list_level and not (line.startswith("- ") or re.match(r"^\d+\.\s+", line)):
                        list_level = 0
                        self.ln(2)
                    self.set_font('Arial', '', 12)
                    self.multi_cell(0, 7, line)
                    self.ln(2)

        pdf = LeasePDF()
        pdf.set_auto_page_break(True, margin=20)
        pdf.add_page()
        pdf.set_left_margin(20)
        pdf.set_right_margin(20)
        pdf.set_title(f"Lease Summary – {file_name}")
        pdf.set_author("Property Deals AI")

        pdf.section_title("Document Information")
        info = [
            ("Input File", input_identifier),
            ("Mode", "Full Document"),
            ("AI Model", engine),
            ("Generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for i, (label, val) in enumerate(info):
            pdf.set_fill_color(245, 245, 245) if i % 2 else pdf.set_fill_color(255, 255, 255)
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(50, 6, f"{label}:", ln=0, fill=True)
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 6, val.encode('latin-1', 'replace').decode('latin-1'), ln=1, fill=True)
        pdf.ln(6)

        pdf.section_title("Lease Summary")
        pdf.render_summary_md(safe_content)

        if st.session_state.get('last_docx_content'):
            pdf.add_page()
            pdf.section_title("Additional Processing")
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, "Original Document Text:", ln=1)
            pdf.render_summary_md(st.session_state['last_docx_content'])
            pdf.ln(4)
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, "AI Summary Response:", ln=1)
            pdf.render_summary_md(st.session_state.get('last_docx_output', ''))

        try:
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
            st.download_button(
                "Download Styled PDF",
                pdf_bytes,
                file_name=f"{file_name}.pdf",
                mime="application/pdf",
                key="lease_export_pdf"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

        # Word Export
        doc = docx.Document()
        sections = doc.sections
        for sec in sections:
            sec.top_margin = docx.shared.Cm(2.5)
            sec.bottom_margin = docx.shared.Cm(2.5)
            sec.left_margin = docx.shared.Cm(2.0)
            sec.right_margin = docx.shared.Cm(2.0)

        doc.styles['Heading 1'].font.name = 'Calibri'
        doc.styles['Heading 1'].font.size = Pt(16)
        doc.styles['Heading 1'].font.color.rgb = docx.shared.RGBColor(44, 134, 171)
        doc.styles['Normal'].font.name = 'Arial'
        doc.styles['Normal'].font.size = Pt(12)

        doc.add_heading("Lease Agreement Summary", level=1)
        doc.add_paragraph(f"Mode: Full Document    AI Model: {engine}", style='Normal')

        for para in paragraphs_word:
            doc.add_paragraph(para, style='Normal')

        buf = io.BytesIO()
        doc.save(buf)
        st.download_button(
            "Download Styled Word",
            buf.getvalue(),
            file_name=f"{file_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="lease_export_word"
        )

        if doc and 'last_docx_content' in st.session_state and 'last_docx_output' not in st.session_state:
            if st.button("Generate Response", key="lease_generate_response"):
                with st.spinner("Generating response based on uploaded document..."):
                    prompt = (
                        f"Lease Summary:\n{summary_content}\n\n"
                        f"Uploaded Document Content:\n{st.session_state['last_docx_content']}\n\n"
                        "Analyze the uploaded document content and generate a relevant response based on the lease summary. "
                        "The content may include comments, instructions, or additional details. Provide a clear and appropriate response, such as further analysis, clarifications, or commentary, as needed."
                    )
                    response = call_mistral(
                        messages=[
                            {"role": "system", "content": "You are a real estate expert processing additional content related to a lease summary."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )
                    st.session_state['last_docx_output'] = response
                    save_interaction(conn, "lease_docx_processing", st.session_state['last_docx_content'], response)
                    st.rerun()


def deal_structuring_ui(conn):
    """Enhanced deal structuring with persistent strategy chat and detailed strategies."""
    st.header("💡 Creative Deal Structuring Bot")
    st.markdown("Get AI-powered strategies tailored to your property deal based on buyer and seller details.")

    # Initialize session state
    if "deal_strategy_memory" not in st.session_state:
        st.session_state.deal_strategy_memory = []
        st.session_state.last_strategies = None
        st.session_state.strategy_confidences = {}

    # Clear chat and strategies
    if st.button("Clear Strategies & Chat", key="clear_strategies"):
        st.session_state.deal_strategy_memory.clear()
        st.session_state.last_strategies = None
        st.session_state.strategy_confidences = {}
        st.rerun()

    # Input form for Buyer and Seller details
    with st.expander("Deal Details", expanded=True):
        st.markdown("### Buyer Situation")
        col1, col2 = st.columns(2)
        with col1:
            deposit = st.selectbox("Available Deposit/Upfront Cash", ["Low/Zero", "Moderate", "High"], key="buyer_deposit")
            credit = st.selectbox("Credit & Mortgage Ability", ["Excellent", "Average", "Poor", "None"], key="buyer_credit")
            experience = st.selectbox("Property Investing Experience", ["No creative strategy done yet", "Have done at least one"], key="buyer_experience")
        with col2:
            risk_appetite = st.selectbox("Risk Appetite", ["Low – very cautious", "Medium – balanced", "High – very comfortable with risk"], key="buyer_risk")
            goal = st.selectbox("Primary Goal", ["Hold long-term (rental/residence)", "Resell for profit ASAP"], key="buyer_goal")
            timeline = st.selectbox("Investment Timeline", ["Few months", "1-2 years", "Longer period"], key="buyer_timeline")

        st.markdown("### Seller Motivation & Situation")
        col1, col2 = st.columns(2)
        with col1:
            seller_motivation = st.text_area("Seller's Motivation & Urgency", placeholder="e.g., relocating, financial distress, etc.", key="seller_motivation")
            financial_difficulties = st.selectbox("Seller Financial Difficulties", ["None", "Mortgage Arrears", "Negative Equity", "Other"], key="seller_financial")
        with col2:
            if financial_difficulties == "Mortgage Arrears":
                months_behind = st.number_input("Months Behind on Mortgage", min_value=0, step=1, key="seller_arrears_months")
                repossession_deadline = st.text_input("Repossession Deadline (if any)", placeholder="e.g., MM/DD/YYYY", key="seller_repossession")
            elif financial_difficulties == "Negative Equity":
                defer_payment = st.selectbox("Seller Willing to Defer Payment?", ["Yes", "No", "Maybe"], key="seller_defer_payment")
            elif financial_difficulties == "Other":
                other_difficulties = st.text_area("Specify Other Financial Difficulties", key="seller_other_difficulties")
        property_type = st.selectbox("Property Type", ["Residential", "Commercial", "Mixed-Use", "Land"], key="property_type")
        occupancy_status = st.selectbox("Occupancy Status", ["Vacant", "Owner-Occupied", "Tenant-Occupied", "Other"], key="occupancy_status")
        property_condition = st.text_area("Property Condition & Repairs Needed", placeholder="e.g., good condition, needs roof replacement, etc.", key="property_condition")

    # Generate strategies
    if st.button("Generate Strategies", type="primary", key="gen_strat"):
        prompt = (
            f"Buyer Situation:\n"
            f"- Available Deposit/Upfront Cash: {deposit}\n"
            f"- Credit & Mortgage Ability: {credit}\n"
            f"- Property Investing Experience: {experience}\n"
            f"- Risk Appetite: {risk_appetite}\n"
            f"- Primary Goal: {goal}\n"
            f"- Investment Timeline: {timeline}\n\n"
            f"Seller Motivation & Situation:\n"
            f"- Motivation & Urgency: {seller_motivation}\n"
            f"- Financial Difficulties: {financial_difficulties}\n"
        )
        if financial_difficulties == "Mortgage Arrears":
            prompt += f"- Months Behind: {months_behind}\n- Repossession Deadline: {repossession_deadline}\n"
        elif financial_difficulties == "Negative Equity":
            prompt += f"- Willing to Defer Payment: {defer_payment}\n"
        elif financial_difficulties == "Other":
            prompt += f"- Other Difficulties: {other_difficulties}\n"
        prompt += (
            f"- Property Type: {property_type}\n"
            f"- Occupancy Status: {occupancy_status}\n"
            f"- Property Condition & Repairs: {property_condition}\n\n"
            f"Generate detailed creative deal structuring strategies for this real estate deal. Provide multiple strategies, each with a clear title (e.g., 'Strategy 1: Title'), detailed description, pros, cons, and implementation steps."
        )
        with st.spinner("Developing strategies..."):
            messages = [
                {"role": "system", "content": "You are a real estate investment strategist. Provide detailed creative deal structuring options."},
                {"role": "user", "content": prompt}
            ]
            strategies = call_mistral(messages=messages)

        # Record and display
        st.session_state.deal_strategy_memory.append(("assistant", strategies))
        st.session_state.last_strategies = strategies
        st.chat_message("assistant").write(strategies)
        st.subheader("Recommended Strategies")
        st.markdown(strategies)

        # Parse strategies for dropdown
        matches = re.findall(
            r"Strategy\s+(\d+):.*?\n(.*?)(?=(?:Strategy\s+\d+:)|\Z)",
            strategies,
            flags=re.DOTALL
        )
        if matches:
            for num, text in matches:
                strategy_key = f"Strategy {num}"
                if strategy_key not in st.session_state.strategy_confidences:
                    st.session_state.strategy_confidences[strategy_key] = 7
        else:
            # Fallback if no numbered strategies found
            strategy_key = "Strategy 1"
            if strategy_key not in st.session_state.strategy_confidences:
                st.session_state.strategy_confidences[strategy_key] = 7
            matches = [("1", strategies)]

        save_interaction(conn, "deal_strategy", prompt, strategies)

    # Strategy evaluation & refinement
    strategies = st.session_state.get("last_strategies")
    if strategies:
        # Parse individual strategies
        matches = re.findall(
            r"Strategy\s+(\d+):.*?\n(.*?)(?=(?:Strategy\s+\d+:)|\Z)",
            strategies,
            flags=re.DOTALL
        )
        if matches:
            strategy_dict = {f"Strategy {num}": text.strip() for num, text in matches}
        else:
            strategy_dict = {"Strategy 1": strategies.strip()}

        labels = list(strategy_dict.keys())
        selected_label = st.selectbox("Which strategy do you prefer?", labels, key="eval_choice")
        selected_text = strategy_dict[selected_label]

        # Show the selected content
        st.markdown(f"**{selected_label}**")
        st.markdown(selected_text)

        # Confidence slider
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
            messages = [
                {"role": "system", "content": "Refine the selected strategy based on user feedback."},
                {"role": "user", "content": refinement_prompt}
            ]
            refinement = call_mistral(messages=messages)

            st.session_state.deal_strategy_memory.append(("assistant", refinement))
            st.chat_message("assistant").write(refinement)
            save_interaction(conn, "deal_strategy_refinement", selected_text, refinement)

        # Chatbot integration
        st.divider()
        st.subheader("Chat About Strategies")
        if "strategy_chat_memory" not in st.session_state:
            st.session_state.strategy_chat_memory = []

        # Display chat history
        for role, message in st.session_state.strategy_chat_memory:
            st.chat_message(role).write(message)

        # Chat input
        user_input = st.chat_input("Ask a question about the strategies...")
        if user_input:
            st.session_state.strategy_chat_memory.append(("user", user_input))
            context = f"Strategies:\n{strategies}\n\nUser Question:\n{user_input}"
            messages = [
                {"role": "system", "content": "You are a real estate strategist answering questions about deal structuring strategies."},
                {"role": "user", "content": context}
            ]
            answer = call_mistral(messages=messages)
            st.session_state.strategy_chat_memory.append(("assistant", answer))
            st.chat_message("assistant").write(answer)
            save_interaction(conn, "deal_strategy_chat", user_input, answer)


def build_guided_prompt(details: dict, detail_level: str) -> str:
    """
    Construct a detailed prompt from guided form data to generate a real estate purchase agreement.
    """
    vendor_name = details['vendor']['name']
    vendor_situation = details['vendor']['situation']
    property_condition = details['vendor']['condition']
    other_challenges = details['vendor'].get('challenges', '')
    family_member = details['vendor'].get('family_member', '')
    family_situation = details['vendor'].get('family_situation', '')

    local_knowledge = details['local_knowledge']['experience']
    comparables = details['local_knowledge']['comparables']
    avg_sale_price = details['local_knowledge']['avg_sale_price']
    required_profit = details['local_knowledge']['required_profit']
    profit_reason = details['local_knowledge']['profit_reason']

    costs = details['costs']
    total_cost = sum(costs.values())

    negative_points = details.get('negative_points', '')
    offers = details['offers']

    social_proof = details.get('social_proof', '')
    proof_of_funds = details.get('proof_of_funds', '')

    comparables_str = "\n".join(
        [f"- {comp['address']}: {comp['status']} at £{comp['price']}" for comp in comparables]
    ) if comparables else "No comparables provided."

    costs_str = "\n".join(
        [f"- {key}: £{value}" for key, value in costs.items()]
    ) + f"\n- Total: £{total_cost}"

    offers_str = "\n".join(
        [f"- Offer {i+1}: £{offer}" for i, offer in enumerate(offers)]
    ) if offers else "No offers provided."

    sections = [
        "Generate a professional real estate purchase agreement with the following details:",
        f"- Vendor Name: {vendor_name}",
        f"- Vendor Situation: {vendor_situation}",
        f"- Property Condition: {property_condition}",
        f"- Other Challenges: {other_challenges}" if other_challenges else "",
        f"- Family Member & Situation: {family_member} - {family_situation}" if family_member and family_situation else "",
        f"- Local Knowledge: {local_knowledge}",
        f"- Comparables:\n{comparables_str}",
        f"- Average Sale Price: £{avg_sale_price}",
        f"- Required Profit (15%): £{required_profit}",
        f"- Reason for Profit: {profit_reason}",
        f"- Detailed Costs:\n{costs_str}",
        f"- Potential Negative Points: {negative_points}" if negative_points else "",
        f"- Proposed Offers:\n{offers_str}",
        f"- Social Proof: {social_proof}" if social_proof else "",
        f"- Proof of Funds: {proof_of_funds}" if proof_of_funds else "",
        f"Level of Detail: {detail_level}.",
        "Format the output as an empathetic offer letter, starting with an acknowledgment of the vendor's situation, expressing willingness to help, and thanking them for their time. Highlight the property and locality positively. Include headings and bold key points as per the provided example. Include incentives like covering legal costs, no estate agent fees, and benefits of a Purchase Lease Option and Exchange with Delayed Completion. Suggest next steps, such as preparing a Heads of Terms document."
    ]
    return "\n".join([s for s in sections if s])

def build_guided_prompt(details: dict, detail_level: str) -> str:
    """
    Construct a detailed prompt from guided form data to generate a real estate purchase agreement.
    """
    vendor_name = details['vendor']['name']
    vendor_situation = details['vendor']['situation']
    property_condition = details['vendor']['condition']
    other_challenges = details['vendor'].get('challenges', '')
    family_member = details['vendor'].get('family_member', '')
    family_situation = details['vendor'].get('family_situation', '')

    local_knowledge = details['local_knowledge']['experience']
    comparables = details['local_knowledge']['comparables']
    avg_sale_price = details['local_knowledge']['avg_sale_price']
    required_profit = details['local_knowledge']['required_profit']
    profit_reason = details['local_knowledge']['profit_reason']

    costs = details['costs']
    total_cost = sum(costs.values())

    negative_points = details.get('negative_points', '')
    offers = details['offers']

    social_proof = details.get('social_proof', '')
    proof_of_funds = details.get('proof_of_funds', '')

    comparables_str = "\n".join(
        [f"- {comp['address']}: {comp['status']} at £{comp['price']}" for comp in comparables]
    ) if comparables else "No comparables provided."

    costs_str = "\n".join(
        [f"- {key}: £{value}" for key, value in costs.items()]
    ) + f"\n- Total: £{total_cost}"

    offers_str = "\n".join(
        [f"- Offer {i+1}: £{offer['amount']} ({offer['description']})" for i, offer in enumerate(offers)]
    ) if offers else "No offers provided."

    sections = [
        "Generate a professional real estate purchase agreement with the following details:",
        f"- Vendor Name: {vendor_name}",
        f"- Vendor Situation: {vendor_situation}",
        f"- Property Condition: {property_condition}",
        f"- Other Challenges: {other_challenges}" if other_challenges else "",
        f"- Family Member & Situation: {family_member} - {family_situation}" if family_member and family_situation else "",
        f"- Local Knowledge: {local_knowledge}",
        f"- Comparables:\n{comparables_str}",
        f"- Average Sale Price: £{avg_sale_price}",
        f"- Required Profit (15%): £{required_profit}",
        f"- Reason for Profit: {profit_reason}",
        f"- Detailed Costs:\n{costs_str}",
        f"- Potential Negative Points: {negative_points}" if negative_points else "",
        f"- Proposed Offers:\n{offers_str}",
        f"- Social Proof: {social_proof}" if social_proof else "",
        f"- Proof of Funds: {proof_of_funds}" if proof_of_funds else "",
        f"Level of Detail: {detail_level}.",
        "Format the output as an empathetic offer letter, starting with an acknowledgment of the vendor's situation, expressing willingness to help, and thanking them for their time. Highlight the property and locality positively. Include headings and bold key points as per the provided example. Include incentives like covering legal costs, no estate agent fees, and benefits of a Purchase Lease Option and Exchange with Delayed Completion. Suggest next steps, such as preparing a Heads of Terms document."
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
                "AI Model Preference", ["in-depth"], horizontal=True, key="offer_ai_model"
            )
            creativity = st.slider("Creativity Level", 0.0, 1.0, 0.3, key="offer_creativity")
            detail_level = st.select_slider(
                "Detail Level", options=["Minimal", "Standard", "Comprehensive"],
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
                st.markdown('#### Vendor Details')
                vendor_name = st.text_input("Vendor Name*", key="vendor_name")
                vendor_situation = st.text_area("Vendor Situation*", placeholder="e.g., relocating, financial distress", key="vendor_situation")
                property_condition = st.text_area("Property Condition*", placeholder="e.g., needs roof replacement", key="property_condition")
                other_challenges = st.text_area("Other Challenges", placeholder="e.g., planning issues, structural concerns", key="other_challenges")
                family_member = st.text_input("Family Member Name", key="family_member")
                family_situation = st.text_area("Family Member Situation", placeholder="e.g., elderly parent needing care", key="family_situation")

                st.markdown('#### Local Knowledge & Comparables')
                local_experience = st.text_area("Local Investment Experience*", placeholder="e.g., I have been investing in [Local Area] for several years, successfully completing [X] deals in the last [Y] months.", key="local_experience")
                st.markdown("**Recent Comparables**")
                comparables = []
                for i in range(3):
                    st.markdown(f"**Comparable {i+1}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        address = st.text_input(f"Property Address {i+1}*", key=f"comp_address_{i}")
                    with col2:
                        status = st.selectbox(f"Status {i+1}*", ["Sold", "Under Offer", "SSTC"], key=f"comp_status_{i}")
                    with col3:
                        price = st.number_input(f"Price {i+1}* (£)", min_value=0.0, step=1000.0, key=f"comp_price_{i}")
                    comparables.append({"address": address, "status": status, "price": price})
                avg_sale_price = st.number_input("Average Sale Price* (£)", min_value=0.0, step=1000.0, key="avg_sale_price")
                required_profit = st.number_input("Required Profit (15%)* (£)", min_value=0.0, step=1000.0, key="required_profit")
                profit_reason = st.text_area("Reason for Profit*", value="I maintain at least a 15% profit margin to protect against unforeseen risks (e.g., market shifts, extra repairs). This margin ensures I can confidently close on the property without complications.", key="profit_reason")

                st.markdown('#### Detailed Costs')
                costs = {
                    "Deposit": st.number_input("Deposit (£)", min_value=0.0, step=1000.0, key="cost_deposit"),
                    "Finance": st.number_input("Finance (£)", min_value=0.0, step=1000.0, key="cost_finance"),
                    "Solicitors": st.number_input("Solicitors (£)", min_value=0.0, step=500.0, key="cost_solicitors"),
                    "Stamp Duty": st.number_input("Stamp Duty (£)", min_value=0.0, step=500.0, key="cost_stamp_duty"),
                    "Estate Agent Fees": st.number_input("Estate Agent Fees (£)", min_value=0.0, step=500.0, key="cost_estate_agent"),
                    "Doors and Windows": st.number_input("Doors and Windows (£)", min_value=0.0, step=500.0, key="cost_doors_windows"),
                    "Joists and Floorboards": st.number_input("Joists and Floorboards (£)", min_value=0.0, step=500.0, key="cost_joists"),
                    "Chimney Breast": st.number_input("Chimney Breast (£)", min_value=0.0, step=500.0, key="cost_chimney"),
                    "Lintels & Structural Work": st.number_input("Lintels & Structural Work (£)", min_value=0.0, step=500.0, key="cost_lintels"),
                    "Roof, Guttering & Fascias": st.number_input("Roof, Guttering & Fascias (£)", min_value=0.0, step=500.0, key="cost_roof"),
                    "External Wall Render": st.number_input("External Wall Render (£)", min_value=0.0, step=500.0, key="cost_render"),
                    "Electrical Wiring & Lighting": st.number_input("Electrical Wiring & Lighting (£)", min_value=0.0, step=500.0, key="cost_electrical"),
                    "Wi-Fi Points": st.number_input("Wi-Fi Points (£)", min_value=0.0, step=100.0, key="cost_wifi"),
                    "Fire Safety Panels": st.number_input("Fire Safety Panels (£)", min_value=0.0, step=500.0, key="cost_fire_safety"),
                    "Studs / Plaster Boards / Insulation": st.number_input("Studs / Plaster Boards / Insulation (£)", min_value=0.0, step=500.0, key="cost_studs"),
                    "Emergency Lights": st.number_input("Emergency Lights (£)", min_value=0.0, step=100.0, key="cost_emergency_lights"),
                    "Cupboards & Skirting Boards": st.number_input("Cupboards & Skirting Boards (£)", min_value=0.0, step=500.0, key="cost_cupboards"),
                    "Window Sills": st.number_input("Window Sills (£)", min_value=0.0, step=100.0, key="cost_window_sills"),
                    "Locks & Ironmongery": st.number_input("Locks & Ironmongery (£)", min_value=0.0, step=100.0, key="cost_locks"),
                    "Plumbing & Heating": st.number_input("Plumbing & Heating (£)", min_value=0.0, step=500.0, key="cost_plumbing"),
                    "Kitchen": st.number_input("Kitchen (£)", min_value=0.0, step=500.0, key="cost_kitchen"),
                    "Bathroom": st.number_input("Bathroom (£)", min_value=0.0, step=500.0, key="cost_bathroom"),
                    "Tiling": st.number_input("Tiling (£)", min_value=0.0, step=500.0, key="cost_tiling"),
                    "Painting & Decoration": st.number_input("Painting & Decoration (£)", min_value=0.0, step=500.0, key="cost_painting"),
                    "Flooring / Carpet": st.number_input("Flooring / Carpet (£)", min_value=0.0, step=500.0, key="cost_flooring"),
                    "Garden & Landscaping": st.number_input("Garden & Landscaping (£)", min_value=0.0, step=500.0, key="cost_garden")
                }

                st.markdown('#### Potential Negative Points')
                negative_points = st.text_area("Potential Negative Points", placeholder="e.g., market risks, structural issues", key="negative_points")

                st.markdown('#### Proposed Offers')
                offers = []
                for i in range(3):
                    st.markdown(f"**Offer {i+1}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        amount = st.number_input(f"Offer {i+1} Amount (£)*", min_value=0.0, step=1000.0, key=f"offer_amount_{i}")
                    with col2:
                        description = st.text_input(f"Offer {i+1} Description*", placeholder="e.g., Cash offer with quick close", key=f"offer_desc_{i}")
                    offers.append({"amount": amount, "description": description})

                st.markdown('#### Social Proof & Proof of Funds')
                social_proof = st.text_input("Google Reviews Link", key="social_proof")
                proof_of_funds = st.text_area("Proof of Funds", placeholder="e.g., Bank statement summary", key="proof_of_funds")

                st.markdown('</div>', unsafe_allow_html=True)

                submitted = st.form_submit_button("Generate Offer Draft")
                if submitted:
                    missing = []
                    for field, msg in {
                        "vendor_name": "Vendor name required",
                        "vendor_situation": "Vendor situation required",
                        "property_condition": "Property condition required",
                        "local_experience": "Local experience required",
                        "avg_sale_price": "Average sale price required",
                        "required_profit": "Required profit required",
                        "profit_reason": "Profit reason required"
                    }.items():
                        if not st.session_state.get(field):
                            missing.append(msg)
                    for i in range(3):
                        if not st.session_state.get(f"comp_address_{i}") or not st.session_state.get(f"comp_status_{i}") or not st.session_state.get(f"comp_price_{i}"):
                            missing.append(f"Comparable {i+1} details required")
                        if not st.session_state.get(f"offer_amount_{i}") or not st.session_state.get(f"offer_desc_{i}"):
                            missing.append(f"Offer {i+1} amount and description required")
                    if missing:
                        for m in missing:
                            st.error(m)
                    else:
                        st.session_state.offer_data['details'] = {
                            'vendor': {
                                'name': vendor_name,
                                'situation': vendor_situation,
                                'condition': property_condition,
                                'challenges': other_challenges,
                                'family_member': family_member,
                                'family_situation': family_situation
                            },
                            'local_knowledge': {
                                'experience': local_experience,
                                'comparables': comparables,
                                'avg_sale_price': avg_sale_price,
                                'required_profit': required_profit,
                                'profit_reason': profit_reason
                            },
                            'costs': costs,
                            'negative_points': negative_points,
                            'offers': offers,
                            'social_proof': social_proof,
                            'proof_of_funds': proof_of_funds
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
            uploaded = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"], key="offer_upload")
            if uploaded and st.button("Analyze & Improve Upload", key="btn_upload_analyze"):
                if uploaded.type == "application/pdf":
                    reader = PdfReader(uploaded)
                    doc_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                elif uploaded.type == "text/plain":
                    doc_text = uploaded.read().decode("utf-8")
                else:
                    doc = Document(uploaded)
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
                doc = Document()
                for line in content.split('\n'):
                    if line.startswith('**') and line.endswith('**'):
                        p = doc.add_paragraph()
                        run = p.add_run(line[2:-2])
                        run.bold = True
                    else:
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
    """Enhanced admin portal with usage analytics, subscription management, and prompt viewer"""
    st.header("🔒 Admin Portal")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["User Management", "Subscription Management", "Content Management", "Usage Analytics", "Prompt Viewer"])

    with tab1:
        st.subheader("User Accounts")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        select_columns = ["username", "role"]
        if "last_login" in columns:
            select_columns.append("last_login")
        if "location_id" in columns:
            select_columns.append("location_id")
        if "created_at" in columns:
            select_columns.append("created_at")

        query = f"SELECT {', '.join(select_columns)} FROM users"
        users = conn.execute(query).fetchall()

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
        st.subheader("Feature Access Control")
        users = conn.execute("SELECT username FROM users").fetchall()
        if not users:
            st.warning("No users found")
        else:
            selected_user = st.selectbox("Select User", [u[0] for u in users])

            sub = conn.execute(
                "SELECT lease_analysis, deal_structuring, offer_generator FROM subscriptions WHERE username = ?",
                (selected_user,)
            ).fetchone()

            if not sub:
                conn.execute(
                    "INSERT INTO subscriptions (username) VALUES (?)",
                    (selected_user,)
                )
                conn.commit()
                sub = (0, 0, 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                lease_access = st.toggle("Lease Analysis", value=bool(sub[0]))
            with col2:
                deal_access = st.toggle("Deal Structuring", value=bool(sub[1]))
            with col3:
                offer_access = st.toggle("Offer Generator", value=bool(sub[2]))

            if st.button("Update Access"):
                conn.execute(
                    """UPDATE subscriptions
                    SET lease_analysis = ?, deal_structuring = ?, offer_generator = ?
                    WHERE username = ?""",
                    (int(lease_access), int(deal_access), int(offer_access), selected_user)
                )
                conn.commit()
                st.success("Access updated successfully!")

    with tab3:
        st.subheader("Training Content")
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

                meta_path = os.path.join(save_dir, f"{uploaded.name}.meta")
                with open(meta_path, "w") as f:
                    json.dump({
                        "uploaded_by": st.session_state.username,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "description": description,
                        "type": file_type.lower()
                    }, f)

                st.success(f"{file_type} uploaded successfully!")

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

    with tab4:
        st.subheader("Usage Analytics")
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

    with tab5:
        st.subheader("Prompt Viewer")
        users = conn.execute("SELECT username FROM users").fetchall()
        if not users:
            st.warning("No users found")
        else:
            selected_user = st.selectbox("Select User to View Prompts", [u[0] for u in users], key="prompt_user_select")
            interactions = conn.execute(
                "SELECT timestamp, feature, input_text, output_text "
                "FROM interactions WHERE username = ? ORDER BY timestamp DESC",
                (selected_user,)
            ).fetchall()
            if not interactions:
                st.info(f"No interactions found for {selected_user}")
            else:
                for ts, feature, input_text, output_text in interactions:
                    with st.expander(f"{ts} • {feature}"):
                        st.markdown("**Prompt/Input**")
                        st.text_area("Input", input_text, height=200, key=f"prompt_input_{ts}_{feature}")
                        st.markdown("**Response**")
                        st.text_area("Output", output_text, height=200, key=f"prompt_output_{ts}_{feature}")
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

# Registration UI
def register_ui(conn):
    st.title("📝 Register")
    st.markdown("Create a new account.")

    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        location_id = st.text_input("Location ID (optional)", placeholder="Enter location ID if applicable")
        submitted = st.form_submit_button("Register")

        if submitted:
            # Input validation
            if not username or not password or not confirm_password:
                st.error("All required fields must be filled.")
                return
            if len(password) < 8:
                st.error("Password must be at least 8 characters long.")
                return
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
            if not re.match(r"^[a-zA-Z0-9_]+$", username):
                st.error("Username can only contain letters, numbers, and underscores.")
                return

            # Check for existing username
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                st.error("Username already exists. Please choose a different username.")
                return

            # Register user
            try:
                hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
                cursor.execute(
                    "INSERT INTO users (username, password, role, location_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, hashed, "user", location_id or None, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                cursor.execute(
                    "INSERT INTO subscriptions (username, lease_analysis, deal_structuring, offer_generator) VALUES (?, ?, ?, ?)",
                    (username, 0, 0, 0)
                )
                conn.commit()
                st.success("Registration successful! You can now log in.")
                logging.info(f"User {username} registered successfully")
            except sqlite3.Error as e:
                st.error(f"Registration failed: {e}")
                logging.error(f"Registration failed for {username}: {e}")

def login_ui(conn):
    """Handle user login and registration with tabbed interface"""
    st.title("🔐 Property Deals AI")
    st.markdown("Access or create your account to use the platform.")

    # Create tabs for Login and Register
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown("### Log In")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if not username or not password:
                    st.error("Please provide both username and password.")
                    return

                cursor = conn.cursor()
                cursor.execute(
                    "SELECT password, role, location_id FROM users WHERE username = ?",
                    (username,)
                )
                user = cursor.fetchone()

                if user and verify_password(user[0], password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = user[1]
                    st.session_state.location_id = user[2] if user[2] else None

                    # Update last login timestamp
                    cursor.execute(
                        "UPDATE users SET last_login = ? WHERE username = ?",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username)
                    )
                    conn.commit()

                    st.success(f"Welcome, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with register_tab:
        register_ui(conn)

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
        """
        <style>
        /* Container around each input section */
        .auction-section {
          background-color: #f0f2f6;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 20px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        /* Section headings */
        .auction-section h2,
        .auction-section h3 {
          color: #2E86AB;
          margin-bottom: 12px;
          font-weight: 600;
        }

        /* Labels and select boxes */
        .auction-section .stSelectbox, 
        .auction-section .stTextArea, 
        .auction-section .stTextInput,
        .auction-section .stNumberInput {
          background-color: #ffffff !important;
          border: 1px solid #d1d5db !important;
          border-radius: 6px !important;
          padding: 8px !important;
        }

        /* Primary buttons */
        .auction-button, 
        .stButton>button {
          background-color: #F18F01 !important;
          color: white !important;
          border-radius: 8px !important;
          padding: 8px 16px !important;
          font-weight: 500 !important;
        }
        .stButton>button:hover {
          background-color: #d67a00 !important;
        }

        /* Slider styling */
        .auction-section .stSlider > div:nth-child(1) {
          color: #2E86AB;
        }

        /* Chat history boxes */
        .st-chat-message {
          border-radius: 8px !important;
          padding: 12px !important;
          margin-bottom: 8px !important;
        }
        .st-chat-message.user {
          background-color: #e8f1fb !important;
        }
        .st-chat-message.assistant {
          background-color: #f7f7f7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Initialize database
    try:
        conn = init_db(DB_PATH)
        conn.close()  # Close initial connection after initialization
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return

    # Ensure login state
    if "logged_in" not in st.session_state:
        st.session_state.update({
            "logged_in": False,
            "username": None,
            "role": None,
            "subscription": {
                "lease_analysis": False,
                "deal_structuring": False,
                "offer_generator": False
            }
        })

    # Authentication flow
    if not st.session_state.logged_in:
        with get_db_connection(DB_PATH) as conn:
            login_ui(conn)
        return

    # After login check, get user's subscription status
    if st.session_state.logged_in:
        try:
            with get_db_connection(DB_PATH) as conn:
                sub = conn.execute(
                    "SELECT lease_analysis, deal_structuring, offer_generator FROM subscriptions WHERE username = ?",
                    (st.session_state.username,)
                ).fetchone()

                if not sub and st.session_state.role != "admin":
                    conn.execute(
                        "INSERT INTO subscriptions (username) VALUES (?)",
                        (st.session_state.username,)
                    )
                    conn.commit()
                    sub = (0, 0, 0)
                elif st.session_state.role == "admin":
                    sub = (1, 1, 1)

                st.session_state.subscription = {
                    "lease_analysis": bool(sub[0]),
                    "deal_structuring": bool(sub[1]),
                    "offer_generator": bool(sub[2])
                }
        except sqlite3.Error as e:
            st.error(f"Error fetching subscription status: {e}")
            return

    # Sidebar navigation - only show accessible features
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    st.sidebar.markdown(f"**Location ID:** {st.session_state.get('location_id', 'Not specified')}")

    features = []
    if st.session_state.subscription.get("lease_analysis") or st.session_state.role == "admin":
        features.append("LeaseBrief Buddy")
    if st.session_state.subscription.get("deal_structuring") or st.session_state.role == "admin":
        features.append("Auction Buddy")
    if st.session_state.subscription.get("offer_generator") or st.session_state.role == "admin":
        features.append("Offer Buddy")
    features.append("History")
    if st.session_state.role == "admin":
        features.insert(-1, "Admin Portal")

    selected = st.sidebar.radio("Navigation", features)

    # Main content
    try:
        with get_db_connection(DB_PATH) as conn:
            if selected == "LeaseBrief Buddy":
                lease_summarization_ui(conn)
            elif selected == "Auction Buddy":
                deal_structuring_ui(conn)
            elif selected == "Offer Buddy":
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
    try:
        with get_db_connection(DB_PATH) as conn:
            chatbot_ui(conn)
    except Exception as e:
        st.error(f"Error in chatbot: {e}")

    # Logout
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()