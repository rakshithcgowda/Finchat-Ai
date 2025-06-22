import os
import io
import re
from datetime import datetime
import bcrypt
import sqlite3
import logging
from contextlib import contextmanager

# Configuration
DB_PATH = "users.db"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def init_db(db_path: str = DB_PATH):
    db_exists = os.path.exists(db_path)
    with get_db_connection(db_path) as conn:
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

def create_default_admin(conn):
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

def extract_text_with_ocr(uploaded_file=None, file_type: str = "pdf", url: str = None):
    # Placeholder: Implement OCR logic here (requires OCR.space API key and dependencies like requests)
    return ["Sample extracted text"]  # Replace with actual OCR implementation

def summarize_lease(pages):
    # Placeholder: Implement lease summarization logic
    text = "\n".join(pages)
    summary = f"Summary of lease with {len(pages)} pages: {text[:500]}..."
    return summary

def generate_deal_strategies(data):
    # Placeholder: Implement deal structuring logic
    prompt = (
        f"Buyer Situation: {data.get('buyer_deposit', '')}\n"
        f"Property Details: Market Price £{data.get('market_price', 0)}\n"
        f"Seller Motivation: {data.get('seller_motivation', '')}\n"
        f"Generate strategies..."
    )
    return f"Strategy 1: {prompt}"

def generate_offer(data):
    # Placeholder: Implement offer generation logic
    prompt = build_guided_prompt(data, data.get('detail_level', 'Standard'))
    return f"Offer for {data.get('vendor', {}).get('name', '')}: {prompt[:500]}..."

def build_guided_prompt(details: dict, detail_level: str) -> str:
    vendor_name = details['vendor']['name']
    return f"Generate offer for {vendor_name} with detail level {detail_level}"

def save_interaction(conn, feature: str, input_text: str, output_text: str):
    if conn:
        conn.execute(
            "INSERT INTO interactions (username, feature, input_text, output_text) VALUES (?, ?, ?, ?)",
            (None, feature, input_text, output_text),
        )
        conn.commit()