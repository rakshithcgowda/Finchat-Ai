from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import sqlite3
from datetime import datetime
import bcrypt
import logging
import re
from app import extract_text_with_ocr, lease_summarization_ui, deal_structuring_ui, offer_generator_ui, init_db, get_db_connection, build_guided_prompt, save_interaction
from fastapi import FastAPI, HTTPException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Added localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "users.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def verify_password(hashed: bytes, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

@app.on_event("startup")
async def startup_event():
    init_db(DB_PATH)

@app.post("/api/lease-summarization")
async def process_lease(file: UploadFile = File(...)):
    try:
        content = await file.read()
        with io.BytesIO(content) as f:
            pages = extract_text_with_ocr(uploaded_file=f, file_type=file.filename.split('.')[-1])
            if not pages:
                raise HTTPException(status_code=400, detail="No readable text extracted")
            with get_db_connection(DB_PATH) as conn:
                summary = lease_summarization_ui(pages)
                save_interaction(conn, "lease_summary", str(pages), summary)
            return {"summary": summary}
    except Exception as e:
        logging.error(f"Lease summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/lease-summarization")
# async def list_lease_summaries():
#     try:
#         with get_db_connection(DB_PATH) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT
#                     id,
#                     timestamp,                  -- use your actual timestamp column name
#                     response_data   AS summary
#                   FROM interactions
#                  WHERE type = 'lease_summary'
#               ORDER BY timestamp DESC
#             """)
#             rows = cursor.fetchall()

#         return {
#             "summaries": [
#                 {"id": r[0], "timestamp": r[1], "summary": r[2]}
#                 for r in rows
#             ]
#         }
#     except Exception as e:
#         logging.error(f"Error fetching lease summaries: {e}")
#         raise HTTPException(status_code=500, detail="Could not fetch summaries")
@app.get("/api/lease-summarization/{summary_id}")
async def get_lease_summary(summary_id: int):
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    id,
                    timestamp,                  -- use your actual timestamp column name
                    response_data   AS summary
                  FROM interactions
                 WHERE type = 'lease_summary' AND id = ?
            """, (summary_id,))
            row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Summary not found")

        return {"id": row[0], "timestamp": row[1], "summary": row[2]}
    except Exception as e:
        logging.error(f"Error fetching lease summary: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch summary")

@app.post("/api/deal-structuring")
async def process_deal(data: dict):
    try:
        with get_db_connection(DB_PATH) as conn:
            strategies = deal_structuring_ui(data)
            save_interaction(conn, "deal_strategy", str(data), strategies)
        return {"strategies": strategies}
    except Exception as e:
        logging.error(f"Deal structuring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/offer-generator")
async def generate_offer_endpoint(data: dict):
    try:
        with get_db_connection(DB_PATH) as conn:
            offer = offer_generator_ui(data)
            save_interaction(conn, "offer_generator", str(data), offer)
        return {"offer": offer}
    except Exception as e:
        logging.error(f"Offer generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register_user(data: dict):
    try:
        username = data.get("username")
        password = data.get("password")
        location_id = data.get("location_id", None)

        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, and underscores")

        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")

            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password, role, location_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, hashed, "user", location_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            cursor.execute(
                "INSERT INTO subscriptions (username, lease_analysis, deal_structuring, offer_generator) VALUES (?, ?, ?, ?)",
                (username, 0, 0, 0)
            )
            conn.commit()
        return {"message": "Registration successful"}
    except sqlite3.Error as e:
        logging.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login_user(data: dict):
    try:
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password are required")

        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password, role, location_id FROM users WHERE username = ?",
                (username,)
            )
            user = cursor.fetchone()

            if not user or not verify_password(user[0], password):
                raise HTTPException(status_code=401, detail="Invalid username or password")

            cursor.execute(
                "UPDATE users SET last_login = ? WHERE username = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username)
            )
            conn.commit()
        return {"message": "Login successful", "role": user[1], "location_id": user[2]}
    except sqlite3.Error as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))