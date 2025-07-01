**Property Deals AI**

A Streamlit-based web application that provides AI-powered tools for real estate professionals, including document summarization, deal structuring, offer generation, and auction document analysis. It integrates OCR, Google Gemini, Mistral, DeepSeek, and OpenAI APIs.

---

## Features

* **LeaseBrief Buddy**: Upload or URL-point a lease agreement (PDF/JPG) and get a concise AI-generated summary.
* **Property Guru**: Input buyer & property details to receive three strategic deal-structuring recommendations.
* **Offer Buddy**: Draft, review, edit, and export empathetic offer letters in PDF/Word/Text/HTML formats.
* **Auction Buddy**: Upload multiple documents to get individual and merged AI summaries and chat with the content.
* **History**: View past interactions by feature with full input-output logs.
* **Admin Portal**: Manage users, subscriptions, training content, usage analytics, and system prompts.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/property-deals-ai.git
   cd property-deals-ai
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   > Alternatively, if you don't have a `requirements.txt`, install manually:

   ```bash
   pip install fpdf pdf2image pytesseract deepseek mistralai pdfplumber pylovepdf \
               docx2txt PyGithub python-docx mistral streamlit bcrypt PyPDF2 streamlit-javascript \
               requests supabase psycopg2 google-generativeai openai torch torchvision pyngrok
   ```

---

## Configuration

1. **Environment variables**:

   * `GOOGLE_API_KEY`: API key for Google Generative AI (Gemini).
   * `MISTRAL_API_KEY`: API key for Mistral.
   * `DEEPSEEK_API_KEY`: API key for DeepSeek.
   * `OCR_API_KEY`: API key for OCR.space.
   * `DATABASE_URL` or individual DB settings (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_PORT`).
   * `NGROK_AUTH_TOKEN`: Token for ngrok tunneling.

2. **.env file** (optional):

   ```ini
   GOOGLE_API_KEY=...
   MISTRAL_API_KEY=...
   DEEPSEEK_API_KEY=...
   OCR_API_KEY=...
   DATABASE_URL=postgresql://user:pass@host:port/dbname
   NGROK_AUTH_TOKEN=...
   ```

---

## Usage

1. **Initialize the database** (tables will be created automatically on first run).

2. **Run the app**:

   ```bash
   streamlit run app.py
   ```

3. **Expose via ngrok** (optional):

   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
   public_url = ngrok.connect(addr="8501", proto="http")
   print("App live at:", public_url)
   ```

4. **Access** the web UI at `http://localhost:8501` or the ngrok URL.

---

## Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── static_data/         # (Optional) Static files, PDFs, templates
├── training_content/    # Uploaded training materials
└── ...                  # Other modules & assets
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
