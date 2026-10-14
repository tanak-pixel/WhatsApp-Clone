# 💬 WhatsApp Style-Cloning Bot (iOS + Gemini API + Streamlit)

Clone your exact texting personality from an iPhone WhatsApp chat export (`.txt` or `.zip`). This project parses your message history, extracts a quantitative **Style Fingerprint** (capitalization habits, missing periods, double-texting rhythm, favorite emojis, and slang), dynamically selects context-matched **few-shot conversation turns**, and runs a local [Streamlit](https://docs.streamlit.io/develop/api-reference/chat) web interface powered by the official [Google GenAI Python SDK](https://googleapis.github.io/python-genai/).

## 🔒 Security & Privacy By Design

Both your **Gemini API key** and your **personal WhatsApp chat exports** are protected from ever being committed to GitHub:

* **Zero Hardcoded Keys:** `whatsapp_clone_bot.py` loads `GEMINI_API_KEY` from a local `.env` file (via [`python-dotenv`](https://pypi.org/project/python-dotenv/) or a built-in fallback parser), environment variables, or [`.streamlit/secrets.toml`](https://docs.streamlit.io/develop/concepts/connections/secrets-management).
* **Strict `.gitignore` Rules:** The included [`.gitignore`](https://docs.github.com/articles/ignoring-files) automatically blocks `.env`, `.streamlit/secrets.toml`, `_chat.txt`, `*WhatsApp*.zip`, and `*.jsonl` fine-tuning exports so neither your credentials nor your private messages are ever tracked by Git.

## 📁 Repository Structure

| File | Purpose |
| :--- | :--- |
| `whatsapp_clone_bot.py` | Main Streamlit web application, iOS WhatsApp `.txt`/`.zip` parser, style fingerprint analyzer, dynamic few-shot prompt builder, and Gemini API client |
| `test_whatsapp_clone_bot.py` | Automated unit test suite verifying iOS chat parsing, `.zip` extraction, `.env` loading, key masking, and zero hardcoded secrets |
| `requirements.txt` | Python dependencies (`google-genai`, `streamlit`, `python-dotenv`) |
| `.env.example` | Safe template for configuring `GEMINI_API_KEY` locally |
| `.gitignore` | Blocks `.env`, `.streamlit/secrets.toml`, and WhatsApp chat files (`_chat.txt`, `*.zip`, `*.jsonl`) from Git commits |

## 🚀 Quickstart (Run Locally)

### 1. Clone the Repo & Create a Virtual Environment

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Your Gemini API Key Securely

Copy the template `.env.example` file to `.env` (which is ignored by Git) and paste your key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/get-started):

```bash
cp .env.example .env
```

Open `.env` in your editor and set your key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Export Your iOS WhatsApp Chat

Following the [WhatsApp Help Center Export Guide](https://faq.whatsapp.com/1180414079177245):

1. Open a 1-on-1 or group chat in **WhatsApp on iPhone**.
2. Tap the **contact or group name** at the top of the screen.
3. Scroll down and tap **Export Chat** → **Without Media**.
4. Save the exported `WhatsApp Chat - <Name>.zip` (or `_chat.txt`) to your computer and upload it in the web UI sidebar.

### 4. Launch the Local Web UI

```bash
streamlit run whatsapp_clone_bot.py
```

*(Want to test before transferring your own chat? Click **Load Built-in Sample iOS Chat** in the left sidebar.)*

## 🧪 Running the Automated Tests

Verify that the parser, style analyzer, and secret protections work offline:

```bash
python3 test_whatsapp_clone_bot.py
# Or run the built-in self-test:
python3 whatsapp_clone_bot.py --self-test
```

## 📤 How to Push This Project to GitHub Safely

Run these commands inside the repository folder. Notice how `git status` will **never** stage `.env` or `_chat.txt`:

```bash
git init
git add .
git status  # Verify that .env and any _chat.txt files are NOT listed
git commit -m "Initial commit: WhatsApp style-cloning bot with Gemini API and Streamlit UI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
