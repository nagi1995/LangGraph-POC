# LangGraph-POC


This repo includes several use cases using [LangGraph](https://www.langchain.com/langgraph), notably a custom project: **09_Smart_Email_Assistant** – an AI-powered email assistant capable of understanding, classifying, summarizing, and responding to emails.

---

## 🔍 Project Highlight: Smart Email Assistant

The **Smart Email Assistant** intelligently processes email data using a structured LangGraph pipeline. It applies natural language processing to understand the content of emails and can:

- ✅ Classify emails by intent: Primary, Promotions, Social, or Updates
- 🧠 Extract structured fields based on email type
- ✉️ Summarize email contents in a concise manner
- 🚫 Detect spam and phishing attempts
- 🤖 Generate a contextual reply (if needed)
- 📅 Extract events and calendar information
- 🔄 Accept user feedback to iteratively improve email drafts

---

## ⚙️ How It Works

The assistant uses the following LangGraph nodes in sequence:

1. **Intent Classification** - Categorizes the email content.
2. **Field Extraction** - Pulls structured data using Pydantic schemas.
3. **Summarization** - Creates a brief summary for quick review.
4. **Spam Detection** - Flags potential spam based on known patterns.
5. **Reply Generation** - Composes replies where applicable.
6. **Event Extraction** - Looks for meeting or deadline-related content.
7. **Human Review Loop** - Incorporates feedback before finalizing reply.

Each node is a function that updates the shared email state.

---
## 🛠️ Requirements

Make sure you have the following installed:

- Python 3.12.9
- `.env` file with your API keys (e.g., GROQ API key)

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🚀 Running the Assistant

In `09_Smart_Email_Assistant/assistant.py`, the last section shows an example of how to run the assistant using a sample email.

You can replace `sample_email` with your own data and call:

```python
response = app.invoke({"email": your_email_instance})
```
---

## ✍️ Feedback Loop

The assistant supports a feedback loop via the terminal:

* After generating a reply, you'll be prompted for feedback.
* If you type `"approved"`, it proceeds to extract events.
* If you provide custom feedback, it rewrites the reply accordingly.

---


## 📌 Note

This repository is built upon [harishneel1/langgraph](https://github.com/harishneel1/langgraph).

