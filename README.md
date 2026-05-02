# 🤖 KushVision AI — Multi-Modal AI Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange)
![AWS](https://img.shields.io/badge/AWS-EC2-yellow)

## 📌 Overview

KushVision AI is a full-stack AI-powered assistant that brings multiple intelligent capabilities into one platform. It combines LLMs, RAG-based document Q&A, real-time web search, AI image generation, and voice interaction — all accessible via a clean web interface deployed on AWS EC2.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 💬 AI Chat | Intelligent conversations powered by Groq LLaMA 3 |
| 🔍 Real-Time Search | Get live web information directly within chat |
| 📄 Document Q&A | Upload PDFs, DOCX, TXT and ask questions via RAG |
| 🎨 Image Generation | Generate AI images from text prompts using FLUX.1-schnell |
| 🎙️ Voice I/O | Talk to the assistant and hear it respond back |

---

## 🏗️ Architecture
User Input (Text / Voice)
↓
Frontend (HTML + CSS + Web Speech API)
↓
Flask Backend (AWS EC2)
↓
┌─────────────────────────────────┐
│  LLM    → Groq LLaMA 3         │
│  RAG    → FAISS + Sentence Trans│
│  Search → SerpAPI               │
│  Image  → FLUX.1-schnell (HF)   │
└─────────────────────────────────┘
↓
Response displayed / spoken to User

---

## 🧠 AI Components

| Component | Technology |
|-----------|-----------|
| LLM | Groq LLaMA 3 |
| RAG Pipeline | LangChain + FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| Image Generation | FLUX.1-schnell (HuggingFace Inference API) |
| Real-Time Search | SerpAPI + LangChain |
| Voice | Web Speech API |

---

## 📁 Project Structure
kushvision-ai/
├── models/
│   ├── llm.py          ← Groq LLM chat logic
│   ├── rag.py          ← RAG pipeline (FAISS + LangChain)
│   ├── image.py        ← Image generation logic
│   └── realtime.py     ← Real-time web search
├── static/
│   ├── welcome.png     ← Background image
│   ├── chat.css / rag.css / image.css
│   └── click.mp3       ← UI sound effect
├── templates/
│   ├── welcome.html / dashboard.html
│   ├── chat.html / rag.html / image.html
├── app.py              ← Flask main app
├── requirements.txt
└── README.md

---

## 🚀 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript, Web Speech API |
| Backend | Python, Flask, Gunicorn |
| LLM | Groq API (LLaMA 3) |
| RAG | LangChain, FAISS, HuggingFace Sentence Transformers |
| Image Generation | FLUX.1-schnell (HuggingFace Inference API) |
| Search | SerpAPI |
| Cloud | AWS EC2 (t3.micro) |
| Web Server | Nginx (Reverse Proxy) |
| SSL | Let's Encrypt (Certbot) |
| Domain | No-IP DDNS |

---

## ☁️ Deployment
AWS EC2 (t3.micro)
└── Nginx (Reverse Proxy + SSL)
└── Gunicorn (WSGI Server)
└── Flask App (Port 5000)
- ✅ SSL via Let's Encrypt (Certbot) — HTTPS enabled
- ✅ Domain — `kushvision.ddns.net`
- ✅ Live URL — https://kushvision.ddns.net

---

## 🌍 Try It Live

👉 **[https://kushvision.ddns.net](https://kushvision.ddns.net)**

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes. AI responses may not always be accurate — use critical judgment.

