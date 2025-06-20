
<h1 align="center">College - Infomate</h1>

<h1 align="center">Kongunadu College Chatbot</h1>
<p align="center">
<b>InfoMate_AI</b> is an AI-powered chatbot web application for answering queries about Kongunadu College of Engineering and Technology.<br>
Built with <b>Flask</b>, <b>Sentence Transformers</b>, and a local <b>Mistral</b> model via <b>Ollama</b>.
</p>
<p align="center">
<a href="#features">Features</a> •
<a href="#demo">Demo</a> •
<a href="#installation">Installation</a> •
<a href="#usage">Usage</a> •
<a href="#project-structure">Project Structure</a> •
<a href="#contributing">Contributing</a> •
<a href="#license">License</a>
</p>

---

## 🧠 Features

- **Chatbot**: Answers questions about Kongunadu College using advanced language models.
- **Web Interface**: Modern, responsive UI (see `templates/index.html`).
- **Contact Form**: Users can submit inquiries.
- **CLI Mode**: Test the chatbot from the command line.
- **Easy Setup**: Minimal configuration required.

---

## 🗂️ Project Structure

```
InfoMate_AI/
  ├── app.py                   # Main Flask app
  ├── chat.py                  # Chatbot logic
  ├── sample.py                # Sample/utility scripts
  ├── requirements.txt         # Python dependencies
  ├── static/                  # CSS, JS, images
  ├── templates/               # HTML templates
  └── lib/                     # Utilities
```

---

## ⚙️ Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (for local Mistral model)
- At least 8GB RAM (16GB recommended)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/InfoMate_AI.git
cd InfoMate_AI
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama & Mistral

- [Download Ollama](https://ollama.com/download)
- Pull the Mistral model:
  ```bash
  ollama pull mistral:latest
  ```
- Start the Ollama server:
  ```bash
  ollama serve
  ```

---

## 🖥️ Usage

### Web Interface

```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000).

### CLI Mode

```bash
python chat.py
```

---

## 🧩 API Endpoints

- `GET /` — Main web interface
- `POST /submit_contact` — Contact form (JSON)
- `POST /chat` — Chatbot queries (JSON)

---

## 📝 Troubleshooting

- **Ollama not responding**: Ensure `ollama serve` is running and model is pulled.
- **No answers**: Ensure your data files are present and accessible.
- **Flask errors**: Check for missing templates or dependency issues.

---

## 🌱 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📚 Resources

- [Ollama](https://ollama.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)

---
