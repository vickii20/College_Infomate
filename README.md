# **Kongunadu College Chatbot**

A smart chatbot for answering questions about Kongunadu College, powered by Flask, Sentence Transformers, and a local Mistral model via Ollama.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is a Flask-based web application that provides a chatbot interface for answering questions about Kongunadu College. It uses Sentence Transformers for text embeddings and a local Mistral model (via Ollama) for answer generation. The application also includes a contact form for users to submit inquiries.

---

## Features

- **Chatbot:** Answers queries about Kongunadu College using local document search and Mistral.
- **CLI Mode:** Test the chatbot directly from the command line.
- **Contact Form:** Users can submit inquiries via a web form.
- **Modular Design:** Easily extendable for new features or data sources.

---

## Project Structure

```
College_Infomate/
├── app.py                # Main Flask application
├── chat.py               # Chatbot logic
├── templates/
│   ├── index.html        # Main web interface
│   ├── chatbot.html      # Chatbot UI
│   └── chatbot_improved.html
├── static/
│   ├── css/              # Stylesheets and images
│   └── js/               # JavaScript files
├── requirements.txt      # Python dependencies
├── Readme.md             # Project documentation
└── ...                   # Other supporting files
```

---

## Prerequisites

- **OS:** Linux, macOS, or Windows (WSL recommended for Windows)
- **Python:** 3.8+
- **RAM:** 8GB minimum (16GB recommended)
- **Ollama** (for running Mistral model locally)
- **Web Browser**

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DigiDARATechnologies/Infomate.git
cd Infomate/College_Infomate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and Configure Ollama

- Download and install from [Ollama's official website](https://ollama.com/).
- Pull the Mistral model:
  ```bash
  ollama pull mistral:latest
  ```
- Start the Ollama server:
  ```bash
  ollama serve
  ```

---

## Configuration

- Ensure Ollama is running before starting the Flask app.

---

## Usage

### Start the Application

1. **Start Ollama server** (if not already running):
   ```bash
   ollama serve
   ```
2. **Run the Flask app:**
   ```bash
   python app.py
   ```
   - Access the web interface at [http://localhost:5000](http://localhost:5000).

### CLI Mode (Optional)

```bash
python chat.py
```
- Enter questions at the prompt, type `quit` to exit.

---

## How it Works

- **Web Interface:** Flask serves the chatbot UI and contact form.
- **Chatbot Pipeline:** User queries are embedded, relevant documents are searched locally, and Mistral generates answers.
- **Contact Form:** Submissions are processed and (optionally) stored or emailed.

---

## API Endpoints

| Method | Endpoint         | Description                                 |
|--------|------------------|---------------------------------------------|
| GET    | `/`              | Serves the main web interface               |
| POST   | `/submit_contact`| Accepts contact form data (JSON response)   |
| POST   | `/chat`          | Accepts a message, returns chatbot response |

---

## Troubleshooting

- **Ollama Not Responding:** Ensure `ollama serve` is running and the model is pulled.
- **No Answers from Chatbot:** Ensure your local document data is available and properly indexed.
- **Flask Errors:** Check for missing templates or dependency issues.

---

## Future Improvements

- Store contact form submissions in a database (e.g., SQLite, PostgreSQL)
- Email notifications for new inquiries
- Enhanced UI with CSS frameworks (e.g., Bootstrap)
- Multi-turn conversation support
- User authentication

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

---
