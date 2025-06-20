## Kongunadu College Chatbot 

## Overview

This project is a Flask-based web application that provides a chatbot interface for answering questions about Kongunadu College. It uses a Retrieval-Augmented Generation (RAG) pipeline powered by Qdrant for vector search, Sentence Transformers for text embeddings, and a local Mistral model (via Ollama) for answer generation. The application also includes a contact form for users to submit inquiries.
Features

## Chatbot: 
Answers queries about Kongunadu College by retrieving relevant documents from a Qdrant vector database and generating responses using a local Mistral model.
CLI Mode: A command-line interface for testing the chatbot directly.

## Project Structure

app.py: Main Flask application defining routes for the web interface, contact form, and chatbot.
chat.py: Implements the RAG pipeline, including Qdrant client setup, embedding generation, document retrieval, and answer generation.
templates/index.html: (Assumed) HTML template for the web interface (not provided but referenced in app.py).
requirements.txt: Lists Python dependencies required for the project.

## Prerequisites

Operating System: Linux, macOS, or Windows (with WSL recommended for Windows users).
Python: Version 3.8 or higher.
Internet Access: Required for downloading dependencies and pulling the Mistral model.
Hardware: At least 8GB RAM (16GB recommended) for running Ollama and the Mistral model locally.
Qdrant Cloud Account: For vector database storage (alternatively, a local Qdrant instance can be used).
Web Browser: For accessing the Flask web interface.

## Installation
Step 1: Clone the Repository
Clone the project repository to your local machine and navigate to the project directory:
git clone https://github.com/DigiDARATechnologies/Infomate.git
cd <repository-directory>

Step 2: Install Python Dependencies
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

The requirements.txt file includes:

Flask: Web framework for the application.
qdrant-client: For interacting with the Qdrant vector database.
sentence-transformers: For generating text embeddings.
langchain-ollama: For interfacing with the Mistral model via Ollama.

If you encounter issues, ensure pip is updated:
pip install --upgrade pip

Step 3: Install and Configure Ollama
Ollama is used to run the Mistral language model locally for answer generation.

Install Ollama:

Windows:
Download the installer from Ollama's official website.
Run the installer and follow the prompts.

Verify installation:ollama --version


Pull the Mistral Model:Pull the mistral:latest model (or the specific version referenced in chat.py):
ollama pull mistral:latest

This downloads the model (several GBs, so ensure a stable internet connection).

Start the Ollama Server:Run the Ollama server in the background:
ollama serve

The server must be running when you start the Flask application. It listens on http://localhost:11434 by default, as configured in chat.py.

Verify the Model:Test the model to ensure it’s working:
ollama run mistral:latest "Hello, how are you?"

You should see a response from the model.


Step 5: Set Up Qdrant
The application uses Qdrant as a vector database to store and retrieve document embeddings.

Option 1: Use Qdrant Cloud (Recommended for Simplicity):

Sign up for a Qdrant Cloud account at Qdrant Cloud.
Create a new cluster and obtain the cluster URL and API key.
Update the QDRANT_URL and QDRANT_API_KEY in chat.py with your credentials:QDRANT_URL = "https://<your-cluster-url>.qdrant.io:6333" //api key generate page =>usage Examples =>  choose python =>shown qdrant url and api key shown just  copy and replace the main code
QDRANT_API_KEY = "<your-api-key>"




Option 2: Run Qdrant Locally:

Install Docker (required for running Qdrant locally):
Linux: sudo apt-get install docker.io
macOS: Install Docker Desktop.
Windows: Install Docker Desktop or use WSL.

Pull and run the Qdrant Docker image:docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Update QDRANT_URL in chat.py to:QDRANT_URL = "http://localhost:6333"

No API key is needed for a local instance (QDRANT_API_KEY = "").


Create a Qdrant Collection:

The application expects a collection named kongunadu_data (defined in chat.py as COLLECTION_NAME).
Use the Qdrant dashboard (available in Qdrant Cloud or at http://localhost:6334/dashboard for a local instance) to create a collection:
Name: kongunadu_data
Vector size: 384 (matches the all-MiniLM-L6-v2 model used by Sentence Transformers)
Distance metric: Cosine


Alternatively, create the collection programmatically using the Qdrant client in Python:from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client.create_collection(
    collection_name="kongunadu_data",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)




Populate the Collection:

The application assumes the kongunadu_data collection contains pre-indexed documents about Kongunadu College.
To add documents, you can use the Qdrant client to upload text content with embeddings. Example:from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["Sample document about Kongunadu College...", "..."]
embeddings = model.encode(documents).tolist()
client.upload_collection(
    collection_name="kongunadu_data",
    vectors=embeddings,
    payload=[{"content": doc} for doc in documents],
    ids=[i for i in range(len(documents))]
)


Replace "Sample document..." with actual college-related content (e.g., FAQs, course details, etc.).




Step 6: Run the Application

Start the Ollama Server (if not already running):
ollama serve


Start the Flask Application:
python app.py

The application runs on http://0.0.0.0:5000 in debug mode. Open a browser and navigate to http://localhost:5000 to access the web interface.

Test the Chatbot in CLI Mode (Optional):Run chat.py directly for a command-line interface:
python chat.py

Enter questions at the prompt and type quit to exit.


## How the Program Works

Web Interface (app.py):

The Flask app serves index.html at the root endpoint (/).
The /submit_contact endpoint processes contact form submissions, returning a JSON response (currently does not persist data).
The /chat endpoint handles chatbot queries by calling answer_question from chat.py and returning the response as JSON.


## RAG Pipeline (chat.py):

Initialization: Connects to Qdrant, loads the Sentence Transformer model (all-MiniLM-L6-v2), and initializes the Mistral model via Ollama.
Query Processing:
Embeds the user’s query using Sentence Transformers.
Searches the kongunadu_data collection in Qdrant for the top 5 relevant documents.
Combines retrieved documents into a context and uses a prompt template to guide the Mistral model.
Generates an answer using the Mistral model and returns it.


Error Handling: Logs errors and returns fallback messages if no documents are found or if errors occur.


## Dependencies:

Flask (3.0.3): Handles HTTP requests and serves the web interface.
qdrant-client (1.12.0): Communicates with the Qdrant vector database for document retrieval.
sentence-transformers (3.2.0): Generates 384-dimensional embeddings for queries and documents.
langchain-ollama (0.2.0): Interfaces with the local Mistral model for answer generation.
logging: Built-in Python module for debugging and logging.



## API Endpoints

GET /: Serves the index.html template.
POST /submit_contact: Accepts form data (name, email, phone, subject, message) and returns a JSON response (success and message).
POST /chat: Accepts a message parameter, processes it through the RAG pipeline, and returns a JSON response with the answer or an error.

Troubleshooting

Ollama Not Responding:
Ensure the Ollama server is running (ollama serve).
Verify the model is pulled (ollama list).
Check http://localhost:11434 is accessible.


## Qdrant Connection Errors:
Verify QDRANT_URL and QDRANT_API_KEY in chat.py.
For local Qdrant, ensure Docker is running and the container is active (docker ps).


## No Answers from Chatbot:
Ensure the kongunadu_data collection exists and contains relevant documents.
Test the query embedding and search directly using the Qdrant client.


## Flask Errors:
Ensure templates/index.html exists.
Check the Flask console output for errors (runs in debug mode).



## Future Improvements

Add database integration (e.g., SQLite, PostgreSQL) for storing contact form submissions.
Implement email notifications for contact form submissions.
Enhance the web interface with CSS frameworks (e.g., Bootstrap) for better styling.
Support multi-turn conversations in the chatbot.
Add authentication for secure access to the web interface.

## License
This project is licensed under the MIT License.
