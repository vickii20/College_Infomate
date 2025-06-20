import logging
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FAISS configuration
FAISS_INDEX_PATH = "college_faiss_index.bin"
DOCUMENTS_PATH = "college_documents.pkl"
EXTRACTED_FILE = "all_extracted_text.txt"

class QueryPreprocessor:
    """Class to preprocess and enhance user queries for better understanding."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common question patterns and their expansions
        self.question_expansions = {
            'admission': ['admission procedure', 'how to apply', 'admission requirements', 'admission criteria'],
            'fee': ['fee structure', 'tuition fee', 'course fee', 'payment', 'cost'],
            'course': ['program', 'degree', 'curriculum', 'subjects', 'syllabus'],
            'placement': ['job', 'career', 'employment', 'recruitment', 'companies'],
            'facility': ['infrastructure', 'amenities', 'equipment', 'resources'],
            'contact': ['phone', 'email', 'address', 'location', 'reach'],
            'faculty': ['professor', 'teacher', 'staff', 'department head'],
            'research': ['publication', 'project', 'funding', 'innovation'],
            'hostel': ['accommodation', 'residence', 'dormitory', 'living'],
            'library': ['books', 'study material', 'digital library', 'reading'],
            'sports': ['gym', 'stadium', 'athletics', 'physical education'],
            'transport': ['bus', 'commute', 'travel', 'transportation']
        }
    
    def clean_query(self, query):
        """Clean and normalize the query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        query = re.sub(r'[^\w\s\?\.]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def expand_query(self, query):
        """Expand query with related terms for better retrieval."""
        expanded_terms = []
        
        # Add original query
        expanded_terms.append(query)
        
        # Check for key terms and add expansions
        for key_term, expansions in self.question_expansions.items():
            if key_term in query:
                for expansion in expansions:
                    if expansion not in query:
                        expanded_terms.append(f"{query} {expansion}")
        
        # Add question variations
        if '?' not in query:
            expanded_terms.append(f"{query}?")
        
        # Add "what is" variations for better understanding
        if not query.startswith(('what', 'how', 'when', 'where', 'why', 'who')):
            expanded_terms.append(f"what is {query}")
            expanded_terms.append(f"tell me about {query}")
        
        return expanded_terms
    
    def extract_keywords(self, query):
        """Extract important keywords from the query."""
        # Tokenize
        tokens = word_tokenize(query.lower())
        
        # Remove stopwords and lemmatize
        keywords = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                keywords.append(lemmatized)
        
        return keywords
    
    def preprocess_query(self, query):
        """Main preprocessing function."""
        # Clean the query
        cleaned_query = self.clean_query(query)
        
        # Extract keywords
        keywords = self.extract_keywords(cleaned_query)
        
        # Expand the query
        expanded_queries = self.expand_query(cleaned_query)
        
        return {
            'original': query,
            'cleaned': cleaned_query,
            'keywords': keywords,
            'expanded': expanded_queries
        }

def load_and_process_documents(file_path=EXTRACTED_FILE):
    """Load documents from extracted.txt and split into chunks."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File {file_path} not found!")
            # Create a sample file with basic content if it doesn't exist
            sample_content = """Kongunadu College Information:
            Kongunadu College is an educational institution.
            The college offers various undergraduate and postgraduate programs.
            It is located in Tamil Nadu, India.
            The college has various departments including Engineering, Arts, and Science.
            """
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            logging.info(f"Created sample {file_path} file")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Handle empty file
        if not content.strip():
            logging.warning(f"File {file_path} is empty or contains only whitespace")
            content = "No content available in the document."
        
        # Split content into chunks (you can adjust chunk size as needed)
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) < 500:  # Chunk size limit
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = ["No meaningful content found in the document."]
        
        logging.info(f"Loaded and processed {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Failed to load documents from {file_path}: {e}")
        # Return a default chunk instead of raising exception
        return ["Default college information: Please provide proper documentation."]

def initialize_faiss_index(documents, embedding_model):
    """Initialize FAISS index with documents."""
    try:
        # Ensure documents is not empty
        if not documents:
            documents = ["Default college information"]
        
        # Generate embeddings for all documents
        embeddings = embedding_model.encode(documents)
        
        # Ensure embeddings is 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        # Save index and documents
        try:
            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, 'wb') as f:
                pickle.dump(documents, f)
            logging.info(f"Saved FAISS index and documents to disk")
        except Exception as save_error:
            logging.warning(f"Could not save FAISS index to disk: {save_error}")
        
        logging.info(f"Created FAISS index with {len(documents)} documents")
        return index, documents
    except Exception as e:
        logging.error(f"Failed to initialize FAISS index: {e}")
        raise

def load_faiss_index():
    """Load existing FAISS index and documents."""
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, 'rb') as f:
                documents = pickle.load(f)
            logging.info(f"Loaded existing FAISS index with {len(documents)} documents")
            return index, documents
        else:
            logging.info("No existing FAISS index found, will create new one")
            return None, None
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        return None, None

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Load and return the embedding model."""
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Failed to load embedding model {model_name}: {e}")
        # Try alternative model
        try:
            logging.info("Trying alternative embedding model...")
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logging.info("Loaded alternative embedding model: paraphrase-MiniLM-L6-v2")
            return model
        except Exception as e2:
            logging.error(f"Failed to load alternative embedding model: {e2}")
            raise

def embed_query(query, embedding_model):
    """Generate embedding for the user query."""
    try:
        # Ensure query is not empty
        if not query or not query.strip():
            query = "default query"
        
        query_embedding = embedding_model.encode([query])
        
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        logging.info(f"Generated embedding for query: {query}")
        return query_embedding.astype('float32')
    except Exception as e:
        logging.error(f"Failed to embed query: {e}")
        raise

def search_documents_enhanced(faiss_index, documents, query_embedding, keywords, embedding_model, limit=8):
    """Enhanced search with multiple strategies."""
    try:
        # Ensure limit doesn't exceed number of documents
        actual_limit = min(limit, len(documents))
        
        # Primary search with original query
        scores, indices = faiss_index.search(query_embedding, actual_limit)
        
        # Retrieve documents
        retrieved_docs = []
        seen_content = set()
        
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(documents):  # Valid index check
                content = documents[idx]
                if content not in seen_content:
                    retrieved_docs.append(content)
                    seen_content.add(content)
        
        # If we don't have enough documents, try keyword-based search
        if len(retrieved_docs) < 3 and keywords:
            # Create keyword-based query
            keyword_query = " ".join(keywords[:5])  # Use top 5 keywords
            if keyword_query:
                keyword_embedding = embed_query(keyword_query, embedding_model)
                keyword_scores, keyword_indices = faiss_index.search(keyword_embedding, actual_limit//2)
                
                for idx in keyword_indices[0]:
                    if 0 <= idx < len(documents):
                        content = documents[idx]
                        if content not in seen_content:
                            retrieved_docs.append(content)
                            seen_content.add(content)
        
        # Ensure we have at least one document
        if not retrieved_docs and documents:
            retrieved_docs = [documents[0]]  # Return first document as fallback
        
        logging.info(f"Retrieved {len(retrieved_docs)} documents from FAISS")
        return retrieved_docs
    except Exception as e:
        logging.error(f"Failed to search FAISS: {e}")
        # Return first document as fallback
        return [documents[0]] if documents else ["No documents available"]

def initialize_llm(model_name="mistral:latest", base_url="http://localhost:11434"):
    """Initialize the local Mistral model via Ollama."""
    try:
        llm = OllamaLLM(model=model_name, base_url=base_url)
        # Test the connection
        test_response = llm.invoke("Hello")
        logging.info(f"Initialized Ollama LLM: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize Ollama LLM: {e}")
        logging.info("Please ensure Ollama is running and the model is available")
        raise

def generate_answer_enhanced(llm, documents, user_query, preprocessed_query):
    """Generate an answer using enhanced prompting and context."""
    try:
        # Ensure we have documents
        if not documents:
            return "No documents available to answer the question."
        
        # Simplified and more focused prompt template
        prompt_template = """
You are a helpful assistant for Kongunadu College of Engineering and Technology. Answer the user's question based on the provided documents.

IMPORTANT: Keep your response concise and directly answer what was asked. Do not add extra information unless specifically requested.

USER QUESTION: {question}

RELEVANT DOCUMENTS:
{context}

Provide a clear, direct answer to the question. If the information is not in the documents, simply say "I don't have that information in my database."

ANSWER:
"""
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["question", "context"]
        )
        
        # Combine documents into context
        context = "\n\n".join([doc.strip() for doc in documents if doc and doc.strip()])
        if not context:
            logging.warning("No valid documents retrieved for context")
            return "I don't have information to answer your question. Please try rephrasing or contact the college directly."
        
        # Ensure query is not empty
        if not user_query or not user_query.strip():
            return "Please provide a valid question."
        
        # Generate the answer
        try:
            chain = prompt | llm
            response = chain.invoke({
                "question": user_query,
                "context": context
            })
        except Exception as llm_error:
            logging.error(f"LLM generation error: {llm_error}")
            return "Sorry, there was an error generating the response. Please check if Ollama is running."
        
        # Clean up the response
        if isinstance(response, str):
            answer = response.strip()
        else:
            answer = str(response).strip()
        
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        
        if not answer:
            logging.info(f"Empty answer generated for question: {user_query}")
            return "I couldn't generate a proper answer for your question."
        
        logging.info(f"Generated enhanced answer for question '{user_query}'")
        return answer
    except Exception as e:
        logging.error(f"Failed to generate answer: {e}")
        return "An error occurred while generating the answer."

def setup_faiss_database():
    """Setup FAISS database with college information."""
    try:
        # Load embedding model
        embedding_model = load_embedding_model()
        
        # Try to load existing index
        faiss_index, documents = load_faiss_index()
        
        if faiss_index is None:
            # Create new index from extracted.txt
            documents = load_and_process_documents()
            faiss_index, documents = initialize_faiss_index(documents, embedding_model)
        
        return faiss_index, documents, embedding_model
    except Exception as e:
        logging.error(f"Failed to setup FAISS database: {e}")
        raise

def answer_question_enhanced(user_query):
    """Enhanced main function to answer user questions using improved RAG with FAISS."""
    try:
        # Initialize components
        faiss_index, documents, embedding_model = setup_faiss_database()
        llm = initialize_llm()
        preprocessor = QueryPreprocessor()
        
        # Preprocess the query
        preprocessed_query = preprocessor.preprocess_query(user_query)
        logging.info(f"Preprocessed query: {preprocessed_query}")
        
        # Use the cleaned query for embedding
        query_embedding = embed_query(preprocessed_query['cleaned'], embedding_model)
        
        # Enhanced document search
        retrieved_documents = search_documents_enhanced(
            faiss_index, 
            documents, 
            query_embedding, 
            preprocessed_query['keywords'],
            embedding_model
        )
        
        # Generate enhanced answer
        answer = generate_answer_enhanced(llm, retrieved_documents, user_query, preprocessed_query)
        
        return answer
    except Exception as e:
        logging.error(f"Error in answer_question_enhanced: {e}")
        return f"Sorry, an error occurred: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Enhanced Kongunadu College Chatbot!")
    print("This version has improved question understanding capabilities.")
    print("Type 'quit' to end the chat.")
    print("Setting up FAISS database...")
    
    try:
        # Setup database once at startup
        setup_faiss_database()
        print("✅ FAISS database ready!")
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("Please check:")
        print("1. If the 'all_extracted_text.txt' file exists")
        print("2. If sentence-transformers is installed: pip install sentence-transformers")
        print("3. If FAISS is installed: pip install faiss-cpu")
        print("4. If NLTK is installed: pip install nltk")
        exit(1)
    
    while True:
        try:
            user_query = input("\n📝 Enter your question: ")
            if user_query.lower() == "quit":
                print("Goodbye! Chat ended.")
                break

            if not user_query.strip():
                print("Please enter a valid question.")
                continue

            print("\n🤔 Processing your question...")
            final_answer = answer_question_enhanced(user_query)
            print(f"\n🎯 Answer:\n{final_answer}")
        except KeyboardInterrupt:
            print("\nGoodbye! Chat ended.")
            break
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Unexpected error in main loop: {e}") 