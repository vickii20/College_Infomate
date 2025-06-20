import logging
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import numpy as np

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

# Qdrant configuration
QDRANT_URL = "https://52c71880-97e5-49f3-ae6f-3b544591fa9c.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.28BkUSCS3mMpCLBcrGxZ6RkaDvX94dV8sE5SPQg_ipc"
COLLECTION_NAME = "deskmate"

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

def initialize_qdrant_client(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=COLLECTION_NAME):
    """Initialize and return a Qdrant client."""
    try:
        client = QdrantClient(url=url, api_key=api_key)
        logging.info(f"Successfully connected to Qdrant at {url}")
        return client, collection_name
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}")
        raise

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Load and return the embedding model."""
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise

def embed_query(query, embedding_model):
    """Generate embedding for the user query."""
    try:
        query_embedding = embedding_model.encode(query).tolist()
        logging.info(f"Generated embedding for query: {query}")
        return query_embedding
    except Exception as e:
        logging.error(f"Failed to embed query: {e}")
        raise

def search_documents_enhanced(qdrant_client, collection_name, query_embedding, keywords, embedding_model, limit=8):
    """Enhanced search with multiple strategies."""
    try:
        # Primary search with original query
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        documents = []
        seen_content = set()
        
        # Add primary results
        for result in search_result:
            content = result.payload.get('content', '')
            if content and content not in seen_content:
                documents.append(content)
                seen_content.add(content)
        
        # If we don't have enough documents, try keyword-based search
        if len(documents) < 3:
            # Create keyword-based query
            keyword_query = " ".join(keywords[:5])  # Use top 5 keywords
            if keyword_query:
                keyword_embedding = embedding_model.encode(keyword_query).tolist()
                keyword_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=keyword_embedding,
                    limit=limit//2,
                    with_payload=True
                )
                
                for result in keyword_results:
                    content = result.payload.get('content', '')
                    if content and content not in seen_content:
                        documents.append(content)
                        seen_content.add(content)
        
        logging.info(f"Retrieved {len(documents)} documents from Qdrant")
        return documents
    except Exception as e:
        logging.error(f"Failed to search Qdrant: {e}")
        raise

def initialize_llm(model_name="mistral:latest", base_url="http://localhost:11434"):
    """Initialize the local Mistral model via Ollama."""
    try:
        llm = OllamaLLM(model=model_name, base_url=base_url)
        logging.info(f"Initialized Ollama LLM: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize Ollama LLM: {e}")
        raise

def generate_answer_enhanced(llm, documents, user_query, preprocessed_query):
    """Generate an answer using enhanced prompting and context."""
    try:
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
        context = "\n\n".join([doc.strip() for doc in documents if doc.strip()])
        if not context:
            logging.warning("No valid documents retrieved for context")
            return "I don't have information to answer your question. Please try rephrasing or contact the college directly."
        
        # Generate the answer
        chain = prompt | llm
        response = chain.invoke({
            "question": user_query,
            "context": context
        })
        
        # Clean up the response
        answer = response.strip()
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        
        if not answer or "I couldn't find" in answer.lower():
            logging.info(f"No answer found for question: {user_query}")
            return "I don't have that information in my database."
        
        logging.info(f"Generated enhanced answer for question '{user_query}': {answer}")
        return answer
    except Exception as e:
        logging.error(f"Failed to generate answer: {e}")
        return "An error occurred while generating the answer."

def answer_question_enhanced(user_query, qdrant_url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=COLLECTION_NAME):
    """Enhanced main function to answer user questions using improved RAG."""
    try:
        # Initialize components
        qdrant_client, collection_name = initialize_qdrant_client(qdrant_url, api_key, collection_name)
        embedding_model = load_embedding_model()
        llm = initialize_llm()
        preprocessor = QueryPreprocessor()
        
        # Preprocess the query
        preprocessed_query = preprocessor.preprocess_query(user_query)
        logging.info(f"Preprocessed query: {preprocessed_query}")
        
        # Use the cleaned query for embedding
        query_embedding = embed_query(preprocessed_query['cleaned'], embedding_model)
        
        # Enhanced document search
        documents = search_documents_enhanced(
            qdrant_client, 
            collection_name, 
            query_embedding, 
            preprocessed_query['keywords'],
            embedding_model
        )
        
        # Generate enhanced answer
        answer = generate_answer_enhanced(llm, documents, user_query, preprocessed_query)
        
        return answer
    except Exception as e:
        logging.error(f"Error in answer_question_enhanced: {e}")
        return f"Sorry, an error occurred while processing your question: {str(e)}. Please try again or contact the college directly."

if __name__ == "__main__":
    print("Welcome to the Enhanced Kongunadu College Chatbot!")
    print("This version has improved question understanding capabilities.")
    print("Type 'quit' to end the chat.")
    
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