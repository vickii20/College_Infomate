from flask import Flask, render_template, request, jsonify
from improved_sample import answer_question_enhanced  # Import the improved chatbot logic
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Here you would typically save this data to a database
        # or send an email notification
        
        return jsonify({
            'success': True,
            'message': 'Form submitted successfully!'
        })

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form.get('message')
        if not user_input:
            return jsonify({'success': False, 'message': 'Message cannot be empty'})

        try:
            # Call the improved answer_question_enhanced function
            response = answer_question_enhanced(user_input)
            return jsonify({'success': True, 'message': response})
        except Exception as e:
            logging.error(f"Error in chat endpoint: {e}")
            return jsonify({'success': False, 'message': f'Error processing query: {str(e)}'})

@app.route('/chat_enhanced', methods=['POST'])
def chat_enhanced():
    """Enhanced chat endpoint with better error handling and logging."""
    if request.method == 'POST':
        user_input = request.form.get('message')
        if not user_input:
            return jsonify({'success': False, 'message': 'Please enter a question.'})

        try:
            logging.info(f"Processing user query: {user_input}")
            
            # Call the improved answer_question_enhanced function
            response = answer_question_enhanced(user_input)
            
            logging.info(f"Generated response for query: {user_input}")
            return jsonify({
                'success': True, 
                'message': response,
                'query': user_input
            })
        except Exception as e:
            logging.error(f"Error processing query '{user_input}': {e}")
            return jsonify({
                'success': False, 
                'message': 'Sorry, I encountered an error processing your question. Please try rephrasing it or contact the college directly.',
                'error': str(e)
            })

@app.route('/health')
def health_check():
    """Health check endpoint to verify the chatbot is working."""
    try:
        # Test the chatbot with a simple query
        test_response = answer_question_enhanced("What is Kongunadu College?")
        return jsonify({
            'status': 'healthy',
            'message': 'Chatbot is working properly',
            'test_response': test_response[:100] + "..." if len(test_response) > 100 else test_response
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': f'Chatbot error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Kongunadu College Chatbot Web Server...")
    print("📝 This version includes improved question understanding capabilities!")
    print("🌐 Server will be available at: http://localhost:5000")
    print("🔧 Enhanced chat endpoint: http://localhost:5000/chat_enhanced")
    print("❤️  Health check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000) 