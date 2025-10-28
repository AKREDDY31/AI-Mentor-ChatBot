import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__, template_folder='.')

# Configure the Gemini API
# This configuration will now use the environment variable set in Render's dashboard
API_KEY = os.getenv("GEMINI_API_KEY")

model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        
        # Set up safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-09-2025",
                                      generation_config={"temperature": 0.75, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
                                      safety_settings=safety_settings)
    except Exception as e:
        print(f"Error configuring Gemini model: {e}")
        model = None
else:
    print("GEMINI_API_KEY not found. Please set it in your environment.")

# System instruction for the mentor
SYSTEM_INSTRUCTION = (
    "You are 'Mentor', an advanced AI designed to be a supportive and empathetic partner. "
    "Your primary goal is to help users with their wellness, self-development, and career focus. "
    "Act as a guide, teacher, and mentor. "
    "Your responses should be encouraging, insightful, and focused on helping the user build self-trust, "
    "manage stress, find life balance, and stay focused on their goals. "
    "Ask clarifying questions to understand their needs better. "
    "Provide actionable, small steps (like 'Micro-Wins') they can take. "
    "Never be judgmental. Always be patient and supportive. "
    "Always respond in Markdown format for good readability."
)

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Renders the HTML file from the same directory
    return render_template('ai_mentor_chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the user."""
    if not model:
        # This error will show if the API key was missing or invalid on startup
        return jsonify({"error": "Gemini model is not configured. Check API key."}), 500

    data = request.json
    
    user_message = data.get('message')
    conversation_history = data.get('history', [])

    if not user_message:
        # This error was the one we were seeing
        return jsonify({"error": "No message provided."}), 400

    # Format the history for the API
    # The history from the client is already in the correct {role, parts} format.
    api_history = []
    for item in conversation_history:
        # Ensure 'parts' is a list of objects with 'text'
        if 'parts' in item and isinstance(item['parts'], list):
            api_history.append(item)
        elif 'text' in item and 'role' in item: # Handle old format just in case
             api_history.append({"role": item['role'], "parts": [{"text": item['text']}]})


    # Add the system instruction at the beginning of the history
    # This guides the model's persona for every new chat.
    full_history = [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        {"role": "model", "parts": [{"text": "Understood. I am ready to help as a supportive and empathetic mentor."}]}
    ] + api_history

    try:
        # Start a chat session with the full history
        chat_session = model.start_chat(history=full_history)
        
        # Send the new message
        response = chat_session.send_message(user_message)
        
        # Return the model's reply
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error during API call: {e}")
        return jsonify({"error": f"An error occurred while contacting the Gemini API: {e}"}), 500

if __name__ == '__main__':
    # Get the port from the environment (Render sets this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app. 
    # '0.0.0.0' is crucial for Render to connect.
    # debug=False is crucial for production.
    app.run(debug=False, host='0.0.0.0', port=port)

