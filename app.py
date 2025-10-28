import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='.')

# Configure the Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    
    # Set up the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 0,
      "max_output_tokens": 8192,
    }

    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
    ]

    model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-09-2025",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None

# System instruction for the mentor
SYSTEM_INSTRUCTION = (
    "You are an AI Mentor. Your role is to act as a supportive guide, teacher, and partner "
    "for the user's wellness and personal development. "
    "You must be empathetic, patient, and encouraging. "
    "Focus on helping the user with topics like self-development, stress management, "
    "life balance, and career focus. "
    "Do not just give generic advice; ask gentle, clarifying questions to understand "
    "the user's specific situation. "
    "Your goal is to help the user feel heard, understood, and empowered to make "
    "positive changes in their life. "
    "Always respond in Markdown format for good readability."
)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('ai_mentor_chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the user."""
    if not model:
        return jsonify({"error": "Gemini model is not configured. Check API key."}), 500

    data = request.json
    user_message = data.get('message')
    conversation_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # Format the history for the API
    api_history = []
    for item in conversation_history:
        role = "user" if item['role'] == 'user' else 'model'
        api_history.append({"role": role, "parts": [{"text": item['text']}]})

    # Add the system instruction at the beginning of the history
    full_history = [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        {"role": "model", "parts": [{"text": "Understood. I am ready to help as a supportive and empathetic mentor."}]}
    ] + api_history

    try:
        # Start the chat session
        chat_session = model.start_chat(history=full_history)
        
        # Send the new message
        response = chat_session.send_message(user_message)
        
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error during API call: {e}")
        return jsonify({"error": f"An error occurred while contacting the Gemini API: {e}"}), 500

if __name__ == '__main__':
    # Get port from environment variable, default to 5000.
    # Render will set the 'PORT' environment variable.
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app. 
    # Binding to '0.0.0.0' makes it accessible externally, which Render needs.
    # We set debug=False because this is a deployment, not development.
    app.run(debug=False, host='0.0.0.0', port=port)

