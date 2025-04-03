import json
import google.generativeai as genai
from flask import Flask, jsonify, request, send_file, send_from_directory
from gtts import gTTS
import os
import uuid

# Set your API key for generative endpoints if needed
API_KEY = 'AIzaSyBdvIaaunoykSOSm7y2h0itakBG7WYyKpk'
genai.configure(api_key=API_KEY)

app = Flask(__name__)

# Directory for generated TTS audio files
tts_output_dir = "static/tts_audio"
os.makedirs(tts_output_dir, exist_ok=True)

def generate_stream(response):
    """Helper function that yields streaming response data."""
    for chunk in response:
        yield f'data: {json.dumps({"text": chunk.text})}\n\n'

@app.route("/")
def index():
    return send_file('web/index.html')

@app.route("/api/generate", methods=["POST"])
def generate_api():
    """
    Prompting endpoint:
    Always combines a default prompt with the user-provided text.
    """
    try:
        data = request.get_json()
        contents = data["contents"]

        # Default prompt to be prepended
        default_prompt = (
            "Describe the image in a natural, simple, descriptive sentence, "
            "using no more than 35 words. Start your description naturally. "
        )
        # Combine default prompt with user text
        user_text = contents[0]["parts"][1]["text"]
        contents[0]["parts"][1]["text"] = default_prompt + user_text

        model_name = data.get("model", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(contents, stream=True)
        return generate_stream(response), {'Content-Type': 'text/event-stream'}
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/automated", methods=["POST"])
def automated_api():
    """
    Automated endpoint:
    Always uses a fixed standard prompt and expects a base64 image.
    """
    try:
        data = request.get_json()
        image_base64 = data.get("image_base64")
        if not image_base64:
            raise Exception("No image provided.")

        standard_prompt = (
            "Observe the image and describe what you see. You need to say what are the major objects or people in the frame and any major action going on. "
            "Explain which objects or actions are to the left or right. Describe only the objects, people, or actions in the frame in 35 words in a natural, simple style."
        )
        contents = [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                {"text": standard_prompt}
            ]
        }]
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(contents, stream=True)
        return generate_stream(response), {'Content-Type': 'text/event-stream'}
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/tts", methods=["POST"])
def tts_api():
    """
    TTS endpoint:
    Receives text and returns a URL to the generated MP3 using gTTS.
    """
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(tts_output_dir, filename)
        tts = gTTS(text, lang="en")
        tts.save(filepath)
        return jsonify({"url": f"/{tts_output_dir}/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/automated")
def automated_page():
    return send_file('web/automated.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
