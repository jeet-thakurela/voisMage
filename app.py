from flask import Flask, request, jsonify, send_file
from huggingface_hub import InferenceClient
import os
from PIL import Image
import io

app = Flask(__name__)

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
client = InferenceClient(model=repo_id, token=os.environ.get("HF_TOKEN"), timeout=120)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    image = client.text_to_image(prompt)
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    import base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)