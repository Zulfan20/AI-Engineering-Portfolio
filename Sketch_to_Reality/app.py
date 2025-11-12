# -----------------------------------------------------------------
# Sketch-to-Reality AI Server
# File: app.py
# -----------------------------------------------------------------
# This server runs the AI models on your GPU.
# It waits for a sketch and a prompt from the webpage,
# generates an image, and sends it back.
# -----------------------------------------------------------------

import torch
import base64
import io
import cv2 # OpenCV for image processing
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

# --- 1. Model Setup ---
# This part is hardware-dependent. Your RTX 2050 has 4GB of VRAM,
# so we MUST optimize for it.

print("Loading AI models... This will take a few minutes and download several GBs.")

# Load the Canny edge detection model (for reading sketches)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float16
)

# Load the main image generation model (Stable Diffusion)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)

# --- 2. VRAM & Speed Optimizations (CRITICAL for your RTX 2050) ---
# Use a faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# This is the most important line:
# It shuffles models from GPU to CPU so you don't run out of VRAM.
pipe.enable_sequential_cpu_offload()

# We are not training, so no gradients needed
pipe.set_progress_bar_config(disable=True)

print("AI Models loaded successfully and optimized for your GPU.")

# --- 3. Flask Server Setup ---
app = Flask(__name__)
# Allow the webpage to talk to this server
CORS(app)

# --- 4. The "Home" Page ---
# This serves the index.html file
@app.route('/')
def home():
    print("Serving the main webpage (index.html).")
    return render_template('index.html')

# --- 5. The "Generate" Endpoint (The AI "Magic") ---
@app.route('/generate', methods=['POST'])
def generate_image():
    print("Received a generation request...")
    try:
        # Get the data from the webpage
        data = request.json
        prompt = data.get('prompt', 'a high-quality photograph')
        base64_image = data.get('image')

        if not base64_image:
            raise ValueError("No image data received.")

        # --- Image Pre-processing ---
        # 1. Decode the base64 sketch
        # The base64 string has a prefix 'data:image/png;base64,' which we remove
        image_data = base64.b64decode(base64_image.split(',')[1])
        sketch_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 2. Convert to OpenCV format (numpy array)
        sketch_np = np.array(sketch_image)

        # 3. Process the sketch with Canny edge detection
        # This turns your simple drawing into a detailed "edge map"
        # that the AI can understand.
        low_threshold = 100
        high_threshold = 200
        canny_map_np = cv2.Canny(sketch_np, low_threshold, high_threshold)
        
        # 4. Convert back to a 3-channel PIL Image
        canny_map_pil = Image.fromarray(canny_map_np)

        # --- Run the AI ---
        print(f"Running AI with prompt: '{prompt}'")
        
        # We run this on the 'cuda' (NVIDIA GPU) device.
        # This will take 10-30 seconds on your RTX 2050.
        output_image = pipe(
            prompt,
            num_inference_steps=20, # 20 steps is fast and good quality
            image=canny_map_pil,
        ).images[0]

        print("Generation complete. Sending image back to webpage.")

        # --- Image Post-processing ---
        # 1. Convert the output image back to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. Send the new image back to the webpage
        return jsonify({
            'image': 'data:image/png;base64,' + img_str
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# --- 6. Start the Server ---
if __name__ == '__main__':
    # We run on http (simple) and port 5000.
    # No HTTPS, no SSL, no headaches.
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)