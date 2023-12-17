from flask import Flask, render_template, request, jsonify
import moviepy.editor as mp
import vosk
from PIL import Image
import pytesseract
import os
import fitz
import requests

app = Flask(__name__)

FREE_GEN_AI_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
FREE_GEN_AI_API_KEY = "Bearer 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"

@app.route('/')
def home():
    return render_template('index.html')

def extract_audio(video_path):
    print("Extracting audio from video...")

    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio

        audio_file_path = 'uploads/extracted_audio.wav'
        audio.write_audiofile(audio_file_path, codec='pcm_s16le')

        print("Audio extracted and saved:", audio_file_path)

        return audio_file_path

    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_vocals(vocals_file_path):
    print("Transcribing vocals...")

    try:
        model_path = 'modelsp2txt'
        model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(model, 44100)

        chunk_size = 1024
        with open(vocals_file_path, 'rb') as wf:
            while True:
                data = wf.read(chunk_size)
                if not data:
                    break
                recognizer.AcceptWaveform(data)

        result = recognizer.Result()
        transcription = result["text"]

        print("Transcription:", transcription)

        return transcription

    except Exception as e:
        print(f"Error transcribing vocals: {e}")
        return None

def generate_tones_with_free_gen_ai(text):
    print("Generating tones with Free Gen AI...")

    try:
        data = {
            "text": text,
            "key": FREE_GEN_AI_API_KEY
        }

        response = requests.post(FREE_GEN_AI_API_URL, json=data)
        generated_tones = response.json()

        print("Generated tones:", generated_tones)

        return generated_tones

    except Exception as e:
        print(f"Error generating tones with Free Gen AI: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    print("Extracting text from PDF...")

    try:
        pdf_document = fitz.open(pdf_path)

        os.makedirs('uploads/pdf2img', exist_ok=True)

        def convert_page_to_image(page, output_image_path):
            pix = page.get_pixmap()
            pix.save(output_image_path, "png")

        extracted_text = []

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            image_filename = f"uploads/pdf2img/page_{page_number + 1}.png"
            convert_page_to_image(page, image_filename)

            image = Image.open(image_filename)
            text = pytesseract.image_to_string(image)
            extracted_text.append(text)

        pdf_document.close()

        return extracted_text

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

@app.route('/extract_audio', methods=['POST'])
def extract_audio_and_transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    print("File saved:", file_path)

    try:
        audio_file_path = extract_audio(file_path)
        vocals_file_path = transcribe_vocals(audio_file_path)  # Fix: Changed variable name from 'vocals_file_path' to 'transcription'
        extracted_text = transcribe_vocals(vocals_file_path)   # Fix: Changed variable name from 'extracted_text' to 'transcription'
        tones = generate_tones_with_free_gen_ai(extracted_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    return render_template('blog.html', extracted_text=extracted_text, tones=tones)

@app.route('/extract_pdf', methods=['POST'])
def extract_pdf_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    print("File saved:", file_path)

    extracted_text = extract_text_from_pdf(file_path)

    if extracted_text:
        tones = generate_tones_with_free_gen_ai(extracted_text)
        return render_template('blog.html', extracted_text=extracted_text, tones=tones)
    else:
        return jsonify({"error": "Extraction failed."}), 500

if __name__ == '__main__':
    app.run(debug=True)
