from flask import Flask, render_template, request, jsonify
import moviepy.editor as mp
import librosa
import numpy as np
import soundfile as sf
import vosk
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def extract_audio(video_path):
    print("Extracting audio from video...")

    try:
        # Use moviepy to extract audio from the video
        video = mp.VideoFileClip(video_path)
        audio = video.audio

        # Save the extracted audio to a separate file in WAV format
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
        # Initialize the Vosk recognizer with the provided model
        model_path = 'modelsp2txt'
        model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(model, 44100)

        # Read the audio file in chunks and perform speech recognition
        chunk_size = 1024
        with open(vocals_file_path, 'rb') as wf:
            while True:
                data = wf.read(chunk_size)
                if not data:
                    break
                recognizer.AcceptWaveform(data)

        # Get the final result
        result = recognizer.Result()

        # Extract transcription from the result
        transcription = result["text"]

        print("Transcription:", transcription)

        return transcription

    except Exception as e:
        print(f"Error transcribing vocals: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    print("Extracting text from PDF...")

    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Ensure the output directory exists
        os.makedirs('uploads/pdf2img', exist_ok=True)

        # Function to convert a PDF page to an image and save it as PNG
        def convert_page_to_image(page, output_image_path):
            pix = page.get_pixmap()
            pix.save(output_image_path, "png")

        # Iterate through pages and convert to images
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            image_filename = f"uploads/pdf2img/page_{page_number + 1}.png"
            convert_page_to_image(page, image_filename)

        # Perform OCR on each image and store extracted text
        extracted_text = []

        for page_number in range(pdf_document.page_count):
            image_filename = f"uploads/pdf2img/page_{page_number + 1}.png"
            image = Image.open(image_filename)
            text = pytesseract.image_to_string(image)
            extracted_text.append(text)

        # Close the PDF document after all operations
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

    # Save the file to the server
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    print("File saved:", file_path)

    # Extract audio from the video in WAV format
    try:
        audio_file_path = extract_audio(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract vocals from the audio
    try:
        vocals_file_path = transcribe_vocals(audio_file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "vocals_path": vocals_file_path,
    })

@app.route('/extract_pdf', methods=['POST'])
def extract_pdf_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the server
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    print("File saved:", file_path)

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(file_path)

    if extracted_text:
        # Return the extracted text
        return jsonify({
            "extracted_text": extracted_text,
        })
    else:
        return jsonify({"error": "Extraction failed."}), 500

if __name__ == '__main__':
    app.run(debug=True)
