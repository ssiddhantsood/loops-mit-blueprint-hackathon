from flask import render_template, request, Response, jsonify, redirect, url_for, session
from werkzeug.utils import redirect

from app import app, APP_ROOT
from podcastGeneratorFromScript import createPodcast
from searchOutputs import getSearch

# import cv2
import os
# from app.predictFace import predict_img
# from app.predictEEG import eeg_prediction
# import mne
# import numpy as np
from datetime import datetime
@app.route('/')
def home():
  return render_template('index.html', title='Home')

@app.route('/generate_podcast', methods=['POST'])
def generate_podcast():
    data = request.get_json()
    script = data['script']
    filename = createPodcast(script)  # Assuming createPodcast generates the MP3 and returns the filename
    mp3_path = url_for('static', filename=filename)  # Assuming the MP3 is saved in the static folder
    return jsonify({'mp3Path': mp3_path})

def process_text(prompt):
    # Placeholder function that returns a paragraph of text and 5 links
    paragraph = "This is a huge paragraph of text that acts as a placeholder. It's meant to simulate a search result description or relevant information that the user might find useful. This text can be customized to return dynamic content based on the search query."
    links = [
        "https://www.google.com/webhp?hl=en&sa=X&ved=0ahUKEwi-p6u-pNiEAxWTrYkEHdViBO8QPAgJ",
        "https://www.cnn.com/2024/03/03/politics/dark-money-searle-foundation-invs/index.html",
        prompt,
        "http://example.com/link4",
        "http://example.com/link5"
    ]
    return paragraph, links

@app.route('/process', methods=['POST'])
def process_prompt():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        
        # Use getSearch to obtain the needed information
        answer, sources, urls, script = getSearch(prompt)
        
        # Render the search_results.html template with the obtained data
        return render_template('search_results.html', answer=answer, sources=sources, urls=urls, script = script, prompt = prompt)


@app.route('/search_results')
def search_results():
    # Retrieve the results from the session
    answer = session.get('answer', '')
    sources = session.get('sources', [])
    sourcesInfo = session.get('sourcesInfo', '')  # Assuming this is a single string or similar
    prompt = session.get('prompt', '')
    
    # Render the search results template with the data
    return render_template('search_results.html', answer=answer, sources=sources, sourcesInfo=sourcesInfo, prompt=prompt)


@app.route('/search')
def generate():
  return render_template('search.html', title='Find Article')

@app.route('/edit')
def edit():
  return render_template('edit.html', title='Edit Object')

@app.route('/upload', methods=["GET", "POST"])
def upload():
  target = os.path.join(APP_ROOT, 'temp/')
  now = datetime.now().time()
  print("Time At Beginning =", now)
  if request.method == 'POST':
    file = request.files['eeg']
    file.save("".join([target, 'eeg.bdf']))
    now = datetime.now().time()
    print("Time After EEG Upload =", now)
    return redirect('/eegresults')
