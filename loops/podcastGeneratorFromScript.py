import os
import serpapi
import openai
from dotenv import load_dotenv
from openai import OpenAI
import nltk
import uuid

load_dotenv()
serpapi_key = os.getenv('SERPAPI_KEY')

def createPodcast(script):
    print("GENERATION IS HAPPENING")
    client = OpenAI(api_key="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")
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=script,
    )
    random_id = uuid.uuid4()
    response.stream_to_file(filename := f"app/static/{str(random_id)}.mp3")
    return f"{str(random_id)}.mp3"

