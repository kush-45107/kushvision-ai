import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt):
    print("PROMPT:", prompt)

    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)

    print("STATUS:", response.status_code)

    if response.status_code != 200:
        print("ERROR:", response.text)
        return None

    image_path = "static/generated.png"

    with open(image_path, "wb") as f:
        f.write(response.content)

    print("IMAGE SAVED")
    return "/static/generated.png"