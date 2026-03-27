import requests
import os
import uuid
import glob
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

def generate_image(prompt, style="realistic"):
    styled_prompt = f"{prompt}, {style} style, high quality, detailed"
    print("PROMPT:", styled_prompt)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": styled_prompt}, timeout=60)
        print("STATUS:", response.status_code)

        if response.status_code != 200:
            print("ERROR:", response.text)
            return None

        # Purani images delete karo (cleanup)
        for old_file in glob.glob("static/generated_*.png"):
            try:
                os.remove(old_file)
            except:
                pass

        # Unique filename har user ke liye
        unique_name = f"generated_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join("static", unique_name)

        with open(image_path, "wb") as f:
            f.write(response.content)

        print("IMAGE SAVED:", unique_name)
        return f"/static/{unique_name}"

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return None

    except Exception as e:
        print("ERROR:", str(e))
        return None