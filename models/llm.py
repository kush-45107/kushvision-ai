import os
import re
from dotenv import load_dotenv
from models.realtime import search_web
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def normal_chat(user_input):

    q = user_input.lower()

    creator_keywords = [
        "who made you",
        "who devlop you",
        "who devloped you",
        "who created you",
        "who built you",
        "owner of this website",
        "who is the owner of this website",
        "who made this website",
        "who created this website",
        "who made kushvision",
        "who built kushvision",
        "who is your developer",
        "who is your creator"
    ]

    f = user_input.lower()

    about_questions = [
        "what is kushvision",
        "what is kushvision ai",
        "kushvision",
        "kushvision ai",
        "tell me about kushvision ai",
        "tell me about this website",
        "what is this",
        "who made you and who are you",
        "who are you and who made you",
        "about this website",
        "about kushvision ai",
        "what is this chatbot",
        "what can you do",
        "what you can do"
        "who are you"
    ]

    if any(keyword in q for keyword in creator_keywords):
        return "Kushagra Srivastawa made me"

    if any(keyword in f for keyword in about_questions):
        return """KushVision AI is an advanced AI chatbot developed by Kushagra Srivastawa 

    It is a smart multimodal AI assistant where you can:

    Chat with AI and ask anything  
    Upload Files and chat with your documents  
    Generate AI images from text  
    Use voice input and get voice responses  

    KushVision AI is designed to provide a complete AI experience with multiple powerful features in one platform."""

    realtime_words = [
        "today",
        "current",
        "latest",
        "news",
        "price",
        "stock",
        "weather",
        "now",
        "this year",
        "match",
        "score",
        "president",
        "prime minister",
        "election",
        "bitcoin",
        "gold price",
        "winner",
        "result",
    ]

    year_match = re.search(r"20\d{2}", q)

    use_live = any(word in q for word in realtime_words) or year_match

    context = ""

    if use_live:
        try:
            context = search_web(user_input)
        except:
            context = ""
   
    final_prompt = f"""
    You are KushVision AI.

    If real-time context provided use it to answer with latest info.
    If no context provided answer normally using your knowledge.

    Context:
    {context}

    User Question:
    {user_input}
    """

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": final_prompt}],
    )

    return message.choices[0].message.content