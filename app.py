from flask import Flask, render_template, request, redirect, url_for
from models.llm import chat as normal_chat
from models.rag import process_file, ask_file
from models.image import generate_image
import markdown

app = Flask(__name__)

chat_history = []

@app.route("/", methods=["GET","POST"])
def welcome():
    if request.method == "POST":
        name = request.form.get("name")
        return redirect(url_for("dashboard", username=name))
    return render_template("welcome.html")

@app.route("/dashboard/<username>")
def dashboard(username):
    return render_template("dashboard.html", name=username)

@app.route("/chat/<username>", methods=["GET","POST"])
def chat_page(username):
    reply = ""
    plain_reply = ""  # clean text for speech (no HTML, no markdown)

    if request.form.get("user_input"):
        user_input = request.form.get("user_input")
        plain_reply = normal_chat(user_input)         # raw text → for speech
        reply = markdown.markdown(plain_reply)        # HTML → for display

        chat_history.append({"q": user_input, "a": reply})

    return render_template("chat.html",
                           reply=reply,
                           plain_reply=plain_reply,
                           history=chat_history,
                           username=username)

@app.route("/rag/<username>", methods=["GET","POST"])
def rag_page(username):
    reply = ""

    if request.files.get("file"):
        file = request.files.get("file")
        if file.filename != "":
            result = process_file(file)
            reply = result

    if request.form.get("user_input"):
        question = request.form.get("user_input")
        answer = ask_file(question)
        reply = answer

    return render_template("rag.html",
                           reply=reply,
                           username=username)

@app.route("/image/<username>", methods=["GET","POST"])
def image_page(username):
    img_path = None

    if request.form.get("image_prompt"):
        prompt = request.form.get("image_prompt")
        img_path = generate_image(prompt)

    return render_template("image.html",
                           image=img_path,
                           username=username)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debu)
