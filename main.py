from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY")
openai_model = "openai/gpt-oss-120b"
groq_model = "groq/compound"
client = OpenAI(
    api_key=key,
    base_url="https://api.groq.com/openai/v1",
)


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    response = client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": "Act like a helpful personal assistant. Keep responses minimal."},
            {"role": "user", "content": question},
        ],
        # temperature=0.7,
        # max_output_tokens=512,
    )
    answer = response.output_text.strip()
    return jsonify({"response": answer}), 200


@app.route("/summarize",methods = ["POST"])
def summarize():
    email_text = request.form.get("email")
    prompt = f'Summarize the following email in 2-3 sentences : {email_text}'

    response = client.responses.create(
        model=openai_model,
        input=[
            {'role':'system' , 'content':"Act like an expert email assistant"},
            {'role':'user',"content":prompt}
        ],
        # temperature=0.3,
        # max_output_tokens=512
    )

    summary = response.output_text.strip()
    return jsonify({"response":summary}),200


if __name__ == "__main__":
    app.run(debug=True)
