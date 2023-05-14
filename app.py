from distutils.log import debug
from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from transformers import pipeline
import yaml
import glob

app = Flask(__name__)

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as data_file:
        data = yaml.safe_load(data_file)
    corpus_data = []
    for conversation in data['conversations']:
        corpus_data.append(conversation[0])
        corpus_data.append(conversation[1])
    return corpus_data

chatbot = ChatBot('ChatBot')

# Load corpus data using modified read_corpus() function
corpus_files = glob.glob('chatterbot_corpus/data/english/*')

trainer = ListTrainer(chatbot)

for corpus_file in corpus_files:
    corpus_data = read_corpus(corpus_file)
    trainer.train(corpus_data)

trainer.train([
    "How are you?",
    "I am good.",
    "What are you doing?",
    "I am chatting with you.",
    "Who made you?",
    "I was created by [ your name ].",
    "What can you do?",
    "I can answer your questions and have conversations with you.",
    "can you introduce yourself?",
    "yes, i'm chatbot ",
    "who is your partners?",
    "my creator",
    "your owner designation?",
    "engineer",
])

qa_pipeline = pipeline("question-answering")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    if "?" in userText:
        qa_response = qa_pipeline({
            "question": userText,
            "context": "chatterbot is a python library for creating chatbots."
        })
        if qa_response['score'] > 0.5:
            return qa_response['answer']

    return str(chatbot.get_response(userText))

if __name__ == "__main__":
    app.run(debug=True)
