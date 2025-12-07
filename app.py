from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch, re

app = Flask(__name__)

# Load your model & tokenizer (adjust paths as needed)
model = T5ForConditionalGeneration.from_pretrained("./saved_t5_hugging_face_models")
tokenizer = T5Tokenizer.from_pretrained("./saved_t5_hugging_face_models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip().lower()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize/", methods=["POST"])
def summarize_api():
    data = request.get_json()
    dialogue = data.get("dialogue", "")
    summary = summarize_dialogue(dialogue)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
