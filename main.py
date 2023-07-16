from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from flask import Flask, request, jsonify
import os

app = Flask(__name__)


class IntentClassifier:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def load_model(self):
        model_path = self.model_dir + "/pytorch_model.bin"
        config_path = self.model_dir + "/config.json"
        tokenizer_path = self.model_dir + "/tokenizer.json"

        if all([file_exists(model_path), file_exists(config_path), file_exists(tokenizer_path)]):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, config=config_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, config=config_path)
            self.classifier = TextClassificationPipeline(
                model=self.model, tokenizer=self.tokenizer)
        else:
            print("Model or tokenizer files not found.")
            print("Downloading the models...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "qanastek/XLMRoberta-Alexa-Intents-Classification")
            self.tokenizer.save_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "qanastek/XLMRoberta-Alexa-Intents-Classification")
            self.model.save_pretrained(self.model_dir)
            self.classifier = TextClassificationPipeline(
                model=self.model, tokenizer=self.tokenizer)

    def classify_intent(self, text) -> str:
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer is not loaded. Please call load_model() first.")
            return "Model or Tokenizer Not Loaded"
        result = self.classifier(text)
        intent = result[0]['label'] if result[0]['score'] > 0.85 else None
        print(intent)
        return intent


def file_exists(path):
    return os.path.isfile(path) and os.path.exists(path)


model_dir = "models"
intent_classifier = IntentClassifier(model_dir)
intent_classifier.load_model()


@app.route('/classify/text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text', '')
    intent = intent_classifier.classify_intent(text)
    response = {'intent': intent}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6666)
