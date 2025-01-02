import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import requests
from bs4 import BeautifulSoup
import PyPDF2
import logging
import yaml
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)


class CodeDataset(Dataset):
    def __init__(self, code_samples):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.code_samples = code_samples

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code = self.code_samples[idx]
        return self.tokenizer.encode(code, return_tensors='pt')


class CodeLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)


class DataHandler:
    def load_training_data(self):
        try:
            with open('C:\\Users\\bijim\\PycharmProjects\\AI_MODEL\\ai_training_data.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'prompts': [], 'categories': [], 'templates': {}}

    def save_training_data(self, data):
        with open('C:\\Users\\bijim\\PycharmProjects\\AI_MODEL\\ai_training_data.json', 'w') as f:
            json.dump(data, f, indent=2)


class ContentFetcher:
    def fetch_web_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error fetching web content: {e}")
            return ""

    def read_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""


class ContentParser:
    def parse_web_content(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        code_blocks = soup.find_all('code')
        parsed_content = []
        for block in code_blocks:
            parsed_content.append(block.get_text())
        return parsed_content

    def parse_pdf_content(self, pdf_text):
        code_pattern = re.compile(r'(def|class|import|from|if|for|while).*:')
        code_blocks = code_pattern.findall(pdf_text)
        return code_blocks


class ModelTrainer:
    def __init__(self, model, optimizer, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def train(self, code_samples, epochs=5):
        logger.info(f"Starting training with {len(code_samples)} samples for {epochs} epochs")
        dataset = CodeDataset(code_samples)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.model(batch, labels=batch).loss
                loss.backward()
                self.optimizer.step()

        logger.info("Training completed")

    def train_classifier(self, vectorizer, classifier, prompts, categories):
        if not prompts:
            return
        X = vectorizer.fit_transform(prompts)
        y = categories
        classifier.fit(X, y)


class AdvancedCodingAssistant:
    def __init__(self, config):
        self.config = config
        self.model = CodeLLM()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=5000)
        self.classifier = MultinomialNB()
        self.data_handler = DataHandler()
        self.content_fetcher = ContentFetcher()
        self.content_parser = ContentParser()
        self.model_trainer = ModelTrainer(self.model, self.optimizer, self.tokenizer)
        self.training_data = self.data_handler.load_training_data()

    def generate_code(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])

    def learn_new_template(self, prompt, code, category, description=''):
        self.training_data['prompts'].append(prompt)
        self.training_data['categories'].append(category)
        self.training_data['templates'][category] = {
            'code': code,
            'description': description
        }
        self.data_handler.save_training_data(self.training_data)
        self.model_trainer.train_classifier(self.vectorizer, self.classifier,
                                            self.training_data['prompts'],
                                            self.training_data['categories'])

    def train_from_source(self, source, source_type='web'):
        if source_type == 'web':
            content = self.content_fetcher.fetch_web_content(source)
            parsed_content = self.content_parser.parse_web_content(content)
        elif source_type == 'pdf':
            content = self.content_fetcher.read_pdf(source)
            parsed_content = self.content_parser.parse_pdf_content(content)
        else:
            raise ValueError("Unsupported source type")

        for code_block in parsed_content:
            self.training_data['prompts'].append(code_block)
            self.training_data['categories'].append('general')

        self.data_handler.save_training_data(self.training_data)
        self.model_trainer.train_classifier(self.vectorizer, self.classifier,
                                            self.training_data['prompts'],
                                            self.training_data['categories'])
        self.model_trainer.train(parsed_content)

    def evaluate_model(self, test_data):
        correct_predictions = 0
        total_predictions = len(test_data)

        for prompt, expected_output in test_data:
            generated_code = self.generate_code(prompt)
            if self.compare_code(generated_code, expected_output):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        return accuracy

    def compare_code(self, generated_code, expected_code):
        return generated_code.strip() == expected_code.strip()

    def preprocess_code(self, code):
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\s+', ' ', code).strip()
        return code


class PromptSystem:
    def __init__(self, ai_model):
        self.ai_model = ai_model

    def start(self):
        print("Welcome to the AI Coding Assistant. Type 'exit' to end the session.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.ai_model.generate_code(user_input)
            print(f"AI: {response}")


# Usage Example
if __name__ == '__main__':
    config = Config('C:\\Users\\bijim\\PycharmProjects\\AI_MODEL\\config.yaml')
    ai_model = AdvancedCodingAssistant(config)
    prompt_system = PromptSystem(ai_model)
    prompt_system.start()
