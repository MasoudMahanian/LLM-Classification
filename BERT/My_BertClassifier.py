import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np


class CustomClassifier(nn.Module):
    """
    یک Classifier ساده که روی خروجی BERT سوار میشود
    """
    def __init__(self, input_size, num_classes, dropout=0.3):
        super(CustomClassifier, self).__init__()

        
        self.fc1 = nn.Linear(input_size,128)
        self.activation = nn.ReLU() 
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.activation(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

class BertClassifier():
    def __init__(self, classes, model_path="/home/masoudmahanian/my_models/bert", device=None):
        self.model_path = model_path


        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device of BertClassifier is: {self.device}")


        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            local_files_only = True)
            print(f"✔ tokenizer is loaded.")

            self.model_bert = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels = len(classes),
                local_files_only = True
            ).to(self.device)
            print(f"✔ bert model is loaded.")
            print(f"Number of parameters: {sum(p.numel() for p in self.model_bert.parameters()):,}")
            
        except Exception as e:
            print(f"✘ error: {e}")

        for param in self.model_bert.parameters():
            param.requires_grad = False

        BERT_OUTPUT_SIZE = 768

        self.classifier = CustomClassifier(BERT_OUTPUT_SIZE, len(classes), dropout=0.3).to(self.device)

        self.model_bert.classifier = self.classifier

        self.model_bert = self.model_bert.to(self.device)

        total_params = 0
        trainable_params = 0
        for name, param in self.model_bert.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✔ {name}: {param.numel():,} Parameter")
            else:
                print(f"✘ {name}: {param.numel():,} Parameter (locked)")
            total_params += param.numel()


        print(f"\nTotal parameters:")
        print(f" Total parameters: {total_params:,}")
        print(f" Trainable parameters: {trainable_params:,}")
        print(f" Trainable parameters ratio: {100 * trainable_params / total_params:.2f}%")

    def tokenize(self, input, max_length):
        
        
        return self.tokenizer(
            input,
            padding = 'max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )