import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from torch import optim
from My_BertClassifier import BertClassifier


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from tqdm import tqdm
from datasets import Dataset, load_dataset

import os 



ds = load_dataset("sh0416/ag_news")
train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])

train_df['label'] = train_df['label'] - 1
test_df['label'] = test_df['label'] - 1

classes = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}


print(pd.Series(train_df['label']).value_counts().sort_index())



# print(train_dataset['description'][0:5])

# quit()
from GPT2_text_generator import My_GPT
agent = My_GPT()




SAMPLE_SIZE = 50000
sample_indices = np.random.choice(len(train_df), SAMPLE_SIZE, replace=False)
sample_data = train_df.iloc[sample_indices]





results=[]
failed_indices = []
for i, (idx, row) in enumerate(tqdm(sample_data.iterrows(), total=SAMPLE_SIZE, desc="Generating")):
    prompt = row['description']
    true_label = row['label']
    # تولید متن
    try:
        result = agent.generate_text(
            prompt=prompt[:200],
            max_length=150,
            temperature=0.8,
            top_p=0.9,
            num_return=1
        )
        # results.append(result[0])
    # except ValueError as e:
    #     print(f"Error: {e} in {i} index")
    #     results.append(prompt)
        generated_text = result[0]
        results.append({
            # 'original_prompt': prompt,
            'generated_text': generated_text,
            'true_label': true_label,
            'label_name': classes[true_label],
            'status': 'success',
            'description' : generated_text
        })
    except Exception as e:
        print(f" error{i}: {e}")
        failed_indices.append(i)
        results.append({
            # 'original_prompt': prompt,
            'generated_text': None,
            'true_label': true_label,
            'label_name': classes[true_label],
            'status': 'failed',
            'error': str(e),
            'description':prompt
        })
    



# ________save__________
results_df = pd.DataFrame(results)
results_df.to_csv('gpt2_generated_texts.csv', index=False)
print(f"\n result is saved in 'gpt2_generated_texts.csv' ")
print(f"sucessfull: {len(results_df[results_df['status'] == 'success'])}")
print(f"unsucessfull: {len(results_df[results_df['status'] == 'failed'])}")



classifier = BertClassifier(classes=classes)
device = classifier.device
# بعد از بارگذاری مدل، دستگاه رو چک کن

def prepare_dataloader(texts, labels, agent, batch_size=32, max_length=128):

    #agent.tokenize
    encodings = agent.tokenize(texts, max_length)

    labels = torch.tensor(labels)
    
    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.labels[idx]
            }
    
    dataset = TextDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_texts = results_df['description'].tolist()
train_labels = results_df['true_label'].tolist()
test_texts = test_df['description'].tolist()
test_labels = test_df['label'].tolist()


train_loader = prepare_dataloader(
    train_texts, train_labels, 
    classifier, 
    batch_size=32
)

test_loader = prepare_dataloader(
    test_texts, test_labels, 
    classifier, 
    batch_size=32
)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, classifier.model_bert.parameters()),
    lr=1e-3
)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10
train_losses = []
test_losses = []
train_accs = []
test_accs = []




# print(f"model device : {next(classifier.model_bert.parameters()).device}")

# در حلقه آموزش، دستگاه batch رو چک کن

for epoch in range(EPOCHS):
    # ---- آموزش ----
    classifier.model_bert.train()
    total_loss = 0
    correct = 0
    total = 0
    classifier.model_bert.train() 
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        optimizer.zero_grad()


        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # print(f"✅ device input_ids: {input_ids.device}")
        # print(f"✅ device input_ids: {attention_mask.device}")
        # print(f"✅ device input_ids: {labels.device}")
        # quit()
        outputs = classifier.model_bert(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'acc': f"{100 * correct / total:.1f}%"
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    
    # ---- ارزیابی روی تست ----
    total_test_loss = 0
    correct_test = 0
    total_test = 0
    test_predictions = []
    classifier.model_bert.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = classifier.model_bert(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_test_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
            
            progress_bar.set_postfix({
                'acc': f"{100 * correct_test / total_test:.1f}%"
            })
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(avg_test_loss)
    test_accs.append(test_accuracy)
    
    print(f"\n Epoch {epoch+1}/{EPOCHS}")

    print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")
    print(f"  Test  Loss: {avg_test_loss:.4f} | Test  Acc: {test_accuracy:.2f}%")

    print("-" * 60)

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Feature Extraction - Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', marker='o')
plt.plot(test_accs, label='Test Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Feature Extraction - Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.savefig("GPT_bert_classifier_training_curves.png", dpi=300, bbox_inches='tight')
plt.close()  

def save_model(model, filepath="GPT_bert_classifier.pth"):

    torch.save(model.state_dict(), filepath)
    print(f"✅ model saved {filepath} !")

save_model(classifier.model_bert, "GPT_bert_classifier.pth")