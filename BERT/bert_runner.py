import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from My_BertClassifier import BertClassifier
import numpy as np 


ds = load_dataset("sh0416/ag_news")
train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])

# اصلاح برچسب‌ها
train_df['label'] = train_df['label'] - 1
test_df['label'] = test_df['label'] - 1

classes = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

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
train_texts = train_df['description'].tolist()
train_labels = train_df['label'].tolist()
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
plt.savefig("bert_classifier_training_curves.png", dpi=300, bbox_inches='tight')
plt.close()  

def save_model(model, filepath="bert_classifier.pth"):

    torch.save(model.state_dict(), filepath)
    print(f"✅ model saved {filepath} !")

save_model(classifier.model_bert, "bert_classifier.pth")