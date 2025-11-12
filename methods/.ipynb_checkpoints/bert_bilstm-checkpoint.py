from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# BiLSTM Classifier
class BiLSTMClassifier(nn.Module):
    def __init__(self, bert, num_classes=3, hidden_dim=128):
        super(BiLSTMClassifier, self).__init__()
        self.bert = bert
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # BiLSTM has 2*hidden_dim output

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_output.last_hidden_state  # Shape: (batch_size, seq_len, 768)
        lstm_out, _ = self.lstm(hidden_states)         # Shape: (batch_size, seq_len, hidden_dim*2)
        pooled_output = lstm_out[:, -1, :]             # Use the last hidden state (batch_size, hidden_dim*2)
        x = self.dropout(pooled_output)
        x = self.fc(x)                                 # Final classification layer
        return x


# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')

    # Print classification report for detailed performance on each class
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive']))


# Main script
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv(r"C:/Users/50643/COMP9444/Project/Data/final_dataset/COVIDSenti-C_cleanest.csv")
    data = data.dropna(subset=['processed'])
    data['label'] = data['label'].map({"neg": 0, "neu": 1, "pos": 2})

    # Split dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['processed'].values,
        data['label'].values,
        test_size=0.2,
        random_state=42
    )

    # Initialize BERT tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Create datasets and loaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize BiLSTM model
    num_classes = 3
    model = BiLSTMClassifier(bert=bert_model, num_classes=num_classes, hidden_dim=128)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

    # Save model and tokenizer
    output_dir = "./sentiment_model_with_bert_bilstm"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model_bilstm.pt"))
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved to:", output_dir)

    # Evaluate the model
    evaluate(model, test_loader, device)
