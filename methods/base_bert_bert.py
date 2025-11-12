import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_0 = time.time()
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


def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def eval_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=["neg", "neu", "pos"])
    return accuracy, report


def bert_bert(dataset_path, save_path, num_epochs=10):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    data = pd.read_csv(dataset_path)
    data['label'] = data['label'].map({"neg": 0, "neu": 1, "pos": 2})

    # spliting
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['processed'].values,
        data['label'].values,
        test_size=0.2,
        random_state=42
    )

    # initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # dataset loader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    from transformers import AdamW

    optimizer = AdamW(model.parameters(), lr=2e-5)

    t_l = []
    # train
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Training loss: {train_loss:.4f}")
        t = time.time()
        t_l.append(t)


    # eval
    accuracy, report = eval_model(model, test_loader)
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    for t in range(num_epochs):
        t_1 = t_l[t]
        print('epoch', t+1, ' ,cost: ', t_1-t_0)


if __name__ == '__main__':
    # main(dataset_path="G://Coding//PycharmProject//pusht_formal//data//COVIDSenti_A_base.csv",
    #      save_path="sentiment_model_A_base")
    bert_bert(dataset_path="G://Coding//PycharmProject//pusht_formal//data//preprocessed//COVIDSenti-A_12000_stem.csv",
              save_path="stemmer_lemma_12000//1031_A_base_stem", num_epochs=15)
    bert_bert(dataset_path="G://Coding//PycharmProject//pusht_formal//data//preprocessed//COVIDSenti-A_12000_lemma.csv",
              save_path="stemmer_lemma_12000//1031_A_base_lemma", num_epochs=15)