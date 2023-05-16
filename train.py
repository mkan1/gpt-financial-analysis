import torch
import json
import random
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class FinancialDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):  # Reduced max_length
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['context'] + ' ' + item['question']
        target_text = item['answer']
        source = self.tokenizer.encode_plus(input_text, max_length=self.max_length, padding='max_length', return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.max_length, padding='max_length', return_tensors="pt")

        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze(),
            'target_attention_mask': target['attention_mask'].squeeze()
        }

def train():
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        # create attention masks
        attention_mask = input_ids.ne(tokenizer.pad_token_id).float()
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # Load data
    with open('dataset/train.json') as f:
        data_dict = json.load(f)

    # Specify the size of the subset you want to sample. Using a smaller size due to compute constraints.
    subset_size = 100

    # Randomly sample a subset of your training data
    data_dict = random.sample(data_dict, subset_size)

    dataset = np.array([
        {
            "context": '\n'.join(data['pre_text'] 
                                    + data['post_text'] 
                                    + [json.dumps(data['table_ori']), json.dumps(data['table'])]),
            "question": data['qa']['question'],
            "answer": data['qa']['answer']
        } for data in data_dict
    ])

    model_name = "t5-small"  # Using smaller T5 model due to compute constraints

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    dataset = FinancialDataset(tokenizer, dataset.tolist())
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn = collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    for epoch in range(10):
        # Create a progress bar
        progress_bar = tqdm(loader, desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        # Save the model after each epoch
        torch.save(model.state_dict(), f't5_finance_{epoch}.pt')
