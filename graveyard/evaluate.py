import torch
import json
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration
from train import FinancialDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    # create attention masks
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float()
    
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

model_path = "t5_finance_9.pt"  # Path to your saved model
model_name = "t5-small"

# Make sure the model is initialized with the same architecture as before
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the state dict of the saved model
model.load_state_dict(torch.load(model_path))

# If you have a GPU available, move the model to GPU
if torch.cuda.is_available():
    model.to('cuda')

# Load validation data
with open('dataset/test.json') as f:
    valid_data_dict = json.load(f)

# Specify the size of the subset you want to sample. Using a smaller size due to compute constraints.
subset_size = 10

# Randomly sample a subset of your training data
valid_data_dict = random.sample(valid_data_dict, subset_size)

valid_dataset = np.array([
    {
        "context": '\n'.join(data['pre_text'] 
                                + data['post_text'] 
                                + [json.dumps(data['table_ori']), json.dumps(data['table'])]),
        "question": data['qa']['question'],
        "answer": data['qa']['answer']
    } for data in valid_data_dict
])

valid_dataset = FinancialDataset(tokenizer, valid_dataset.tolist())
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize the model for evaluation
model.eval()

# Move the model to the correct device
model.to(device)

# Initialize a variable to hold the total validation loss
total_val_loss = 0
bleu_score = 0
examples = []

# Loop over the validation data
for batch in valid_loader:
    # Move the batch tensors to the same device as the model
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Compute the model outputs
    with torch.no_grad():  # Deactivate gradients for the following block
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        predictions = model.generate(input_ids)

        # Convert predictions to text
        predicted_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in predictions]
        print(predicted_text)

        # Compute BLEU score
        references = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in labels]
        for i in range(len(predicted_text)):
            bleu_score += sentence_bleu([references[i].split()], predicted_text[i].split())

        # Store examples
        if len(examples) < 5:  # Change this number as needed
            for i in range(len(predicted_text)):
                examples.append((predicted_text[i]))

    # Accumulate the validation loss
    total_val_loss += loss.item()

# Compute the average validation loss and BLEU score
average_val_loss = total_val_loss / len(valid_loader)
average_bleu_score = bleu_score / (len(valid_loader) * valid_loader.batch_size)  # Check if this is the correct denominator for your case

print(f"Validation Loss: {average_val_loss}")
print(f"Average BLEU Score: {average_bleu_score}")
print(f"Example: {examples}")

