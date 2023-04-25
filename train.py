import json
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFDistilBertForQuestionAnswering, AutoTokenizer

f = open('dataset/train.json')
dict = json.load(f)


arr = np.array(
    [
        [
        '\n'.join(data['pre_text'] + data['post_text']),
        json.dumps(data['table_ori']),
        json.dumps(data['table']),
        data['qa']['question'],
        data['qa']['answer']
    ] for data in dict])

X = arr[:, :-1] # input matrix for training data
Y = arr[:, -1]  # output matrix for training data

# Choose a pre-trained tokenizer based on the model you want to use
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Concatenate the elements along the second dimension to create a single string for each data point
X_text = [' '.join(item) for item in X]

# Tokenize and encode input data
X_encoded = [tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=512) for text in X_text]

# Tokenize and encode output data
Y_tokenized = [tokenizer.tokenize(answer) for answer in Y]
Y_encoded = [tokenizer.convert_tokens_to_ids(tokens) for tokens in Y_tokenized]

# Pad the output sequences
max_output_length = max([len(answer) for answer in Y_encoded])
Y_padded = np.zeros((len(Y_encoded), max_output_length))
for i, answer in enumerate(Y_encoded):
    Y_padded[i, :len(answer)] = answer

# Check pre-processing
print("Encoded input shape:", X_encoded[0].input_ids.shape)
print("Padded output shape:", Y_padded.shape)

random_index = random.randint(0, len(X) - 1)

decoded_input = tokenizer.decode(X_encoded[random_index].input_ids.squeeze())
print("Original input text:\n", X[random_index])
print("Decoded input text:\n", decoded_input)

decoded_output = tokenizer.decode(Y_padded[random_index, :].astype(int))
print("Original output text:\n", Y[random_index])
print("Decoded output text:\n", decoded_output)


# Split the data into training and validation sets
X_train_encoded, X_val_encoded, Y_train_padded, Y_val_padded = train_test_split(X_encoded, Y_padded, test_size=0.2, random_state=42)

# Choose a pre-trained model based on the model you want to use
model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

# Convert lists of dicts to dicts of lists
X_train_encoded_dict = {key: np.stack([x[key] for x in X_train_encoded], axis=0) for key in X_train_encoded[0].keys()}
X_val_encoded_dict = {key: np.stack([x[key] for x in X_val_encoded], axis=0) for key in X_val_encoded[0].keys()}

# Define the optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss)

# Train the model
history = model.fit(X_train_encoded_dict, Y_train_padded, epochs=3, batch_size=8, validation_data=(X_val_encoded_dict, Y_val_padded))
