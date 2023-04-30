import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset (context, question, answer)
# Assuming your dataset is in the form of a list of dictionaries
f = open('dataset/train.json')
dict = json.load(f)


dataset = np.array(
    [
        {
            "context": '\n'.join(data['pre_text'] 
                                    + data['post_text'] 
                                    + [json.dumps(data['table_ori']), json.dumps(data['table'])]),
            "question":
                data['qa']['question'],
            "answer":
                data['qa']['answer']
} for data in dict])

# Define hyperparameters
EMBEDDING_DIM = 128
LSTM_UNITS = 128
MAX_SEQ_LEN = 512
BATCH_SIZE = 32
EPOCHS = 10

# Tokenize and preprocess data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, model_max_length=MAX_SEQ_LEN)

VOCAB_SIZE = tokenizer.vocab_size

def preprocess_data(data):
    input_sequences, output_sequences = [], []

# probably should add [SEP] tag between context and question
    for d in data:
        input_sequence = [tokenizer.cls_token_id] + tokenizer.encode(d["context"] + d["question"], add_special_tokens=False)[:MAX_SEQ_LEN]
        output_sequence = [tokenizer.cls_token_id] + tokenizer.encode(d["answer"], add_special_tokens=False) + [tokenizer.sep_token_id]
        input_sequences.append(input_sequence)
        output_sequences.append(output_sequence)

    return pad_sequences(input_sequences, maxlen=MAX_SEQ_LEN, padding="post"), pad_sequences(output_sequences, maxlen=MAX_SEQ_LEN + 1, padding="post")

X, y = preprocess_data(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train[:, :-1].shape)
print("y_train target shape:", y_train[:, 1:].shape)

# Encoder
encoder_input = Input(shape=(MAX_SEQ_LEN,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN, mask_zero=True)(encoder_input)
encoder_lstm = LSTM(LSTM_UNITS, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_input = Input(shape=(MAX_SEQ_LEN,))
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN, mask_zero=True)(decoder_input)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(VOCAB_SIZE, activation="softmax"))
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2seq model
model = Model([encoder_input, decoder_input], decoder_outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    [X_train, y_train[:, :-1]],
    y_train[:, 1:],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([X_test, y_test[:, :-1]], y_test[:, 1:]),
)