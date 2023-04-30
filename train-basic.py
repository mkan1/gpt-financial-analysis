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

def buildInferenceModel():
    # Inference encoder model
    encoder_model = Model(encoder_input, encoder_states)

    # Inference decoder model
    decoder_state_input_h = Input(shape=(LSTM_UNITS,))
    decoder_state_input_c = Input(shape=(LSTM_UNITS,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

encoder_model, decoder_model = buildInferenceModel()

def predict(input_text):
    input_seq = pad_sequences(tokenizer.encode(input_text, add_special_tokens=False), maxlen=MAX_SEQ_LEN, padding="post")
    input_seq = np.reshape(input_seq, (1, MAX_SEQ_LEN))

    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)

    decoded_sentence = []

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer.convert_ids_to_tokens([sampled_token_index])[0]
        
        if sampled_token == tokenizer.sep_token or len(decoded_sentence) > MAX_SEQ_LEN:
            break

        decoded_sentence.append(sampled_token)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return " ".join(decoded_sentence)

input_text = """Company X released its financial report for the fiscal year 2022. The total revenue for the year was $12.5 billion, representing a 15% increase from the previous year. The company's gross profit margin improved to 48% from 45% in 2021. Operating expenses grew by 8%, reaching $3.2 billion. The net income for the year was $2.1 billion, a 20% increase compared to the previous year.

In 2022, Company X's R&D expenditure was $1.5 billion, accounting for 12% of total revenue. The company's sales and marketing expenses totaled $1.2 billion, or 9.6% of total revenue. Company X declared a dividend of $0.50 per share, a 25% increase from the $0.40 per share dividend in 2021.
. Example question 1.

How much did the dividend per share increase compared to the previous year?
"""
print(predict(input_text))

## Evaluation

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
def evaluate_model(model, tokenizer, X_test, y_test, max_seq_len):
    bleu_scores = []

    for i in range(len(X_test)):
        input_sequence = np.array([X_test[i]])
        generated_sequence = np.argmax(model.predict(input_sequence), axis=-1)

        # Convert token indices back to text
        reference = tokenizer.decode(y_test[i][1:]).split()  # Ground truth, excluding the start token
        candidate = tokenizer.decode(generated_sequence[0]).split()  # Generated output
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)

    avg_bleu_score = np.mean(bleu_scores)
    return avg_bleu_score

# Evaluate the model on the test data
avg_bleu_score = evaluate_model(model, tokenizer, X_test, y_test, MAX_SEQ_LEN)
print("Average BLEU score:", avg_bleu_score)
