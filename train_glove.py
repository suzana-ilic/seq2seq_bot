#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import LSTM, Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
import numpy as np
import pandas as pd
import re
import json

np.random.seed(42)

def store_js(filename, data):
    with open(filename, 'w') as f:
        f.write('export default ' + json.dumps(data, indent=2))

BATCH_SIZE = 32
NUM_EPOCHS = 200
HIDDEN_UNITS = 64
MAX_INPUT_SEQ_LENGTH = 17
MAX_TARGET_SEQ_LENGTH = 27
MAX_VOCAB_SIZE = 2000
QUESTIONS = 'data/Q1.csv'
ANSWERS = 'data/Q2.csv'
WEIGHT_FILE_PATH = 'model/glove-word-weights.h5'

input_counter = Counter()
target_counter = Counter()

input_texts = []
target_texts = []

# loading data
with open('data/Q1.csv', 'r', encoding='utf8') as f:
    QUESTIONS = f.read().split('\n')
    
with open('data/Q2.csv', 'r', encoding='utf8') as f:
    ANSWERS = f.read().split('\n')


prev_words = []
for line in QUESTIONS:
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    if len(prev_words) > 0:
        input_texts.append(prev_words)
        for w in prev_words:
            input_counter[w] += 1

    prev_words = next_words

prev_words = []
for line in ANSWERS:
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    if len(prev_words) > 0:
        target_words = next_words[:]
        target_words.insert(0, '<SOS>')
        target_words.append('<EOS>')
        for w in target_words:
            target_counter[w] += 1
        target_texts.append(target_words)

    prev_words = next_words
    
input_word2idx = {}
target_word2idx = {}
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_word2idx[word[0]] = idx + 2
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1
    

input_word2idx['<PAD>'] = 0
input_word2idx['<UNK>'] = 1
target_word2idx['<UNK>'] = 0

input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)

np.save('model/glove-word-input-word2idx.npy', input_word2idx)
np.save('model/glove-word-input-idx2word.npy', input_idx2word)
np.save('model/glove-word-target-word2idx.npy', target_word2idx)
np.save('model/glove-word-target-idx2word.npy', target_idx2word)

# Store necessary mappings for tfjs
store_js('js/mappings/glove-input-word2idx.js', input_word2idx)
store_js('js/mappings/glove-input-idx2word.js', input_idx2word)
store_js('js/mappings/glove-target-word2idx.js', target_word2idx)
store_js('js/mappings/glove-target-idx2word.js', target_idx2word)

encoder_input_data = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in input_word2idx:
            w2idx = input_word2idx[w]
        encoder_input_wids.append(w2idx)

    encoder_input_data.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print(context)
np.save('model/glove-word-context.npy', context)
store_js('js/mappings/glove-word-context.js', context)

def generate_batch(input_data, output_text_data):
    num_batches = len(input_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = 0  # default [UNK]
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


# embedding matrx
embeddings_index = {}
f = open('data/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

count = 0
embedding_matrix = np.zeros((len(input_word2idx) + 1, 100))
for word, index in input_word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    else:
        count += 1
print("Unknown: ", count) 

# embedding layer
embedding_layer = Embedding(len(input_word2idx) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=encoder_max_seq_length,
                            trainable=False) 


encoder_inputs = Input(shape=(None,), name='encoder_inputs')
#encoder_embedding = embedding_layer
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(embedding_layer(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#early stopping on valid perplexity
from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_ppx', patience=2)

# perplexity
from keras.losses import categorical_crossentropy
from keras import backend as K
import math

def ppx(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return perplexity

optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[ppx])


json = model.to_json()
open('model/glove-word-architecture.json', 'w').write(json)

X_train, X_test, y_train, y_test = train_test_split(encoder_input_data, target_texts, test_size=0.05, random_state=42)

print(len(X_train))
print(len(X_test))

train_gen = generate_batch(X_train, y_train)
test_gen = generate_batch(X_test, y_test)

train_num_batches = len(X_train) // BATCH_SIZE
test_num_batches = len(X_test) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint,callback])

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('model/glove-encoder-weights.h5')

new_decoder_inputs = Input(batch_shape=(1, None, num_decoder_tokens), name='new_decoder_inputs')
new_decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='new_decoder_lstm', stateful=True)
new_decoder_outputs, _, _ = new_decoder_lstm(new_decoder_inputs)
new_decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='new_decoder_dense')
new_decoder_outputs = new_decoder_dense(new_decoder_outputs)
new_decoder_lstm.set_weights(decoder_lstm.get_weights())
new_decoder_dense.set_weights(decoder_dense.get_weights())

new_decoder_model = Model(new_decoder_inputs, new_decoder_outputs)

new_decoder_model.save('model/glove-decoder-weights.h5')

