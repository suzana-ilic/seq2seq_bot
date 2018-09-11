#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
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


def store_js(filename, data):
    with open(filename, 'w') as f:
        f.write('export default ' + json.dumps(data, indent=2))

np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 50
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 18
MAX_TARGET_SEQ_LENGTH = 18
MAX_VOCAB_SIZE = 800
DATA_PATH = 'data/bot_data.tsv'
WEIGHT_FILE_PATH = 'model/word-weights.h5'

input_counter = Counter()
target_counter = Counter()
input_texts = []
target_texts = []

# loading data

with open(DATA_PATH, 'r', encoding='utf8') as f:
    lines = f.read().split('\n')

# pre-processing

prev_words = []
for line in lines:
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    

    if len(prev_words) > 0:
        input_texts.append(prev_words)
        for w in prev_words:
            input_counter[w] += 1

        target_words = next_words[:]
        target_words.insert(0, '<SOS>')
        target_words.append('<EOS>')
        for w in target_words:
            target_counter[w] += 1
        target_texts.append(target_words)

    prev_words = next_words

    
input_word2idx = dict()
target_word2idx = dict()
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

np.save('model/word-input-word2idx.npy', input_word2idx)
np.save('model/word-input-idx2word.npy', input_idx2word)
np.save('model/word-target-word2idx.npy', target_word2idx)
np.save('model/word-target-idx2word.npy', target_idx2word)

# Store necessary mappings for tfjs
store_js('js/mappings/input-word2idx.js', input_word2idx)
store_js('js/mappings/input-idx2word.js', input_idx2word)
store_js('js/mappings/target-word2idx.js', target_word2idx)
store_js('js/mappings/target-idx2word.js', target_idx2word)

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
np.save('model/word-context.npy', context)
store_js('js/mappings/word-context.js', context)

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


encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                              input_length=encoder_max_seq_length, name='encoder_embedding')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

json = model.to_json()
open('model/word-architecture.json', 'w').write(json)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_input_data, target_texts, test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('model/encoder-weights.h5')

new_decoder_inputs = Input(batch_shape=(1, None, num_decoder_tokens), name='new_decoder_inputs')
new_decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='new_decoder_lstm', stateful=True)
new_decoder_outputs, _, _ = new_decoder_lstm(new_decoder_inputs)
new_decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='new_decoder_dense')
new_decoder_outputs = new_decoder_dense(new_decoder_outputs)
new_decoder_lstm.set_weights(decoder_lstm.get_weights())
new_decoder_dense.set_weights(decoder_dense.get_weights())

new_decoder_model = Model(new_decoder_inputs, new_decoder_outputs)

new_decoder_model.save('model/decoder-weights.h5')
