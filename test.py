#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model, model_from_json, load_model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
import re
import numpy as np
import nltk

HIDDEN_UNITS = 64


class chatbot(object):
    model = None
    encoder_model = None
    decoder_model = None
    input_word2idx = None
    input_idx2word = None
    target_word2idx = None
    target_idx2word = None
    max_encoder_seq_length = None
    max_decoder_seq_length = None
    num_encoder_tokens = None
    num_decoder_tokens = None

    def __init__(self):
        self.input_word2idx = np.load('model/word-input-word2idx.npy').item()
        self.input_idx2word = np.load('model/word-input-idx2word.npy').item()
        self.target_word2idx = np.load('model/word-target-word2idx.npy').item()
        self.target_idx2word = np.load('model/word-target-idx2word.npy').item()
        context = np.load('model/word-context.npy').item()
        self.max_encoder_seq_length = context['encoder_max_seq_length']
        self.max_decoder_seq_length = context['decoder_max_seq_length']
        self.num_encoder_tokens = context['num_encoder_tokens']
        self.num_decoder_tokens = context['num_decoder_tokens']

        self.encoder_model = load_model('model/encoder-weights.h5')
        self.decoder_model = load_model('model/decoder-weights.h5')

    def reply(self, input_text):
        input_seq = []
        input_wids = []
        for word in nltk.word_tokenize(input_text.lower()):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_encoder_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_word2idx['<SOS>']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        self.decoder_model.layers[-2].reset_states(states=states_value)
        while not terminated:
            output_tokens = self.decoder_model.predict(target_seq)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != '<SOS>' and sample_word != '<EOS>':
                    target_text += ' ' + sample_word
            
            if sample_word == '<EOS>' or target_text_len >= self.max_decoder_seq_length:
                terminated = True 
            
            target_text = re.sub("i 'm", "I'm", target_text)
            target_text = re.sub("he 's", "he's", target_text)
            target_text = re.sub("do n't", "don't", target_text)
            target_text = re.sub("(:+\s?)+d", ":D", target_text)
            target_text = re.sub("(\s?)+'", "'", target_text)
            target_text = re.sub("i ", "I ", target_text)
            target_text = re.sub("(\s?)+,", ",", target_text)
            target_text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', target_text)                        
            target_text = re.sub("(:+\s?)+\)", ":)", target_text)
            target_text = re.sub("(;+\s?)+\)", ";)", target_text)
            target_text = re.sub("can ’ t", "can't", target_text)
            target_text = re.sub("ca n’t", "can't", target_text)
            target_text = re.sub("ca n't", "can't", target_text)
            target_text = re.sub("\( ", "(", target_text)
            target_text = re.sub(" \)", ")", target_text)
            target_text = re.sub("i'd", "I'd", target_text)
            target_text = re.sub("`` ", "", target_text)
            target_text = re.sub("''", "", target_text)
            target_text = re.sub(" ``", "", target_text)
            target_text = re.sub("\( ", "(", target_text)
            target_text = re.sub(" \)", ")", target_text)            
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1
        
        return target_text.strip('.')
        
    def test_run(self):        
        print(self.reply("where are you?"))
        print(self.reply("who are you?"))
        print(self.reply("that's not funny")) 
        print(self.reply("let's do something fun!"))
        print(self.reply("what's the meaning of life"))
        print(self.reply("I'm hungry can you order pizza"))
        print(self.reply("are you self-aware?"))
        print(self.reply("what do you think about singularity"))
        print(self.reply("why"))
        print(self.reply("humans and robots should work together to make the world a better place. what do you think"))
        
        


def main():
    model = chatbot()
    model.test_run()

if __name__ == '__main__':
    main()
