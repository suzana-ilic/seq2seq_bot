import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';
import wordTokenize from './tokenizer';
import inputWord2idx from './mappings/input-word2idx';
import wordContext from './mappings/word-context';
import targetWord2idx from './mappings/target-word2idx';
import targetIdx2word from './mappings/target-idx2word';

/**
 * Main application to start on window load
 */
class Main {
  /**
   * Constructor creates and initializes the variables needed for
   * the application
   */
  constructor() {
    // Initiate variables
    this.conversationEle = document.getElementById('conversation');
    this.inputText = document.getElementById('input-text');
    this.modelLoadAnimEle = document.getElementById('model-load-anim');
    this.inputText.onkeyup = (evt) => {
        evt.preventDefault();
        if (evt.keyCode === 13) {
            this.sendChat();
        } 
    }
    this.chatContent = [];
    this.temperature = 0;
    this.temperatureSlider = document.getElementById('slider-range');
    this.temperatureDisplay = document.getElementById('temperature-display');
    this.temperatureSlider.oninput = (evt) => {
        this.temperature = parseFloat(evt.target.value);
        this.temperatureDisplay.textContent = this.temperature;
    }
    this.temperatureDisplay.textContent = this.temperature;
    this.temperatureSlider.value = this.temperature;

    Promise.all([
        tf.loadModel('decoder-model/model.json'),
        tf.loadModel('encoder-model/model.json'),
    ]).then(([decoder, encoder]) => {
        this.decoder = decoder;
        this.encoder = encoder;
        this.enableGeneration();
    });
  }

  /**
   * Called after model has finished loading or generating.
   * Sets up UI elements for generating text.
   */
  enableGeneration() {
    this.inputText.placeholder = "Sarcastobot is live. Start typing..";
    this.inputText.disabled = "";
    this.conversationEle.classList.add('ready');
  }

  async sendChat() {
    let inputText = this.inputText.value;
    this.inputText.value = '';
    this.updateChatbox('YOU', inputText);

    const states = tf.tidy(() => {
        const input = this.convertSentenceToTensor(inputText);
        return this.encoder.predict(input);
    });

    this.decoder.layers[1].resetStates(states);

    let responseTokens = [];
    let terminate = false;
    let nextTokenID = targetWord2idx['<SOS>'];
    let numPredicted = 0;
    while (!terminate) {
        const outputTokenTensor = tf.tidy(() => {
            const input = this.generateDecoderInputFromTokenID(nextTokenID);
            const prediction = this.decoder.predict(input);
            return this.sample(prediction.squeeze());
        });

        const outputToken = await outputTokenTensor.data();
        outputTokenTensor.dispose();
        nextTokenID = Math.round(outputToken[0]);
        const word = targetIdx2word[nextTokenID];
        numPredicted++;
        console.log(outputToken, nextTokenID, word);

        if (word !== '<EOS>' && word !== '<SOS>') {
            responseTokens.push(word);
        }

        if (word === '<EOS>'
            || numPredicted >= wordContext.decoder_max_seq_length) {
            terminate = true;
        }

        await tf.nextFrame();
    }

    this.updateChatbox('BOT', this.convertTokensToSentence(responseTokens));

    states[0].dispose();
    states[1].dispose();
  }

  generateDecoderInputFromTokenID(tokenID) {
      const buffer = tf.buffer([1, 1, wordContext.num_decoder_tokens]);
      buffer.set(1, 0, 0, tokenID);
      return buffer.toTensor();
  }

  /**
   * Randomly samples next word weighted by model prediction.
   */
  sample(prediction) {
    return tf.tidy(() => {
      if (this.temperature == 0) {
        return prediction.argMax();
      }
      if (this.temperature == 1) {
          return tf.randomUniform(prediction.shape).argMax();
      }
      const temperature = tf.scalar(this.temperature);
      prediction = prediction.div(temperature);
      prediction = prediction.exp();
      prediction = prediction.div(prediction.sum());
      prediction = prediction.mul(tf.randomUniform(prediction.shape));
      return prediction.argMax();
    });
  }

  convertSentenceToTensor(sentence) {
    let inputWordIds = [];
    wordTokenize(sentence).map((x) => {
        x = x.toLowerCase();
        let idx = '1';
        if (x in inputWord2idx) {
            idx = inputWord2idx[x];
        }
        inputWordIds.push(Number(idx));
    });
    if (inputWordIds.length < wordContext.encoder_max_seq_length) {
        inputWordIds =
            Array.concat(
                new Array(
                    wordContext.encoder_max_seq_length-inputWordIds.length+1)
                    .join('0').split('').map(Number),
                inputWordIds
            );
    } else {
        inputWordIds = inputWordIds.slice(0, wordContext.encoder_max_seq_length);
    }
    console.log(inputWordIds);
    return tf.tensor2d(inputWordIds, [1, wordContext.encoder_max_seq_length]);
  }

  convertTokensToSentence(tokens) {
      return tokens.join(' ');
  }

  updateChatbox(user, text) {
    const row = document.createElement('div');
    row.classList.add('conversation__row');
    row.classList.add(
      user == 'BOT' ? 'conversation__row--bot' : 'conversation__row--you');
    const bubble = document.createElement('div');
    bubble.className = 'conversation__bubble';
    bubble.textContent = user == 'BOT' ? this.applyOutputRegex(text) : text;
    row.appendChild(bubble);
    this.conversationEle.appendChild(row);
    this.conversationEle.scrollTop = this.conversationEle.scrollHeight;
  }

  applyOutputRegex(text) {
      text = text.replace(/i 'm/g, "I'm");
      text = text.replace(/he 's/g, "he's");
      text = text.replace(/do n't/g, "don't");
      text = text.replace(/(:+\s?)+d/g, ":D");
      text = text.replace(/(\s?)+'/g, "'");
      text = text.replace(/i /g, "I ")
      text = text.replace(/(\s?)+,/g, ",");
      text = text.replace(/\s([?.!"](?:\s|$))/g, "$1");
      text = text.replace(/(:+\s?)+\)/g, ":)");
      text = text.replace(/(;+\s?)+\)/g, ";)");
      text = text.replace(/can ’ t/g, "can't");
      text = text.replace(/"ca n’t/g, "can't");
      text = text.replace(/ca n't/g, "can't");
      text = text.replace(/\( /g, "(");
      text = text.replace(/ \)/g, ")");
      text = text.replace(/i'd/g, "I'd");
      text = text.replace(/`` /g, "");
      text = text.replace(/''/g, "");
      text = text.replace(/ ``/g, "");
      return text;
  }
}

window.addEventListener('load', () => new Main());
