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
    this.chatTextBoxDiv = document.getElementById('chatbox-div');
    this.chatTextBox = document.getElementById('generated-chat');
    this.inputText = document.getElementById('input-text');
    this.inputText.onkeyup = (evt) => {
        evt.preventDefault();
        if (evt.keyCode === 13) {
            this.sendChat();
        } 
    }
    this.chatContent = [];

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
            return prediction.squeeze().argMax();
        });

        const outputToken = await outputTokenTensor.data();
        outputTokenTensor.dispose();
        const word = targetIdx2word[outputToken[0]];
        numPredicted++;
        nextTokenID = outputToken[0];
        console.log(outputToken, word);

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
        inputWordIds = inputWordIds.slice(0, 18);
    }
    console.log(inputWordIds);
    return tf.tensor2d(inputWordIds, [1, 18]);
  }

  convertTokensToSentence(tokens) {
      return tokens.join(' ');
  }

  updateChatbox(user, text) {
      this.chatContent.push({user, text});

      let textBoxString = '';
      for (let i = 0; i < this.chatContent.length; i++) {
          const content = this.chatContent[i].user === 'BOT' ? 
              this.applyOutputRegex(this.chatContent[i].text) : 
              this.chatContent[i].text;
          textBoxString +=
            this.chatContent[i].user + ': ' + content + '\n\n';
      }

      this.chatTextBox.innerText = textBoxString;
      console.log(this.chatContent);

      // needed only for style
      this.chatTextBoxDiv.hidden = false;
      this.chatTextBox.rows = this.chatContent.length*2+1 > 10 ? 
          10 : this.chatContent.length*2+1;
      this.chatTextBox.scrollTop = this.chatTextBox.scrollHeight;
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
      return text;
  }
}

window.addEventListener('load', () => new Main());
