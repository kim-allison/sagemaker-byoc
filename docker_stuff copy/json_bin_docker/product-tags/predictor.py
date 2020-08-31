from __future__ import absolute_import, division, print_function, unicode_literals
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from model.net_pb import KobertSequenceFeatureExtractor, KobertCRF, KobertBiLSTMCRF, KobertBiGRUCRF

import os
import json
import pickle
import flask
import torch
from gluonnlp.data import SentencepieceTokenizer
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from data_utils.utils import Config
from decode import DecoderFromNamedEntitySequence 
from torchcrf import CRF

import tensorflow.compat.v1 as tf
from flask import jsonify

'''
WARNINGS:
/usr/local/lib/python3.6/dist-packages/konlpy/tag/_okt.py:16: UserWarning: "Twitter" has changed to "Okt" since KoNLPy v0.4.5.
  warn('"Twitter" has changed to "Okt" since KoNLPy v0.4.5.')
/usr/local/lib/python3.6/dist-packages/konlpy/tag/_okt.py:16: UserWarning: "Twitter" has changed to "Okt" since KoNLPy v0.4.5.
  warn('"Twitter" has changed to "Okt" since KoNLPy v0.4.5.')
/usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:60: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.1 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
'''

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
model_config = Config(json_path="config.json")

class ModelHandler(object):
    model = None
    tokenizer = None
    ner_to_index = None
    vocab = None
    
    ner_to_index = None
    index_to_ner = None
    token_to_index = None
    index_to_token = None

    @classmethod
    def get_vocab(cls):
        if cls.vocab == None:
            with open("vocab.pkl", 'rb') as f:
                cls.vocab = pickle.load(f)
        return cls.vocab

    @classmethod
    def get_tokenizer(cls):
        if cls.tokenizer == None:
            tok_path = "./tokenizer_78b3253a26.model"
            ptr_tokenizer = SentencepieceTokenizer(tok_path)

            cls.tokenizer = Tokenizer(vocab=cls.vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
        return cls.tokenizer

    @classmethod
    def get_something_to_index(cls):
        if cls.ner_to_index == None:
            with open("ner_to_index.json", 'rb') as f:
                cls.ner_to_index = json.load(f)
                cls.index_to_ner = {v: k for k, v in cls.ner_to_index.items()}

            with open("token2idx_vocab.json", 'rb') as f:
                cls.token_to_index = json.load(f)
                cls.index_to_token = {str(v): k for k, v in cls.token_to_index.items()}
        return [cls.ner_to_index, cls.index_to_ner, cls.token_to_index, cls.index_to_token]

    @classmethod
    def get_model(cls):
        if cls.model == None:
            model = KobertBiLSTMCRF(config=model_config, num_classes=len(cls.ner_to_index), vocab=cls.vocab)
            model_dict = model.state_dict()
            # 여기!
            checkpoint = torch.load(os.path.join(model_path, "best-epoch-387-step-5800-acc-0.978.bin"), map_location=torch.device('cpu'))

            convert_keys = {}
            for k, v in checkpoint['model_state_dict'].items():
                new_key_name = k.replace("module.", '')
                if new_key_name not in model_dict:
                    print("{} is not int model_dict".format(new_key_name))
                    continue
                convert_keys[new_key_name] = v

            model.load_state_dict(convert_keys)
            model.eval()
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to('cpu')
            cls.model = model
        return cls.model
        
app = flask.Flask(__name__) # Flask app

@app.route('/ping', methods=['GET'])
def ping(): # Check health
    health = (ModelHandler.get_vocab() is not None) and (ModelHandler.get_tokenizer() is not None) and (ModelHandler.get_something_to_index() is not None) and  (ModelHandler.get_model() is not None)
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation(): # Do an inference on a single batch of data
    # data = None

    if flask.request.content_type == 'application/json':

        vocab = ModelHandler.get_vocab()
        tokenizer = ModelHandler.get_tokenizer()
        ner_to_index, index_to_ner, token_to_index, index_to_token = ModelHandler.get_something_to_index()

        model = ModelHandler.get_model()

        decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

        input_json = flask.request.get_json() # Get input data (key = "products")
        input_data = input_json['products']

        output_json = {}
        output_json['tags'] = []

        for input_text in input_data:
            list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
            x_input = torch.tensor(list_of_input_ids).long()

            emission = model(x_input, using_pack_sequence=False) # Make predictions
            num_classes = len(ner_to_index)
            crf = CRF(num_tags=num_classes, batch_first=True) # 순서 (rearrange tag sequences)
            list_of_pred_ids = crf.decode(emission)

            input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=False)
            unkTokenList = makeUNKTokenList(input_text, input_token)
            input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=unkTokenList)
            
            output_json['tags'].append(decoding_ner_sentence[6:-5])

        return jsonify(output_json)
    else:
        return flask.Response(response='._.', status=415, mimetype='application/json')

def makeUNKTokenList(input_text, input_token):
    # 예외 특수문자 처리
    if 'Ⅱ' in input_text:
        input_text = input_text.replace('Ⅱ', 'II')
    # 아래 조건문에 있는 ' ' 에 공백이 아닌 특수문자가 들어가 있습니다.
    if ' ' in input_text:
        input_text = input_text.replace(' ', ' ')
    if '＋' in input_text:
        input_text = input_text.replace('＋', '+')
    if '™' in input_text:
        input_text = input_text.replace('™', 'TM')

    # 본 함수 시작
    input_text = input_text.strip().replace(' ', '')
    input_token = ''.join(input_token[1:-1]).replace('▁', '')

    unkTokenList = list()
    for idx in range(len(input_text)):
        if input_text[idx] != input_token[idx]:
            unkTokenList.append(input_text[idx])
            input_token = input_token[:idx] + input_text[idx] + input_token[idx+5:]
    return unkTokenList