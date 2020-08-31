# ''': FIX
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

import tensorflow.compat.v1 as tf   # BERT?
# import protobuf

# Model
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

class ModelHandler(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model == None:
            path_to_graph = os.path.join(model_path, 'output.pb')
            NER_graph = tf.Graph()

            with NER_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(path_to_graph, 'rb') as f:
                    serialized_graph = f.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name="") 

            cls.model = NER_graph
        return cls.model

    # Move 'make prediction' function here (as a class method)

# Flask app
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping(): # Check health
    health = ModelHandler.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='text/plain')

@app.route('/invocations', methods=['POST'])
def transformation(): # Do an inference on a single batch of data
    data = None

    # 1) INPUT: convert Korean text input to NER code array
    if flask.request.content_type == 'text/plain':

        '''CHECK file locations'''
        model_config = Config(json_path="config.json")
        tok_path = "./tokenizer_78b3253a26.model"
        ptr_tokenizer = SentencepieceTokenizer(tok_path)

        with open("vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)

        tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

        with open("ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

        f = flask.request.get_data()
        # ftype = str(type(f))
        string_f = f.decode("utf-8") 
        lines = string_f.splitlines(True)

        with open("result.txt", 'w', encoding='utf-8-sig') as w:
            # w.write('start\n')
            # w.write(ftype)
            # w.write('\nand\n')
            # w.write(string_f)
            # w.write('\nend\n')
            index = 0
            for i in range(len(lines)):
                input_text = ''
                if i% 4 == 1:
                    input_text = lines[i][3:]
                    addInfo = lines[i+1][3:]
                if input_text == '':
                    continue

                index += 1
                # print("\n## " + str(index) + "\n")

                list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
                x_input = torch.tensor(list_of_input_ids).long()

                w.write('## '+str(index)+'\n')
                w.write(addInfo)
                # w.write('\n'+str(list_of_input_ids))

                predictions = run_inference_for_single_data(list_of_input_ids[0], ModelHandler.get_model())  
                
                # 2) OUTPUT: convert NER code to Korean text (FILE)
                emission = torch.tensor(predictions['output'])
                num_classes = len(ner_to_index)
                crf = CRF(num_tags=num_classes, batch_first=True) # 순서 (rearrange tag sequences)
                list_of_pred_ids = crf.decode(emission)

                input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=False)
                unkTokenList = makeUNKTokenList(input_text, input_token)
                input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=unkTokenList)
                
                w.write(str(list_of_ner_word) + '\n')
                w.write(str(decoding_ner_sentence[6:-5]) + '\n')

        return flask.Response(response=open("result.txt", 'r', encoding='utf-8-sig'), status=200, mimetype='text/plain')
    else:
        return flask.Response(response='This predictor only supports TEXT data', status=415, mimetype='text/plain')

def run_inference_for_single_data(data, NER_graph):
    with tf.Session(graph=NER_graph) as sess:

        input_tensor = NER_graph.get_tensor_by_name('input:0')
        target_operation_names = ['output']
        tensor_dict = {}
        for key in target_operation_names:
            op = None
            try:
                op = NER_graph.get_operation_by_name(key)
            except:
                continue

            tensor = NER_graph.get_tensor_by_name(op.outputs[0].name)
            tensor_dict[key] = tensor

        output_dict = sess.run(tensor_dict, feed_dict={input_tensor: [data]})

        return output_dict

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