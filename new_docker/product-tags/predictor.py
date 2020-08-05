# ''': FIX
import os
import json
import pickle
import flask

# import tarfile
import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker import get_execution_role

import torch
from gluonnlp.data import SentencepieceTokenizer
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from data_utils.utils import Config
from decode import DecoderFromNamedEntitySequence 
from torchcrf import CRF
'''
    Above files-need to put them in a specific location-WHERE?
    opt/ml의 어디? 
    > 지금은 /opt/program에 (product-tags)
'''

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Model (load+hold model)
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls): # Load
        if cls.model == None:
            # INPUT_TENSOR_NAME = 'inputs'
            # exported_model = classifier.export_savedmodel(export_dir_base = 'export/Servo', serving_input_reciever_fn = serving_input_fn)
            # with tarfile.open(os.path.join(model_path, 'model4.tar.gz'), mode='w:gz') as inp: 
            #     '''
            #         opt/ml/model에 있는 어떤 type의 model?
            #         https://aws.amazon.com/blogs/machine-learning/bring-your-own-pre-trained-mxnet-or-tensorflow-models-into-amazon-sagemaker/
            #     '''
            #     # tar = tarfile.open(inp, "r:gz")
            #     # tar.extractall()
            #     # tar.close()
            #     archive.add('export', recursive=True)

            role = 'AmazonSageMaker-ExecutionRole-20200615T164342'
            # role = get_execution_role()

            # sagemaker_session = sagemaker.Session()
            # inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')
            sagemaker_model = TensorFlowModel(model_data = 's3://sagemaker-bucket-cj2/model/model4.tar.gz',
                                  role = role,
                                  container_log_level=20,
                                  name='DP-MODEL5',
                                  framework_version='1.15'
                                  )
            predictor = sagemaker_model.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
            cls.model = predictor
        return cls.model

    def serving_input_fn():
        feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    
    @classmethod
    def predict(cls, input):
        '''
        Args: 
            input (application/json): the data on which to do the predictions. There will be one prediction per array.
        '''
        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping(): # Check health
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

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

        '''
            Assuming request.data is a string: name of txt file
            > NER_OY_data.txt as an example
            > 지금은 /opt/program에 (product-tags)

            HERE:?
        '''
        f = flask.request.data.decode("utf-8") 
        lines = f.splitlines(True)
        index = 0

        with open("NER_OY_result.txt", 'w', encoding='utf-8-sig') as w:
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
                # print(list_of_input_ids)
                # print(x_input)

                data = {"instances": list_of_input_ids}
                predictions = ScoringService.predict(data)

                # 2) OUTPUT: convert NER code to Korean text (FILE)
                emission = torch.tensor(predictions['predictions'])
                num_classes = len(ner_to_index)
                crf = CRF(num_tags=num_classes, batch_first=True)
                list_of_pred_ids = crf.decode(emission)

                input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=False)
                unkTokenList = makeUNKTokenList(input_text, input_token)
                input_token, list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids, unkTokenList=unkTokenList)

                w.write('## '+str(index)+'\n')
                w.write(addInfo)
                w.write(str(list_of_ner_word) + '\n')
                w.write(str(decoding_ner_sentence[6:-5]) + '\n')
        
            '''RETURN a file: NER_OY_result.txt'''
        return flask.Response(response=open("NER_OY_result.txt", 'r'), status=200, mimetype='text/plain')
    else:
        return flask.Response(response='This predictor only supports TEXT data', status=415, mimetype='text/plain')

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