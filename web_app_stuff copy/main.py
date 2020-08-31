import sys
import flask, json, boto3, sagemaker
from flask import render_template, jsonify

ENDPOINT_NAME = 'endpoint'
client = boto3.client('runtime.sagemaker')

app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    output = []

    if flask.request.method == 'POST':
        file_json = flask.request.files['file'].read()

        try:
            real_json = file_json.decode('utf8').replace("'", '"')
            data = json.loads(real_json)
        except ValueError as e:
            return render_template('output.html', output=output)

        response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Accept='application/json', Body=json.dumps(data))
        response_body = response['Body']
        response_dic = json.loads(response_body.read())

        product_num = len(data["products"])
        for i in range(0, product_num):
            output.append({'input': data["products"][i], 'output': response_dic["tags"][i]})

    return render_template('output.html', output=output)