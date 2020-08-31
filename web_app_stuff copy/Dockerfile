FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY ./app /app
RUN pip3 install --upgrade pip
RUN pip3 install boto3 sagemaker awscli
RUN aws configure set region ap-northeast-2
RUN aws configure set aws_access_key_id AKIA5SIZZPEDDP5NHPWG
RUN aws configure set aws_secret_access_key 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