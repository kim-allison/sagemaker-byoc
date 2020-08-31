[COMPILED 정리]

[1] SageMaker Endpoint Deployment via TensorFlow Serving 구성
Prediction 만들기: Client application > POST -> /invocations -> predictions: use AWS SDK
확인: SageMaker > /ping -> container (check if the container is working)

Step 1. NGINX
AWS SageMaker -> NGINX -> TensorFlow Serving: reverse proxy
1)	AWS Sagemaker -> NGINX
Ex) listen 8080 deferred;   # Configures the server to listen to the port 8080

2)	NGINX -> TensorFlow Serving
Ex) location /invocations { proxy pass http://localhost:8501/v1/models/half_plus_three:predict; }
# Redirects requests from SageMaker to TF Serving

요약: NGINX redirects requests in the format {address}:8080/invocations to 
http://localhost:8501/v1/models/helf_plus_three:predict, which is the request format expected by TF serving.

Step 2. Docker
To be able to serve using AWS SageMkaker, a container needs to implement a web server 
that handles the requests /invocations and /ping, on port 8080.

Step 3. ECR
Push Docker image to AWS ECR repository.

Step 4. SageMaker Endpoint
Deploy.

[2] Tests
Step1과 2에 대한 테스트들 (모든 test들의 nginx.conf는 동일 (1). 모든 Docker 이미지는 Linux에서 run (2))
See: https://medium.com/ml-bytes/how-to-create-a-tensorflow-serving-container-for-aws-sagemaker-4853842c9751
< where I got the framework for the nginx.conf file (modified to apply my model's name)

*중요1: TensorFlow model이 형식이 Saved Model이어야함. Saved Model에 관한 내용은 SavedModel.py (영어) 참고.
*중요2: 모든 Dockerfile안에 model 이름은 '--model_name=half_plus_three'로 'half_plus_three'라고 hard coded in.
-> /invocation을 부르면 (curl하는 CMD step), 이런 에러가 뜰 수 있음: '... Latest(some_model_name) ...'
-> 그러면 Dockerfile 안에서 '--model_name=some_model_name'으로 바꿔서 시도.
(??? 이유는 모르겠지만 처음 쓴 model 이름을 계속 써야함 ???)
*중요3: model 이름이 바뀌면 nginx.conf에서도 error가 뜰 수 밖에 없음. line 15을 보면 model PATH url에 
지금 사용하고 있는 model 이름 'half_plus_three'가 적용되어 있음. 중요2와 같이 고쳐야함.
*중요4: Test C 진행하려면 AWS Access Key ID와 Secret Access Key가 필요함.

A. LOCAL 'half_plus_three', sample model from tensorflow/serving GitHub repo
(input_num * 0.5 + 3를 prediction으로 output함)

/saved_model_half_plus_three/ (각 파일에 관한 내용은 파일 안 comments에 있음)
└── saved_model_half_plsu_three/
    └── 00000123/
        └── assets/
            └── foo
        └── variables/                      
            └── variables.data-?????-of-?????
            └── variables.index
        └── saved_model.pb  
└── Dockerfile
└── nginx.conf

CMD: (saved_model_half_plus_three parent directiory에 있다고 가정)
docker build -t testing .
docker run --rm -p 8080:8080 testing   
# 'NET_LOG: Entering the event loop ...' 또는 'Exporting HTTP/REST API at:localhost:8501'라면 성공!
curl -X POST http://localhost:8080/invocations -d "{\"instances\": [0.1,0.2,0.3]}"  # Windows input format

OUTPUT:
{"predictions": [3.05, 3.1, 3.15]}

B. LOCAL S3에 있던 'model2' (Saved Model 형식 'model2')

/model2_to_test/ (파일 안 comments 참고)
└── Dockerfile
└── model2_to_test.tar.gz
└── nginx.conf

*특이사항: A과 다르게 model만 stored된 directory가 없다. Dockerfile의 line 6를 보면 
'ADD model2_to_test.tar.gz /model2_to_test'라는 command가 tar.gz를 unzip하는 과정에서 container내에 
model2_to_test라는 directory가 생성되고 그 안에 또 새로 생긴 model이 version directory (00000001)안에
model이 stored됨.

CMD: (model2_to_test directiory에 있다고 가정)
docker build -t testing .               # A 보다 좀 시간이 걸림
docker run --rm -p 8080:8080 testing
curl -X POST http://localhost:8080/invocations -d "{\"instances\": [[2, 529, 5944, 7176, 7673, 6579, 6043, 606, 423, 420, 45, 6141, 6682, 2152, 6855, 7443, 4285, 7227, 3]]}"  
# 여기도 A 보다 좀 시간이 걸림 (Docker memory 8GB로 set).

OUTPUT:
{"predictions": [[[-15.4804869, -24.1950378, -29.6348152, -29.769352, -27.1416473, -25.5898876, -25.8102512, -28.0569248, -28.2105637, -28.1918392, -27.9692974, -27.9142666, -27.2957497, -27.8690872, -28.558073, -27.3269, -27.3798065],                   
                  [-25.0822754, -26.9999542, -30.3719139, -30.5484905, -26.811039, -18.8830624, -24.973444, -29.0582027, -28.6967049, -27.8556423, -27.8034821, -27.0916958, -27.4764214, -23.2975903, -28.5081234, -27.1044693, -27.8526649]
...

C.a. S3에 있는 'model2' (Saved Model 형식 'model2')

/model2_to_pull/ (파일 안 comments 참고)
└── Dockerfile
└── nginx.conf

*특이사항: 이제 파일들이 2개 밖에 없음. AWS CLI를 사용해서 S3 bucket에 있는 model을 access해야함.
1) ***AWS credentials 필요 (region은 현재 hard coded in. Keys는 argument에서 input)
2) Model PATH 필요 (다른 bucket을 사용한다면)
3) OPTIONAL: Container안에 destination directory PATH (현재 /var/www/html)
-> aws s3 cp를 통해 bucket에 stored된 tar.gz file copy -> container 안에서 tar.gz unzip 
-> (original tar.gz는 아직도 container에 있음. Unzip 과정에서 model은 version directory 00000001에 저장됨) 
Unzip된 model+version directory를 새 model PATH (mkdir로 새로운 directory 만듬)으로 옮김
*여기서 ADD를 사용할 수 없는 이유: ADD는 unzip하지만 local로 옮기는 command. C에 model는 container에 있음.

CMD: (model2_to_pull directiory에 있다고 가정)
docker build -t testing                 # 굉장히 빠름!
    --build-arg AWS_ACCESS_KEY_ID=<access_key_id> 
    --build-arg AWS_SECRET_ACCESS_KEY=<secre_access_key> .               
# Dockerfile에 credentials을 hard code하는건 원치않으니까 argument으로 넣을 수 있게 함.
# ??? From IAM create role 방법도 있음. Docker container에 S3 permission을 주는것 
# ??? 참고: https://stackoverflow.com/questions/51409209/dockerfile-copy-files-from-amazon-s3-or-another-source-that-needs-credentials
docker run --rm -p 8080:8080 testing
curl -X POST http://localhost:8080/invocations -d /"{\"instances\": [[2, 529, 5944, 7176, 7673, 6579, 6043, 606, 423, 420, 45, 6141, 6682, 2152, 6855, 7443, 4285, 7227, 3]]}"  

OUTPUT:
{"predictions": [[[-15.4804869, -24.1950378, -29.6348152, -29.769352, -27.1416473, -25.5898876, -25.8102512, -28.0569248, -28.2105637, -28.1918392, -27.9692974, -27.9142666, -27.2957497, -27.8690872, -28.558073, -27.3269, -27.3798065], 
                  [-25.0822754, -26.9999542, -30.3719139, -30.5484905, -26.811039, -18.8830624, -24.973444, -29.0582027, -28.6967049, -27.8556423, -27.8034821, -27.0916958, -27.4764214, -23.2975903, -28.5081234, -27.1044693, -27.8526649]
...

*TIP: 엇 container안에 servable이 없다는 error때문에 docker run이 infinitely run된다면 
container안의 내용물을 확인해야함.
CMD: (docker run하는 도중에)
docker exec -it <container-name> bash   # ex) docker exec -it modest_hawkings bash
-> 먼저 /var/www/html 확인. 여기에 model이 saved되어 있어야 함. 없으면 PATH를 새로 만들어서 test.
Model이 있는데 servable을 찾지 못하면 model의 file format을 확인. tar.gz file이 아닌 것을 확인.

>> B와 C의 predictions는 동일함. Given prediction과는 소수점 4자리까지 동일.
GIVEN:
                 [[[-15.480492, -24.195038, -29.634819, -29.769352, -27.141647, -25.589891, -25.810253, -28.056925, -28.21057 , -28.191841, -27.969297, -27.914268, -27.29575 , -27.869087, -28.558075, -27.326902, -27.379808],
                   [-25.082268, -26.999954, -30.371908, -30.54849 , -26.811035, -18.88307 , -24.973436, -29.058199, -28.696703, -27.855635, -27.803478, -27.091692, -27.476421, -23.297579, -28.50812, -27.104465, -27.85266 ]
...

C.b. 위에서 (c.a.에서) 만든 Docker image를 ECR repository에 push
(CMD commands에서 보다시피 default image 이름은 'testing')
See: https://bluese05.tistory.com/51

1) AWS ECS console: preprocessing
ECR는 ECR 전용 콘솔이 없음. ECS 콘솔 (get started)에서 wizard를 취소 (cancel)하고 왼쪽 메뉴에
보이는 Repository를 선택 후 create.
# test를 위해 만든 repo 이름은 'repo-test'

2) CMD: AWS CLI 설정
pip install awscli                                  # AWS CLI
aws ecr help                                        # 확인
aws configure                                       # AWS credentials
AWS Access Key ID [None]: <access_key_id>          
AWS Secret Access Key [None]: <secret_access_key>   
Default region name [None]: ap-northeast-2
Default output format [None]: json

3) CMD: 생성한 repo 접근 설정
aws ecr get-login-password --region ap-northeast-2 # ECR에 로그인 
# 중요: --no-include-email없이는 deprecated command가 returned됨!
# 위 실행 성공시 ECR repo에 접속 가능한 docker login command를 볼 수 있다. OR only the password!
# (https://docs.aws.amazon.com/AmazonECR/latest/userguide/common-errors-docker.html#error-403)

# docker login -u AWS -p <password> <aws_account_id>.dkr.ecr.<region>.amazonaws.com
# 중요: <aws-login-id>는 username이 아님. 콘솔 로그인할때 username위에 있는 id.

# 위에 암호화된 키를 복사해 run (repo에 접속). 
# 'Login Succedded'가 return되어야함.

4) CMD: Docker image를 ECR repo에 업로드하기
docker images       # 확인
docker tag <image_name>:latest <aws-login-id>.dkr.ecr.ap-northeast-2.amazonaws.com/<repo_name>:<tag>
docker images       # 확인: image의 repo 정보가 ECR repo URL로 변경됨
                    # ??? testing은 아직도 있는데 정보가 같은 repo URL repo가 만들어짐 ???
docker push <aws-login-id>.dkr.ecr.ap-northeast-2.amazonaws.com/<repo_name>:<tag> # 시간이 오래 걸림

# 모두 Pushed가 되면 ECR console에 가서 repo 확인

5) AWS ECS console: image 확인

OPTIONAL: 6) CMD: Docker host에 image를 받으려면
docker pull <aws-login-id>.dkr.ecr.us-east-1.amazonaws.com/<repo_name>:latest

C.c. Deploy a SageMaker Endpoint
See: https://medium.com/ml-bytes/how-to-deploy-an-aws-sagemaker-container-using-tensorflow-serving-4587dad76169

=======================================================================================================

*중요: tar.gz 만드려면 saved_mode.pb와 variables 폴더를 'version 폴더'에 담아 zip해야함. ex) '00000001'
(version 폴더가 없으면 endpoint deployment할때 에러가 생김: tar.gz가 올바른 format이 아니라고
-> local에서 container를 만들어서 test해봤는데 version 폴더가 필요함)
D.a. LOCAL 'model3' 
현재 S3에 model3.tar.gz 파일을 locally test

/model3_to_test/ (model2_to_test와 비슷함. PATH만 다름)
└── Dockerfile
└── model3.tar.gz
└── nginx.conf

*특이사항: B에서 unzip 과정에서 새로 생기는 version directory '00000001'를 직접 만들어서 zip함.
(tar화에 사용한 00000001 폴더가 정리 folder안에 있음)

OUTPUT:
Previous predictions과 동일.
{"predictions": [[[-15.4804869, -24.1950378, -29.6348152, -29.769352, -27.1416473, -25.5898876, -25.8102512, -28.0569248, -28.2105637, -28.1918392, -27.9692974, -27.9142666, -27.2957497, -27.8690872, -28.558073, -27.3269, -27.3798065], 
                  [-25.0822754, -26.9999542, -30.3719139, -30.5484905, -26.811039, -18.8830624, -24.973444, -29.0582027, -28.6967049, -27.8556423, -27.8034821, -27.0916958, -27.4764214, -23.2975903, -28.5081234, -27.1044693, -27.8526649]
...

D.b. AWS CONSOLE notebook에서 deploy endpoint
S3에 model3.tar.gz 파일 업로드 후 notebook에서 endpoint 생성

model3_test.ipynb -> DP-Endpoint3 (endpoint)
SAMPLE OUTPUT:
{'predictions': [[[-15.4804897, -24.1950378, -29.634819, -29.7693539, -27.1416473, -25.5898914, -25.8102512, -28.0569248, -28.2105713, -28.191843, -27.9693012, -27.9142704, -27.2957535, -27.8690872, -28.558073, -27.3269024, -27.3798027],
                  [-25.0822716, -26.9999542, -30.3719139, -30.5484905, -26.8110352, -18.88307, -24.9734383, -29.0581989, -28.696701, -27.8556366, -27.8034801, -27.091692, -27.4764233, -23.2975807, -28.5081177, -27.1044674, -27.8526592]
...                 

=======================================================================================================

-variables 폴더가 비어있음: https://github.com/tensorflow/models/issues/1988
