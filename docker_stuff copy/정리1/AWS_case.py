[문제]
tensorflow_model_server를 사용해서 ONNX model을 serve하는데 문제가 있다.

에러 메세지: 
E tensorflow_serving/core/aspired_versions_manager.cc:359] Servable {name: model version: 1} cannot be loaded: Not found: Could not find meta graph def matching supplied tags: { serve }. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`
Failed to start server. Error: Unknown: 1 servable(s) did not become available: {{{name: model version: 1} due to error: Not found: Could not find meta graph def matching supplied tags: { serve }. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`}, }

[AWS 답]
-SavedModel에 serving tag에 correspond하는 그래프가 없다.
-*리뷰: A and B 에 비슷한 문제들에 대한 해결 방법들이 있음.
-SavedModel을 inspect해봐라 (SavedModel CLI 사용).
-> Correct tag를 export/ save하는 process에서 specify (*리뷰: C로 inspect model)

[리뷰]
A. tag-set 에러 (tensorflow_model_server사용시)
https://github.com/tensorflow/models/issues/3530

질문: 
-tensorflow repo에서 다운 받은 'flowers' database
-Existing checkpoint에서 inception-v3 model을 fine tuning함
(See: https://github.com/tensorflow/models/tree/master/research/slim)
-Exported된 그래프를 freeze함 
(freeze = 그래프와 checkpoint variables가 합쳐진 하나의 파일을 만드는것)
-> 성공

/home/alfred/Testing/OLD/Ejemplo/frozen/
└── 1
├── saved_model.pb  # the frozen graph
└── variables
├── saved_model.data-00000-of-00001
└── saved_model.index

(비슷한 에러가 남. saved_model_cli로 확인하면 tag-set가 없음)

답:
Model을 tag set 'serve'로 다시 freeze해라.

builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],)
builder.save()

그러면 (리뷰: C 참고)
saved_model_cli  show --dir output_dir  # tag set 확인

B. pb 파일을 tensor flow serving에 업로드 중 에러
https://github.com/tensorflow/serving/issues/491

질문:
-SaveModel을 사용해서 TensorFlow model file들을 export함.
-TensorFlow Serving을 사용해서 파일들을 load함.

(model version에 관한 비슷한 에러)

답:
SavedModel에 serving tag에 correspond하는 그래프가 없다.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py#L26

Exported된 SavedModel은 각 그래프에 correspond하는 tag-set가 있어야한다.
SavedModel을 export하거나 save할때 specify해야한다.
SavedModel의 tag-sets를 inspect하려면 SavedModel CLI를 사용.
!!! https://www.tensorflow.org/programmers_guide/saved_model_cli !!! (404 에러)
-> 리뷰: C

C. SavedModel CLI
https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface

####################

[Stack Overflow 답변]

There are two APIs for deploying TensorFlow models: tensorflow.Model and tensorflow.serving.Model. 
It isn't clear from the code-snippet which one you're using, but the SageMaker docs recommend the latter
deploying from pre-existing s3 artifacts:

from sagemaker.tensorflow.serving import Model

model = Model(model_data='s3://mybucket/model.tar.gz', role='MySageMakerRole')

predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
# Reference: https://github.com/aws/sagemaker-python-sdk/blob/c919e4dee3a00243f0b736af93fb156d17b04796/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst#deploying-directly-from-model-artifacts

If you haven't specified an image argument for tensorflow.Model, SageMaker should be using 
the default TensorFlow serving image (seems like "../tensorflow-inference").
image (str) – A Docker image URI (default: None). If not specified, a default image for 
TensorFlow Serving will be used. If all of this seems needlessly complex to you, I'm working on 
a platform that makes this set up a single line of code -- 
I'd love for you to try it, dm me at https://twitter.com/yoavz_.