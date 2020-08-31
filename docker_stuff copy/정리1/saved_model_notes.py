<Saved Model>
Unviersal serialization format > TensorFlow

Ex) 
Maybe? https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two.py

Features:
-Multiple graphs sharing a single set of variables and assets > single SavedModel
Each graph is associated with a specific set of tags to allow identifications during a load/ restore operation
-Support for 'SignatureDefs'
Signature = set of inputs and ouputs ~inputs, outputs, method_name > graphs used for inference tasks
'SignatureDefs' > generic support for signatures that may need to be saved with the graphs
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md
-Support for 'Assets'
'Assets' < ops depend on external files for initialization ~ vocabularies
They are copied to the SavedModel location and can be read when loading a specific meta graph def 
-Support to clear devices before generating the SavedModel

NOT supported in SavedModel:
-Implicit versioning
-Garbage collection
-Atomatic writes to the SavedModel location

SavedModel < builds upon existing TensorFlow primitives ~ TensorFlow Saver, MetaGraphDef
SavedModel wraps a TensorFlow Saver, whihc is used to generate the variable checkpoints
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py
Existing TensorFlow Inference Model Format > replaced by SavedModel = a canonical way to export TensorFlow graphs for serving

Sample directory:
assets/                             # Contains auxiliary files such as vocabularies, etc.
assets.extra/                       # NOT managed by the SavedModel libraries and NOT loaded by the graph
                                    # Where higher-level libraries and users can add their own assets that coexist with the model                        
variables/                          # Includes output from the TensorFlow Saver
    variables.data-?????-of-?????
    variables.index
saved_model.pb                      # Includes the graph definitions as MetaGraphDef protocol buffers

APIS:
-Builder (Python): provides functionality to save multiple meta graph defs, associated variables and assets

To build a Saved Model:
A. First meta graph MUST be saved with variables and assets # Q. WHAT are these variables?
                                                            # Cannot access them in the examples because they have weird format (WHAT even is the file format...?)
B. Subsequent meta graphs saved with their graph definitions
C. If assets need to be saved and written/ copied to disk > can be provided when the meta graph def is added
D. If multiple meta graph defs are associated with an asset of the same name > only the first version is retained

-Tags: reflect the meta graph capabilites or use-cases 
> annotate a meta graph with is functionality ~ serving or training
Each meta graph added to the SavedModel must be annotated with user specified tags
The meta graph def whose tag-set exactly matches those specified in the loader API > loaded by the loader
If no meta graph def is found matching the specified tags > error returned

Ex) Loader with a requirement to serve on GPU hardware
> would be able to load only meta graph annotated with tags 'serve,gpu'
< where this set of tags are defined in tensorflow::LoadSavedModel(...)

Common issues:
1. A saved_model.pb file with an empty variables directory
< the graph is "frozen" aka all variables are converted to constant nodes in the graph
> no "variables" left in the computations
> directory is empty

>> Just modify the function "_write_saved_model" in "exporter.py" like this:
1. Use the default graph which is already loaded into the global environment instead of generating a frozen one
2. Don't forget to modify the caller's arguments.

def _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs):
  """Writes SavedModel to disk.
  Args:
    saved_model_path: Path to write SavedModel.
    trained_checkpoint_prefix: path to trained_checkpoint_prefix.
    inputs: The input image tensor to use for detection.
    outputs: A tensor dictionary containing the outputs of a DetectionModel.
  """
  saver = tf.train.Saver()
  with session.Session() as sess:
    saver.restore(sess, trained_checkpoint_prefix)
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

    tensor_info_inputs = {
          'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
    tensor_info_outputs = {}
    for k, v in outputs.items():
      tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

    detection_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
              inputs=tensor_info_inputs,
              outputs=tensor_info_outputs,
              method_name=signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  detection_signature,
          },
      )
    builder.save()