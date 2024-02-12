import tensorflow as tf
import os 

def convert(meta_path, ouput_pb_name='output_model.pb'):
    output_node_names = ['output']

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(meta_path, '..')))

        # Freeze
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)
        
        with open(ouput_pb_name, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

if __name__ == '__main__':
    # use .meta file in order to restore model graph - it should be something like saved_models/model250.meta  
    meta_path = './saved_models/model250.meta'
    convert(meta_path)