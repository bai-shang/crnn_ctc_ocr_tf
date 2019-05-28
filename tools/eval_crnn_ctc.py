from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import tensorflow as tf

import numpy as np

from crnn_model import model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/', 'Path to the directory containing data tf record.')

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')

tf.app.flags.DEFINE_integer(
    'num_threads', 4, 'The number of threads to use in batch shuffling') 

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

# ------------------------------------LSTM prameters------------------------------------
tf.app.flags.DEFINE_integer(
    'lstm_hidden_layers', 2, 'The number of stacked LSTM cell.')

tf.app.flags.DEFINE_integer(
    'lstm_hidden_uints', 256, 'The number of units in each LSTM cell')

# ------------------------------------Char dictionary------------------------------------

tf.app.flags.DEFINE_string(
    'char_map_json_file', './char_map/char_map.json', 'Path to char map json file')

FLAGS = tf.app.flags.FLAGS

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict))    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict))
    
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _read_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannott find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.image.decode_jpeg(features['images'])
    images.set_shape([32, None, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 4, tf.int32)
    imagenames = features['imagenames']
    return images, labels, sequence_length, imagenames

def _eval_crnn_ctc():
    tfrecord_path = os.path.join(FLAGS.data_dir, 'validation.tfrecord')
    images, labels, sequence_lengths, imagenames = _read_tfrecord(tfrecord_path=tfrecord_path)

    # decode the training data from tfrecords
    batch_images, batch_labels, batch_sequence_lengths, batch_imagenames = tf.train.batch(
        tensors=[images, labels, sequence_lengths, imagenames], batch_size=FLAGS.batch_size, dynamic_pad=True,
        capacity=1000 + 2*FLAGS.batch_size, num_threads=FLAGS.num_threads)

    input_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32, None, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size], name='input_sequence_lengths')

    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='test',
                                    hidden_num=FLAGS.lstm_hidden_uints,
                                    layers_num=FLAGS.lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(images=input_images, sequence_length=input_sequence_lengths)

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_lengths, merge_repeated=False)

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        test_sample_count += 1
    step_nums = test_sample_count // FLAGS.batch_size

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accuracy = []

        for _ in range(step_nums):
            imgs, lbls, seq_lens, names = sess.run([batch_images, batch_labels, batch_sequence_lengths, batch_imagenames])
            preds = sess.run(ctc_decoded, feed_dict={input_images:imgs, input_labels:lbls, input_sequence_lengths:seq_lens})

            preds = _sparse_matrix_to_list(preds[0], char_map_dict)
            lbls = _sparse_matrix_to_list(lbls, char_map_dict)


            #print(preds)
            #print(lbls)
            for index, lbl in enumerate(lbls):
                pred = preds[index]
                total_count = len(lbl)
                correct_count = 0
                try:
                    for i, tmp in enumerate(lbl):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            for index, img in enumerate(imgs):
                print('Predict {:s} image with gt label: {:s} <--> predict label: {:s}'.format(names[index], lbls[index], preds[index]))
        
        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        print('Mean test accuracy is {:5f}'.format(accuracy))

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

def main(unused_argv):
    _eval_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 

