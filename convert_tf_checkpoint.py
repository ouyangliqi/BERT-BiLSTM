"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import torch
import numpy as np
from modules.layers.bert_modeling import BertConfig, BertModel


def convert(args):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(args.bert_config_file)
    model = BertModel(config)

    # Load weights from TF model
    path = args.tf_checkpoint_path
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split('/')
        if name[0] in ['redictions', 'eq_relationship']:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), args.pytorch_dump_path)


if __name__ == "__main__":
    BERT_BASE_DIR = './chinese_L-12_H-768_A-12'
    args = {
        'tf_checkpoint_path': BERT_BASE_DIR + '/bert_model.ckpt',
        'bert_config_file': BERT_BASE_DIR + '/bert_config.json',
        'pytorch_dump_path': BERT_BASE_DIR + '/pytorch_model.bin',
    }
    convert(args)

