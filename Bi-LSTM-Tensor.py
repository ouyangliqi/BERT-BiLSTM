"""
  code by chloe Ouyang

"""
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as etree
import os
import re
from bert_serving.client import BertClient
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score
import random

def getTrain(path):
    """
    get the train data from weibo and relevant sentiment
    """
    sentence = []
    polarity = []
    for i in os.listdir(path):
        if i == 'train.txt':
            with open(path + i) as f:
                temp = f.readlines()
            for i in temp:
                sentence.append(i.split(',')[0])
                polarity.append(i.split(',')[1].strip())
            continue
        # if not i.startswith('.'):
        #     tree = etree.parse(path + i)
        #     root = tree.getroot()
        #     for child in root:
        #         if child.find("sentence").get('polarity'):
        #             temp = child.find("sentence").get('polarity')
        #             if temp == 'NEG' or temp == 'POS':
        #                 polarity.append(temp)
        #                 sentence.append(child.find("sentence").text)
    for i in range(len(sentence)):
        sentence[i] = ''.join(sentence[i].split(' '))

    for i in range(len(polarity)):
        if polarity[i] == 'NEG':
            polarity[i] = 0
        if polarity[i] == 'negative':
            polarity[i] = 0
        if polarity[i] == 'POS':
            polarity[i] = 1
        if polarity[i] == 'positive':
            polarity[i] = 1
    return polarity, sentence


def getTest(path):
    test = []
    polarity = []
    for i in os.listdir(path):
        if i == 'test.txt':
            with open(path + i) as f:
                temp = f.readlines()
            for i in temp:
                sentence.append(i.split(',')[0])
                polarity.append(i.split(',')[1].strip())
            continue
        # if not i.startswith('.'):
        #     tree = etree.parse(path + i)
        #     root = tree.getroot()
        #     for child in root:
        #         if child.find("sentence").get('polarity'):
        #             temp = child.find("sentence").get('polarity')
        #             if temp == 'NEG' or temp == 'POS':
        #                 polarity.append(temp)
        #                 test.append(child.find("sentence").text)
    for i in range(len(test)):
        test[i] = ''.join(test[i].split(' '))

    for i in range(len(polarity)):
        if polarity[i] == 'NEG':
            polarity[i] = 0
        if polarity[i] == 'negative':
            polarity[i] = 0
        if polarity[i] == 'POS':
            polarity[i] = 1
        if polarity[i] == 'positive':
            polarity[i] = 1
    return polarity, test


def get_sentence_batch(batch_size, data_x, data_y, data_seqlens, word_list):
    instance_indices = list(range(len(len(data_x))))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word_list[n] for n in data_x[i].split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


def shuffle_xy(label, sentence):
    print(sentence)
    index = [i for i in range(len(sentence))]
    index = random.shuffle(index)
    new_sentence = [sentence[i] for i in index]
    new_label = [label[i] for i in index]
    return new_label, new_sentence


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='./BiLSTM-training.log',
                        filemode='w')

    logging.info('Training start')

    tf.reset_default_graph()

    model_path = './chinese_L-12_H-768_A-12'

    # 3 words sentences (=sequence_length is 3)
    labels, sentences = getTrain("./data/train/")

    print(labels)
    print(sentences)

    # Bi-LSTM(Attention) Parameters
    embedding_dim = 768
    n_hidden = 128  # number of hidden units in one cell
    n_step = 128  #
    n_class = 2  # 0 or 1
    batch_size = len(sentences)

    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    vocab = [token for token in tokenizer.vocab]
    word_list = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    print('constructing input---------------------------------------------------------------')
    input_batch = []
    for s in sentences:
        input_batch.append(np.asarray([word_list[n] for n in tokenizer.tokenize(s)]))

    input_batch = keras.preprocessing.sequence.pad_sequences(input_batch,
                                                             value=word_list["[PAD]"],
                                                             padding='post',
                                                             maxlen=128)

    target_batch = []
    for out in labels:
        target_batch.append(np.eye(n_class)[out])  # ONE-HOT : To using Tensor Softmax Loss function

    # LSTM Model
    X = tf.placeholder(tf.int32, [None, n_step], name='inputs')
    Y = tf.placeholder(tf.int32, [None, n_class], name='label')
    # out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

    weights = {
        'linear_layer': tf.Variable(tf.truncated_normal([n_hidden * 2, n_class], mean=0, stddev=.01))
    }

    biases = {
        'linear_layer': tf.Variable(tf.truncated_normal([n_class], mean=0, stddev=.01))
    }

    # initialized the embedding
    print("# Load word embeddings----------------------------------------------------")
    emb = model.embeddings.word_embeddings.weight.data  # token embedding
    # position_embeddings in tensorflow called PositionalEmbedding
    # token_type_embeddings in tensorflow called SegmentEmbedding
    embeddings = emb.numpy()
    embed = tf.nn.embedding_lookup(embeddings, X)

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

    # output : [batch_size, len_seq, n_hidden], states : [batch_size, n_hidden]
    outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed, dtype=tf.float32)

    outputs = tf.concat([outputs[0], outputs[1]], 2)  # output[0] : lstm_fw, output[1] : lstm_bw
    # print(outputs.shape)  # (?, 128, 10) (batch_size, n_step, n_hidden) -> (0, 1, 2)
    outputs = tf.transpose(outputs, [1, 0, 2])  # [n_step, batch_size, n_hidden]
    # print(outputs.shape)  # (128, ?, 10)
    outputs = outputs[-1]  # [batch_size, n_hidden]
    # print(outputs.shape)  # (?, 10)

    logits = tf.matmul(outputs, weights['linear_layer']) + biases['linear_layer']
    # print(logits.shape)
    prediction = tf.nn.softmax(logits)
    # print(prediction.shape)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # saver
    saver = tf.train.Saver()

    # Training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(1000):
            _, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: input_batch, Y: target_batch})
            if (epoch + 1) % 10 == 0:
                print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        # Test -------------------------------------------------------------------------------
        test_sen, test = getTest("./data/test/")

        test_batch = []
        for sentence in test:
            test_batch.append(np.asarray([word_list[n] for n in tokenizer.tokenize(sentence)]))

        test_batch = keras.preprocessing.sequence.pad_sequences(test_batch,
                                                                value=word_list["[PAD]"],
                                                                padding='post',
                                                                maxlen=128)
        test_tagert_batch = []
        for out in test_sen:
            test_tagert_batch.append(np.eye(n_class)[out])  # ONE-HOT : To using Tensor Softmax Loss function

        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: test_batch, Y: test_tagert_batch}))

        y_accuracy, y_pred = sess.run([accuracy, prediction], feed_dict={X: test_batch, Y: test_tagert_batch})
        # print(len(np.argmax(y_pred, 1)))
        print('Testing Accuracy:%f'%(accuracy_score(test_sen, np.argmax(y_pred, 1))))
        print('F1 score:%f'%(f1_score(test_sen, np.argmax(y_pred, 1))))
        fpr, tpr, _ = roc_curve(test_sen, np.argmax(y_pred, 1))
        print('AUC:%f'%(auc(fpr, tpr)))
        print('Recall %f'%(recall_score(test_sen, np.argmax(y_pred, 1))))
        save_path = saver.save(sess, "./model2/Bilstm.ckpt")
        print("Model saved in path: %s" % save_path)

    logging.info('Training Finished')
