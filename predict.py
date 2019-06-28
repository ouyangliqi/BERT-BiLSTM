"""
    code by chloe Ouyang
    Predict sentiment using pre-trained model
"""
from pytorch_pretrained_bert import BertTokenizer, BertModel
import tensorflow as tf
import re
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score


def getTest(path):
    sentence = []
    polarity = []
    with open(path) as f:
        temp = f.readlines()
    for i in temp:
        sentence.append(i.split(',')[0])
        polarity.append(i.split(',')[1].strip())
    for i in range(len(polarity)):
        if polarity[i] == '0':
            polarity[i] = 0
        if polarity[i] == '1':
            polarity[i] = 1
        if polarity[i] == 'negative':
            polarity[i] = 0
        if polarity[i] == 'positive':
            polarity[i] = 1
    return sentence, polarity

embedding_dim = 768
n_hidden = 128  # number of hidden units in one cell
n_step = 128  # all sentence is consist of 3 words
n_class = 2  # 0 or 1

checkpiont_path = '/home/user1/chloe/BERT+BiLSTM/model/Bilstm.ckpt'
model_path = './chinese_L-12_H-768_A-12'

model = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

vocab = [token for token in tokenizer.vocab]
word_list = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# s, p = getTest('./data/test/original.txt')

s, p = getTest('./data/test/translation.txt')

test_batch = []
for sentence in s:
    test_batch.append(np.asarray([word_list[n] for n in tokenizer.tokenize(sentence)]))

test_batch = keras.preprocessing.sequence.pad_sequences(test_batch,
                                                             value=word_list["[PAD]"],
                                                             padding='post',
                                                             maxlen=128)
tf.reset_default_graph()

X = tf.placeholder(tf.int32, [None, n_step], name='inputs')
Y = tf.placeholder(tf.int32, [None, n_class], name='label')
# out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

weights = {
    'linear_layer': tf.Variable(tf.truncated_normal([n_hidden * 2, n_class], mean=0, stddev=.01))
}

biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([n_class], mean=0, stddev=.01))
}

emb = model.embeddings.word_embeddings.weight.data  # token embedding
# position_embeddings in tensorflow called PositionalEmbedding
# token_type_embeddings in tensorflow called SegmentEmbedding
embeddings = emb.numpy()
embed = tf.nn.embedding_lookup(embeddings, X)

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed, dtype=tf.float32)

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

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpiont_path)
    y_pred = sess.run(prediction, {X: test_batch})
    # print(np.argmax(y_pred, 1))

# print(p)
print('Testing Accuracy:%f'%(accuracy_score(p, np.argmax(y_pred, 1))))
print('F1 score:%f'%(f1_score(p, np.argmax(y_pred, 1))))
fpr, tpr, _ = roc_curve(p, np.argmax(y_pred, 1))
print('AUC:%f'%(auc(fpr, tpr)))
print('Recall %f'%(recall_score(p, np.argmax(y_pred, 1))))

# # Create some variables.
# v1 = tf.get_variable("weights", shape=[n_hidden * 2, n_class])
# v2 = tf.get_variable("biases", shape=[n_class])
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, checkpiont_path)
#     print("Model restored.")
#     # Check the values of the variables
#     print("v1 : %s" % v1.eval())
#     print("v2 : %s" % v2.eval())
#
