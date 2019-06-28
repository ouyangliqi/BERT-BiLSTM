from bert_serving.client import BertClient
import os
import xml.etree.ElementTree as etree
from bert import modeling, tokenization
import tensorflow as tf
from bert_base.bert import modeling
from pytorch_pretrained_bert import BertTokenizer, BertModel

def getEmbedding(sentence):
    """
    Word embedding using BertClient
    """
    bc = BertClient('localhost')
    var = bc.encode([x for x in sentence.strip() if x.strip()])[0]  # embedding
    return var


def getTrain(path):
    """
    get the train data from weibo and relevant sentiment
    """
    sentence = []
    polarity = []
    for i in os.listdir(path):
        if not i.startswith('.'):
            tree = etree.parse(path+i)
            root = tree.getroot()
            for child in root:
                if child.find("sentence").get('polarity'):
                    polarity.append(child.find("sentence").get('polarity'))
                    sentence.append(child.find("sentence").text)
    for i in range(len(polarity)):
        if polarity[i] == 'NEG':
            polarity[i] = 0
        if polarity[i] == 'POS':
            polarity[i] = 1
    return polarity, sentence


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    '''
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    '''
    return max_seq_length


def load_Model(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, args):
    is_training = True

    # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
    total_loss, logits, trans, pred_ids = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)

    tvars = tf.trainable_variables()
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars,
                                                        init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 打印变量名
    logger.info("**** Trainable Variables ****")

    # 打印加载模型的参数
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        logger.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)


def get_embeddings(mname):
    '''Gets pretrained embeddings of Bert-tokenized tokens or subwords
    mname: string. model name.
    '''
    print("# Model name:", mname)

    print("# Load pre-trained model tokenizer (vocabulary)")
    tokenizer = BertTokenizer.from_pretrained(mname)

    print("# Construct vocab")
    vocab = [token for token in tokenizer.vocab]

    print("# Load pre-trained model")
    model = BertModel.from_pretrained(mname)

    print("# Load word embeddings")
    emb = model.embeddings.word_embeddings.weight.data
    emb = emb.numpy()

    print("# Write")
    with open("{}.{}.{}d.vec".format(mname, len(vocab), emb.shape[-1]), "w") as fout:
        fout.write("{} {}\n".format(len(vocab), emb.shape[-1]))
        assert len(vocab)==len(emb), "The number of vocab and embeddings MUST be identical."
        for token, e in zip(vocab, emb):
            e = np.array2string(e, max_line_width=np.inf)[1:-1]
            e = re.sub("[ ]+", " ", e)
            fout.write("{} {}\n".format(token, e))



def load_Model():
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
    total_loss, logits, trans, pred_ids = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)

    tvars = tf.trainable_variables()
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars,
                                                        init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 打印变量名
    logger.info("**** Trainable Variables ****")

    # 打印加载模型的参数
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        logger.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)


class BertBiLSTM(object):
    def __init__(self, bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, init_checkpoint):
        self.bert_config = bert_config
        self.is_training = is_training
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.num_labels = num_labels
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.init_checkpoint = init_checkpoint

        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )

        self.output_layer = self.model.get_pooled_output()
        self.hidden_size = self.output_layer.shape[-1].value
        self.output_weights = tf.get_variable(
            "output_weights", [self.hidden_size, self.num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

if __name__ == '__main__':
    # polarity, sentence = getTrain("./data/train/")
    # for i in sentence:
    #     em = getEmbedding(i)
    #     print(len(em))

    bert_dir = './chinese_L-12_H-768_A-12'
    p = Pool(16)
    with tqdm(total=len(mnames)) as pbar:
        for _ in tqdm(p.imap(get_embeddings, mnames)):
            pbar.update()

