import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from bert import modeling
from bert import optimization
import os
import time
import datetime
from datetime import timedelta

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpointDir",
                    "./chinese_L-12_H-768_A-12/",
                    "model  save path")

bert_path = FLAGS.checkpointDir

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)
flags.DEFINE_bool("is_training", True, "is training")

flags.DEFINE_integer("batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("tag_vocab_size", 2, "Total tag size for label")

flags.DEFINE_string("model_version", "4", "model_version ")

# 5e-5
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("buckets", "resource", "buckets info")


class BertTextClassify(object):
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

        with tf.variable_scope("loss"):
            if is_training:
                self.output_layer = tf.nn.dropout(self.output_layer, keep_prob=0.9)

            logits = tf.matmul(self.output_layer, self.output_weights)
            self.logits = tf.nn.bias_add(logits, output_bias)
            self.probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            self.per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            self.loss = tf.reduce_mean(self.per_example_loss)

        with tf.name_scope("train_op"):
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            global_step = tf.train.get_or_create_global_step()
            optimizer = optimization.AdamWeightDecayOptimizer(learning_rate=FLAGS.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)


def main(_):
    print('Loading data...')

    tag_vocab_size = FLAGS.tag_vocab_size

    input_path = FLAGS.buckets + "tfner.records*"
    files = tf.train.match_filenames_once(input_path)

    """
      inputs是你数据的输入路径

    """
    input_ids, input_mask, label_ids = inputs(files, batch_size=FLAGS.batch_size, num_epochs=3)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    is_training = FLAGS.is_training
    init_checkpoint = FLAGS.init_checkpoint
    use_one_hot_embeddings = False

    model = BertTextClassify(bert_config, is_training, input_ids, input_mask
                             , None, label_ids, tag_vocab_size
                             , use_one_hot_embeddings, init_checkpoint)

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("restore sucess  on cpu or gpu")

    session = tf.Session()
    session.run(tf.global_variables_initializer())


    print("**** Trainable Variables ****")
    for var in tvars:
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
            print("name ={0}, shape = {1}{2}".format(var.name, var.shape,
                                                     init_string))

    print("bertlstmner  model will start train .........")

    print(session.run(files))
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    start_time = time.time()
    for i in range(20000):
        _, loss_train = session.run([model.train_op, model.loss])
        if i % 1000 == 0:
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2},' \
                  + '  Cost: {2}  Time:{3}'
            print(msg.format(i, loss_train, time_dif, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            start_time = time.time()
        if i % 40000 == 0 and i > 0:
            saver.save(session, FLAGS.checkpointDir + "bertmodel/model.ckpt", global_step=i)
    coord.request_stop()
    coord.join(threads)
    session.close()


if __name__ == "__main__":
    tf.app.run()