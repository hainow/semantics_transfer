import os
import random
import time

import tensorflow as tf

from file_io import load_sent
from nn import cnn
from options import load_arguments
from vocab import Vocabulary, build_vocab


class Model(object):
  def __init__(self, args, vocab):
    dim_emb = args.dim_emb
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    n_filters = args.n_filters

    self.dropout = tf.placeholder(tf.float32,
                                  name='dropout')
    self.learning_rate = tf.placeholder(tf.float32,
                                        name='learning_rate')
    self.x = tf.placeholder(tf.int32, [None, None],  # batch_size * max_len
                            name='x')
    self.y = tf.placeholder(tf.float32, [None],
                            name='y')

    # embeding
    embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
    x = tf.nn.embedding_lookup(embedding, self.x)

    # import CNN
    self.logits = cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')

    # Sigmoid binomiial ditribution?
    self.probs = tf.sigmoid(self.logits)

    # cross entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=self.y, logits=self.logits)

    # mean over all samples
    self.loss = tf.reduce_mean(loss)

    # optimize
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
      .minimize(self.loss)

    self.saver = tf.train.Saver()


def create_model(sess, args, vocab):
  model = Model(args, vocab)
  if args.load_model:
    print('Loading model from', args.model)
    ckpt = tf.train.get_checkpoint_state(args.model)
    if ckpt and ckpt.model_checkpoint_path:
      try:
        print("Trying to restore from a checkpoint...")
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model is restored from checkpoint {}".format(
          ckpt.model_checkpoint_path))
      except Exception as e:
        print("Cannot restore from checkpoint due to {}".format(e))
        pass
  else:
    print('Creating model with fresh parameters.')
    sess.run(tf.global_variables_initializer())
  return model


def evaluate(sess, args, vocab, model, x, y):
  probs = []
  batches = get_batches(x, y, vocab.word2id, args.batch_size)
  for batch in batches:
    p = sess.run(model.probs,
                 feed_dict={model.x: batch['x'],
                            model.dropout: 1})
    probs += p.tolist()

  y_hat = [p > 0.5 for p in probs]
  same = [p == q for p, q in zip(y, y_hat)]
  return 100.0 * sum(same) / len(y), probs


def get_batches(x, y, word2id, batch_size, min_len=5):
  pad = word2id['<pad>']
  unk = word2id['<unk>']

  batches = []
  s = 0
  while s < len(x):
    t = min(s + batch_size, len(x))

    _x = []
    max_len = max([len(sent) for sent in x[s:t]])
    max_len = max(max_len, min_len)
    for sent in x[s:t]:
      sent_id = [word2id[w] if w in word2id else unk for w in sent]
      padding = [pad] * (max_len - len(sent))
      _x.append(padding + sent_id)

    batches.append({'x': _x,
                    'y': y[s:t]})
    s = t

  return batches


def prepare(path, suffix=''):
  data0 = load_sent(path + 'formal' + suffix)
  data1 = load_sent(path + 'informal' + suffix)
  x = data0 + data1
  y = [0] * len(data0) + [1] * len(data1)
  z = sorted(zip(x, y), key=lambda i: len(i[0]))
  return zip(*z)


if __name__ == '__main__':
  args = load_arguments()

  if args.train:
    train_x, train_y = prepare(args.train)

    if not os.path.isfile(args.vocab):
      build_vocab(train_x, args.vocab)

  # prepare vocabulary
  # we set the embeding dimension
  # we read a pickel file (presumably with the data?)
  # randomly initialize the vector
  # normalize the random vectors
  # embedings are normalized
  vocab = Vocabulary(args.vocab)
  print('vocabulary size', vocab.size)

  # prepare datasets:
  # read form file,
  # zip
  # order them
  if args.dev:
    dev_x, dev_y = prepare(args.dev)

  # same thing
  if args.test:
    test_x, test_y = prepare(args.test, '')
    # test_x, test_y = prepare(args.test)

  if args.test_transfer:
    test_x_tsf, test_y_tsf = prepare(args.test_transfer, '.tsf')

  # get the configuration object from tensorflow
  config = tf.ConfigProto()
  # allow dynamic allocation of memory
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:

    # load model, if we haf .ckpt we load it
    # otherwise, we set them fresh
    model = create_model(sess, args, vocab)
    # MODEL:

    # embeding
    # CNN
    # sigmoid (sigmoid binomial ditribution?)
    # cross entropy

    if args.train:
      batches = get_batches(train_x, train_y,
                            vocab.word2id, args.batch_size)
      random.shuffle(batches)

      start_time = time.time()
      step = 0
      loss = 0.0
      best_dev = float('-inf')
      learning_rate = args.learning_rate

      for epoch in range(1, 1 + args.max_epochs):
        print('--------------------epoch %d--------------------' %
              epoch)

        for batch in batches:
          step_loss, _ = sess.run([model.loss, model.optimizer],
                                  feed_dict={model.x: batch['x'],
                                             model.y: batch['y'],
                                             model.dropout: args.dropout_keep_prob,
                                             model.learning_rate: learning_rate})

          step += 1
          loss += step_loss / args.steps_per_checkpoint

          if step % args.steps_per_checkpoint == 0:
            print('step %d, time %.0fs, loss %.2f' \
                  % (step, time.time() - start_time, loss))
            loss = 0.0

        if args.test:
          acc, _ = evaluate(sess, args, vocab, model, test_x, test_y)
          print('test accuracy %.2f' % acc)
          if acc > best_dev:
            best_dev = acc
            print('Saving the better model now ...')
            checkpoint_path = os.path.join(args.model, 'model.ckpt')
            model.saver.save(sess, args.model, global_step=step)
            print('\tSaved to {}'.format(checkpoint_path))

    if args.test_transfer:
      acc, _ = evaluate(sess, args, vocab, model, test_x_tsf, test_y_tsf)
      print('test TRANFER accuracy %.2f' % acc)
      # checkpoint_path = os.path.join(args.model, 'model.ckpt')
      # model.saver.save(sess, args.model, global_step=step)
      # print('\tSaved to {}'.format(checkpoint_path))
