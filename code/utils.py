import os
import pickle
import random

import numpy as np
import tensorflow as tf


def strip_eos(sents):
  return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
          for sent in sents]


def feed_dictionary(model,
                    batch,
                    rho,
                    epsilon,
                    gamma,
                    dropout=1,
                    learning_rate=None,
                    anneal=1,
                    C=25):
  feed_dict = {model.dropout: dropout,
               model.learning_rate: learning_rate,
               model.rho: rho,
               model.epsilon: epsilon,
               model.gamma: gamma,
               model.anneal: anneal,
               model.C: C,
               model.batch_len: batch['len'],
               model.batch_size: batch['size'],
               model.enc_inputs: batch['enc_inputs'],
               model.dec_inputs: batch['dec_inputs'],
               model.targets: batch['targets'],
               model.weights: batch['weights'],
               model.labels: batch['labels']}
  return feed_dict


def makeup(_x, n):
  x = []
  for i in range(n):
    x.append(_x[i % len(_x)])
  return x


def reorder(order, _x):
  x = list(range(len(_x)))
  for i, a in zip(order, _x):
    x[i] = a
  return x


# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise(x, unk, word_drop=0.0, k=3):
  n = len(x)
  for i in range(n):
    if random.random() < word_drop:
      x[i] = unk

  # slight shuffle such that |sigma[i]-i| <= k
  sigma = (np.arange(n) + (k + 1) * np.random.rand(n)).argsort()
  return [x[sigma[i]] for i in range(n)]


def get_batch(x, y, word2id, noisy=False, min_len=5):
  pad = word2id['<pad>']
  go = word2id['<go>']
  eos = word2id['<eos>']
  unk = word2id['<unk>']

  rev_x, go_x, x_eos, weights = [], [], [], []
  max_len = max([len(sent) for sent in x])
  max_len = max(max_len, min_len)
  # max_len = min(max_len, max_seq_len*2)

  # i = 0
  for sent in x:
    # if len(sent) > max_seq_len * 2 or len(y[i]) > max_seq_len * 2:
    #   i += 1
    #   continue
    sent_id = [word2id[w] if w in word2id else unk for w in sent]
    l = len(sent)
    padding = [pad] * (max_len - l)
    _sent_id = noise(sent_id, unk) if noisy else sent_id
    rev_x.append(padding + _sent_id[::-1])
    go_x.append([go] + sent_id + padding)
    x_eos.append(sent_id + [eos] + padding)
    weights.append([1.0] * (l + 1) + [0.0] * (max_len - l))

    # i += 1

  # if i == 0:
  #   print("\n\n\n**** BATCH SIZE is ZERO\n\n\n")
  #   return None

  return {'enc_inputs': rev_x,
          'dec_inputs': go_x,
          'targets': x_eos,
          'weights': weights,
          'labels': y,
          'size': len(x),
          'len': max_len + 1}


def get_batches(x0, x1, word2id, batch_size, noisy=False,
                unparallel=False, max_seq_len=-1):
  if len(x0) < len(x1):
    x0 = makeup(x0, len(x1))
  if len(x1) < len(x0):
    x1 = makeup(x1, len(x0))
  n = len(x0)

  order0 = range(n)
  if unparallel:
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*z)
  # else:
  #   z = zip(order0, x0)
  # order0, x0 = zip(*z)

  order1 = range(n)
  if unparallel:
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)
  # else:
  #   z = zip(order1, x1)
  # order1, x1 = zip(*z)

  batches = []
  s = 0
  count = 0
  violations = 0
  while s < n:
    t = min(s + batch_size, n)

    sources, targets = [], []

    for i in range(s, t):
      if len(x0[i]) > max_seq_len or len(x1[i]) > max_seq_len:
        violations += 1
        continue

      sources.append(x0[i])
      targets.append(x1[i])


    if len(sources) > 0:

      current_batch = get_batch(sources + targets,
                                [0] * len(sources) + [1] * len(targets),
                                word2id, noisy)
      # current_batch = get_batch(x0[s:t] + x1[s:t],
      #                          [0] * (t - s) + [1] * (t - s), word2id, noisy,
      #                          max_seq_len)

      batches.append(current_batch)

      count += 1

    s = t

  print("\n\n*** Violations {}/{} batches ***\n\n".format(violations, count))
  return batches, order0, order1






def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer,
                         trainable=trainable)


def get_mass_test_lines(dataset='../data/yelp/sentiment.test', n_samples=100):
  neg_file = dataset + 'formal'
  pos_file = dataset + 'informal'
  assert os.path.exists(neg_file)
  assert os.path.exists(pos_file)

  if n_samples > 0:
    with open(neg_file) as f:
      neg_lines = random.sample(f.readlines(), n_samples // 2)
    with open(pos_file) as f:
      pos_lines = random.sample(f.readlines(), n_samples // 2)
  else:
    # take all whatever available
    with open(neg_file) as f:
      neg_lines = f.readlines()
    with open(pos_file) as f:
      pos_lines = f.readlines()
  # post processing
  neg_lines = [line.strip().split() for line in neg_lines]
  pos_lines = [line.strip().split() for line in pos_lines]
  print(len(neg_lines), len(pos_lines))
  return neg_lines, pos_lines


def pickle_to_disk(data, dir_path, filename):
  data = np.array(data)

  if not os.path.exists(dir_path): os.mkdir(dir_path)

  file_path = os.path.join(dir_path, filename)
  with open(file_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def _post_process_pickle(data):
  # data_len = data.shape[0]
  element_shape = [i for i in np.array(data[0]).shape]
  #
  # if len(element_shape) == 1:
  #   new_shape = [data_len] + element_shape
  # else: # == 2
  #   new_shape =
  # converted_data = np.zeros(new_shape)
  #
  # for i in range(data_len):
  #   converted_data[i] = np.array(data[i])
  if len(element_shape) > 1:
    converted_data = np.vstack(data)
  else:
    converted_data = np.hstack(data)

  return converted_data


def load_pickle_from_disk(dir_path, filename):
  with open(os.path.join(dir_path, filename), 'rb') as f:
    converted_data = _post_process_pickle(pickle.load(f))
    return converted_data