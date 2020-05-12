import sys

import tensorflow as tf

from utils import load_pickle_from_disk

user_flags = []


def DEFINE_string(name, default_value, doc_string):
  tf.app.flags.DEFINE_string(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
  tf.app.flags.DEFINE_integer(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
  tf.app.flags.DEFINE_float(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
  tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def read_data(dir_path="../data/yelp/latent"):
  train_zc = load_pickle_from_disk(dir_path, 'train_zc.pkl')
  # val_zc = load_pickle_from_disk(dir_path, 'val_zc.pkl')
  test_zc = load_pickle_from_disk(dir_path, 'test_zc.pkl')

  train_zy = load_pickle_from_disk(dir_path, 'train_zy.pkl')
  # val_zy = load_pickle_from_disk(dir_path, 'val_zy.pkl')
  test_zy = load_pickle_from_disk(dir_path, 'test_zy.pkl')

  train_y = load_pickle_from_disk(dir_path, 'train_y.pkl')
  # val_y = load_pickle_from_disk(dir_path, 'val_y.pkl')
  test_y = load_pickle_from_disk(dir_path, 'test_y.pkl')

  data_dict = {
    "train_zc": train_zc,
    # "val_zc": val_zc,
    "test_zc": test_zc,
    "train_zy": train_zy,
    # "val_zy": val_zy,
    "test_zy": test_zy,
    "train_y": train_y,
    # "val_y": val_y,
    "test_y": test_y,
  }

  print('-' * 80)
  for k in data_dict:
    print(k, data_dict[k].shape)
  print('-' * 80)

  return data_dict


def create_batch_tf_dataset(data_dict,
                            type='content',
                            batch_size=32,
                            buffer_size=10000):
  s = 'c' if type == 'content' else 'y'
  train_dataset = tf.data.Dataset. \
    from_tensor_slices((data_dict['train_z{}'.format(s)],
                        data_dict['train_y']))
  train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
  batched_train_dataset = train_dataset.batch(batch_size=batch_size)
  batched_train_dataset = batched_train_dataset.repeat()  # forever

  # val_dataset = tf.data.Dataset. \
  #   from_tensor_slices((data_dict['val_z{}'.format(s)],
  #                       data_dict['val_y']))
  # batched_val_dataset = val_dataset.batch(batch_size=batch_size)

  test_dataset = tf.data.Dataset. \
    from_tensor_slices((data_dict['test_z{}'.format(s)],
                        data_dict['test_y']))
  batched_test_dataset = test_dataset.batch(batch_size=batch_size)

  return {
    "train": batched_train_dataset,
    # "val": batched_val_dataset,
    "test": batched_test_dataset
  }


def _inference(x,
               model_type='mlp',
               is_train=True,
               n_outputs=20):
  if model_type == 'mlp':
    return _mlp(x, is_train=is_train, num_classes=n_outputs)


def _mlp(x, is_train, dims=(1000, 512, 128), num_classes=2, drop_keep_prob=0.5):
  for layer_id, next_dim in enumerate(dims):
    curr_dim = x.get_shape()[-1].value  # get_shape() returns a <list>

    with tf.variable_scope("layer_{}".format(layer_id)):
      w = tf.get_variable("w", [curr_dim, next_dim])  # w's name: "layer_2/w"

    x = tf.matmul(x, w)

    # drop_out
    x = tf.cond(is_train,  # a tensor
                lambda: tf.nn.dropout(x, drop_keep_prob),
                lambda: x)

    x = tf.nn.relu(x)

  curr_dim = x.get_shape()[-1].value  # get_shape() returns a <list>
  with tf.variable_scope("logits"):
    w = tf.get_variable("w", [curr_dim, num_classes])  # w's name: "logits/w"
    logits = tf.matmul(x, w)
  return logits


def create_tf_ops(data_dict=None,
                  model_type='mlp',
                  n_outputs=2,
                  init_lr=0.001,
                  l2_reg=1e-3,
                  batch_size=32):
  """ Create and finalize a TF graph including ops """
  dataset_dict = create_batch_tf_dataset(data_dict,
                                         batch_size=batch_size)
  train_dataset = dataset_dict["train"]
  # val_dataset = dataset_dict["val"]
  test_dataset = dataset_dict["test"]

  # for conciseness, this iterator will be shared between train, val, test
  # will switch the dataset when respective initializer is called first
  shared_iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types,
    train_dataset.output_shapes
  )
  imgs, labels = shared_iterator.get_next()

  # Indicates whether we are in training or in test mode for inferennce graph
  is_training = tf.placeholder(tf.bool)

  # shared weights for inference as well
  logits = _inference(imgs,
                      model_type=model_type,
                      is_train=is_training,
                      n_outputs=n_outputs
                      )

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")
  # loss function
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
  train_loss = tf.reduce_mean(xentropy)
  l2_loss = tf.losses.get_regularization_loss()
  train_loss += l2_reg * l2_loss

  # optimizer
  lr = tf.train.exponential_decay(init_lr, global_step * 64,
                                  50000, 0.98, staircase=True)
  optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

  # train
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions
  preds = tf.to_int32(tf.argmax(logits, axis=1))

  # put everything into an ops dict
  ops = {
    "global_step": global_step,
    "is_training": is_training,
    "train_loss": train_loss,
    "preds": preds,
    "labels": labels,
    "train_iterator": shared_iterator.make_initializer(train_dataset),
    # "val_iterator": shared_iterator.make_initializer(val_dataset),
    "test_iterator": shared_iterator.make_initializer(test_dataset),
    "train_op": train_op,
  }
  return ops


def main():
  flags = tf.app.flags
  FLAGS = flags.FLAGS

  DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
  DEFINE_string("data_dir", "output", "Path to pickled latent data files")
  DEFINE_string("output_dir", "output", "Path to log folder")
  DEFINE_string("type", "content", "content or style")
  DEFINE_string("model_name", "mlp",
                "Name of the method. [softmax|feed_forward|conv]")
  DEFINE_integer("n_epochs", 100, "How many epochs to run in total")
  DEFINE_integer("train_steps", 8000, "How many batches per epoch")
  DEFINE_integer("log_every", 8000, "How many steps to log")
  DEFINE_integer("n_classes", 2, "Number of classes")
  DEFINE_integer("batch_size", 32, "Batch size")
  DEFINE_float("init_lr", 1e-3, "Init learning rate")

  print("Loading data")
  data_dict = read_data(FLAGS.data_dir)

  # computational graph
  g = tf.Graph()
  tf.reset_default_graph()

  with g.as_default():
    ops = create_tf_ops(data_dict=data_dict,
                        model_type=FLAGS.model_name,
                        n_outputs=FLAGS.n_classes,
                        init_lr=FLAGS.init_lr,
                        l2_reg=1e-3,
                        batch_size=FLAGS.batch_size)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)

    # hook up with a session to train
    with tf.train.SingularMonitoredSession(
        config=config, checkpoint_dir=FLAGS.output_dir) as sess:

      # training loop
      print("-" * 80)
      print("Starting training")

      for epoch in range(1, FLAGS.n_epochs + 1):
        sess.run(ops["train_iterator"])  # init dataset iterator
        for step in range(1, FLAGS.train_steps + 1):
          _, loss, preds, labels = \
            sess.run([ops["train_op"],
                      ops["train_loss"],
                      ops["preds"],
                      ops["labels"]
                      ],
                     feed_dict={ops["is_training"]: True})

          if step > 0 and step % 10 == 0:
            acc = sum(preds == labels) / float(len(labels))
            print("Epoch %d Batch %d: loss = %.3f train_accuracy = %.3f" %
                  (epoch, step, loss, acc))

          if step % FLAGS.log_every == 0:
            # this will reset train_dataset as well
            get_eval_accuracy(ops, sess, step, "test")

      print("-" * 80)
      print("Training done. Eval on TEST set")
      get_eval_accuracy(ops, sess, step, "test")

      print("Training done. Eval on TRAIN set")
      get_eval_accuracy(ops, sess, step, "train")


def get_eval_accuracy(ops, sess, step, name="val"):
  if name == "val":
    sess.run(ops["val_iterator"])
  elif name == "test":
    sess.run(ops["test_iterator"])
  elif name == "train":
    sess.run(ops["train_iterator"])
  else:
    raise NotImplementedError

  if name == "train":
    stop_batch = 534628 // 64

  n_samples, n_corrects = 0, 0
  count = 0
  while True:
    try:
      count += 1
      if name == "train":
        if count == stop_batch:
          break
      if count  % 100 == 0:
        print("Batch {}".format(count))


      global_step, val_loss, preds, labels = \
        sess.run([ops["global_step"],
                  ops["train_loss"],
                  ops["preds"],
                  ops["labels"]],
                 feed_dict={ops["is_training"]: False})
      n_samples += preds.shape[0]
      n_corrects += sum(preds == labels)
    except tf.errors.OutOfRangeError:
      break

  # another way is to average all vall_acc above
  total_vall_acc = n_corrects / float(n_samples)

  log_string = "\n{}: ".format(name)
  log_string += "step={0:<6d}".format(step)
  log_string += " acc={0:.3f} against {1:<3d} samples\n".format(
    total_vall_acc, n_samples)
  print(log_string)
  sys.stdout.flush()


if __name__ == "__main__":
  # data_dict = read_data()
  # import pdb; pdb.set_trace()

  main()
