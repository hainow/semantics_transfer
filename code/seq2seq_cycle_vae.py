import sys
import time

import beam_search
import greedy_decoding
from accumulator import Accumulator
from file_io import load_sent, write_sent
from nn import *
from options import load_arguments
from utils import *
from vocab import Vocabulary, build_vocab


class Model(object):
  def __init__(self, args, vocab):
    # dimensions: output, intermediate, hidden
    dim_y = args.dim_y
    dim_z = args.dim_z
    dim_h = dim_y + dim_z

    # dimension embeding
    dim_emb = args.dim_emb
    n_layers = args.n_layers
    max_len = args.max_seq_length
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    n_filters = args.n_filters
    beta1, beta2 = 0.5, 0.999
    grad_clip = 30.0

    self.dropout = tf.placeholder(tf.float32,
                                  name='dropout')
    self.learning_rate = tf.placeholder(tf.float32,
                                        name='learning_rate')
    self.rho = tf.placeholder(tf.float32,
                              name='rho')
    self.epsilon = tf.placeholder(tf.float32,
                                  name='epsilon')
    self.gamma = tf.placeholder(tf.float32,
                                name='gamma')
    self.anneal = tf.placeholder(tf.float32, name='anneal')  # beta
    self.C = tf.placeholder(tf.float32, name='C')  # beta

    # he uses cudnn then the batch_len is fized (is padding somewhere?)
    self.batch_len = tf.placeholder(tf.int32,
                                    name='batch_len')
    self.batch_size = tf.placeholder(tf.int32,
                                     name='batch_size')

    # size * len
    self.enc_inputs = tf.placeholder(tf.int32, [None, None],  # size * len
                                     name='enc_inputs')

    self.dec_inputs = tf.placeholder(tf.int32, [None, None],
                                     name='dec_inputs')
    self.targets = tf.placeholder(tf.int32, [None, None],
                                  name='targets')
    self.weights = tf.placeholder(tf.float32, [None, None],
                                  name='weights')
    self.labels = tf.placeholder(tf.float32, [None],
                                 name='labels')

    # reshape labels
    labels = tf.reshape(self.labels, [-1, 1])

    # here we have all embedings
    embedding = tf.get_variable('embedding',
                                initializer=vocab.embedding.astype(np.float32))

    with tf.variable_scope('projection'):
      proj_W = tf.get_variable('W', [dim_h, vocab.size])
      proj_b = tf.get_variable('b', [vocab.size])

      # two different embedings are defined? (why is that?)
    enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
    dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

    #####   auto-encoder   #####
    # we perfrom a linear transformation of labels to dim_y
    # we concatenate the dimensionality of z that is the intermediate vector
    # since we dont have a hiden vector yet we intiialize it with 0
    init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
                            tf.zeros([self.batch_size, dim_z])], 1)

    # we connect the encoded inputs
    # init_state computed above as the first hidden state
    cell_e = create_cell(dim_h, n_layers, self.dropout)
    _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
                             initial_state=init_state, scope='encoder')

    z = z[:, dim_y:]

    # VAE part
    mu = linear(z, dim_out=dim_z, scope='mu')
    log_sigma = linear(z, dim_out=dim_z, scope='sigma')

    sigma = tf.exp(log_sigma)

    # KL Divergence loss
    kld = -0.5 * tf.reduce_mean(
      tf.reduce_sum(
        1 + tf.log(tf.square(sigma) + 0.0001)
        - tf.square(mu)
        - tf.square(sigma), 1))

    self.kld_loss = tf.multiply(tf.abs(kld - self.C), self.anneal)

    samples = tf.random_normal(shape=[tf.shape(z)[0], dim_z])
    z = mu + (samples * sigma)
    print("after VAE: {}".format(z.shape))

    # for disentanglement test
    self.encoded_style = z[:, :dim_y]  # output for dim_y
    self.encoded_content = z
    print("after VAE: content_shape={} style_shape={}".
          format(self.encoded_content.shape, self.encoded_style.shape))

    # cell_e = create_cell(dim_z, n_layers, self.dropout)
    # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
    #    dtype=tf.float32, scope='encoder')

    # they perform the exact same concatenation
    # original
    self.h_ori = tf.concat([linear(labels, dim_y,
                                   scope='generator'), z], 1)

    # not sure to understand what is this one? (why 1- is subtracted)
    # transfered
    self.h_tsf = tf.concat([linear(1 - labels, dim_y,
                                   scope='generator', reuse=True), z], 1)

    # we use here the h_orig
    cell_g = create_cell(dim_h, n_layers, self.dropout)
    # we are using the outputs
    g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
                                     initial_state=self.h_ori,
                                     scope='generator')

    # we are concatenating h original and the outputs of the generator
    # TODO TO UNDERSTAND
    teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

    g_outputs = tf.nn.dropout(g_outputs, self.dropout)
    g_outputs = tf.reshape(g_outputs, [-1, dim_h])
    g_logits = tf.matmul(g_outputs, proj_W) + proj_b

    # here we get the logits of the generator
    print("Check the shape1 of loss: ", self.targets.shape, g_logits.shape)

    loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(self.targets, [-1]), logits=g_logits)

    # regularitzation
    loss_rec *= tf.reshape(self.weights, [-1])
    self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)

    #####   feed-previous decoding   #####
    go = dec_inputs[:, 0, :]
    soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
                                self.gamma)
    hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)


    print(self.h_ori.shape, go.shape)
    print(self.h_tsf.shape, go.shape)

    # soft_h_ori: all outputs
    soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
                                             cell_g, soft_func,
                                             scope='generator')
    # soft_h_ori: all outputs
    soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
                                             cell_g, soft_func,
                                             scope='generator')

    hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
                                                  cell_g, hard_func,
                                                  scope='generator')
    hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
                                                  cell_g, hard_func,
                                                  scope='generator')

    # hard_h_tsf
    # we start with hidden state =0
    # init_state2 = tf.concat([linear(labels, dim_y, scope='encoder'),
    # tf.zeros([self.batch_size, dim_z])], 1)

    enc_hard_inputs_cyc = tf.nn.embedding_lookup(embedding,
                                                 tf.argmax(self.hard_logits_tsf,
                                                           axis=2))

    init_state_cyc = tf.concat([linear(1 - labels, dim_y, scope='encoder_cyc'),
                                tf.zeros([self.batch_size, dim_z])], 1)

    _, z_cyc = tf.nn.dynamic_rnn(cell_e, enc_hard_inputs_cyc,
                                 initial_state=init_state_cyc, scope='encoder')
    z_cyc = z_cyc[:, dim_y:]

    self.h_tsf_cyc = tf.concat([linear(labels, dim_y,
                                       scope='generator', reuse=True), z_cyc],
                               1)

    go = enc_inputs[:, 0, :]

    hard_h_tsf_cyc, self.hard_logits_tsf_cyc = rnn_decode(self.h_tsf_cyc, go,
                                                          max_len,
                                                          cell_g, hard_func,
                                                          scope='generator')

    # cyc_out=tf.argmax(self.hard_logits_tsf_cyc, axis=2)

    # print("LOSS TEST: labels {} \t {} logits {}".format(self.enc_inputs.shape,
    #                                               self.hard_logits_tsf_cyc.shape,
    #                                               self.hard_logits_tsf_cyc[
    #                                                            :, :tf.shape(
    #                                                       self.enc_inputs)[1],
    #                                                            :].shape))

    self.loss_rec_cyc = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=self.enc_inputs,
      logits=self.hard_logits_tsf_cyc[:, :tf.shape(self.enc_inputs)[1], :])

    print("Check the shape2 of loss: ", self.enc_inputs.shape,
          self.hard_logits_tsf_cyc[:, :tf.shape(self.enc_inputs)[1], :].shape)

    self.loss_rec_cyc = tf.reduce_sum(self.loss_rec_cyc) / tf.to_float(
      self.batch_size)
    # labels=tf.reshape(self.enc_inputs, [-1]), logits=self.hard_logits_tsf_cyc)

    #####   discriminator   #####
    # (function in nn.py)

    # a batch's first half consists of sentences of one style,
    # and second half of the other
    half = self.batch_size // 2
    zeros, ones = self.labels[:half], self.labels[half:]
    soft_h_tsf = soft_h_tsf[:, :1 + self.batch_len, :]

    # here a cnn is used
    # they compute two losses

    # discriminator for one style
    self.loss_d0, loss_g0 = discriminator(teach_h[:half], soft_h_tsf[half:],
                                          ones, zeros, filter_sizes, n_filters,
                                          self.dropout,
                                          scope='discriminator0')

    # discriminator for the second style
    self.loss_d1, loss_g1 = discriminator(teach_h[half:], soft_h_tsf[:half],
                                          ones, zeros, filter_sizes, n_filters,
                                          self.dropout,
                                          scope='discriminator1')

    #####   optimizer   #####

    self.loss_adv = loss_g0 + loss_g1

    self.loss = self.loss_rec + self.rho * self.loss_adv + self.epsilon * self.loss_rec_cyc + self.kld_loss

    theta_eg = retrive_var(['encoder', 'generator',
                            'embedding', 'projection'])
    theta_d0 = retrive_var(['discriminator0'])
    theta_d1 = retrive_var(['discriminator1'])

    opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

    grad_rec, _ = zip(*opt.compute_gradients(self.loss_rec, theta_eg))
    grad_adv, _ = zip(*opt.compute_gradients(self.loss_adv, theta_eg))
    grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
    grad, _ = tf.clip_by_global_norm(grad, grad_clip)

    self.grad_rec_norm = tf.global_norm(grad_rec)
    self.grad_adv_norm = tf.global_norm(grad_adv)
    self.grad_norm = tf.global_norm(grad)

    self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
    self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
    self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
    self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

    self.saver = tf.train.Saver()


def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
  batches, order0, order1 = get_batches(data0, data1,
                                        vocab.word2id, args.batch_size,
                                        max_seq_len=args.max_seq_length)

  # data0_rec, data1_rec = [], []
  data0_tsf, data1_tsf = [], []
  losses = Accumulator(len(batches),
                       ['loss', 'rec', 'adv', 'd0', 'd1', 'loss_rec_cyc',
                        'loss_kld'])
  for batch in batches:
    rec, tsf = decoder.rewrite(batch)
    half = batch['size'] // 2
    # data0_rec += rec[:half]
    # data1_rec += rec[half:]
    data0_tsf += tsf[:half]
    data1_tsf += tsf[half:]

    loss, loss_rec, loss_adv, loss_d0, loss_d1, loss_rec_cyc, loss_kld = \
      sess.run([model.loss,
                model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1,
                model.loss_rec_cyc, model.kld_loss],
               feed_dict=feed_dictionary(model=model,
                                         batch=batch,
                                         rho=args.rho,
                                         epsilon=args.epsilon,
                                         gamma=args.gamma_min,
                                         anneal=args.anneal,
                                         C=args.C))

    # feed_dict order: model, batch, rho, epsilon, gamma, dropout=1, learning_rate=None, anneal=1
    losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1, loss_rec_cyc,
                loss_kld])

  n0, n1 = len(data0), len(data1)
  # data0_rec = reorder(order0, data0_rec)[:n0]
  # data1_rec = reorder(order1, data1_rec)[:n1]
  data0_tsf = reorder(order0, data0_tsf)[:n0]
  data1_tsf = reorder(order1, data1_tsf)[:n1]

  if out_path:
    # write_sent(data0_rec, out_path+'.0'+'.rec')
    # write_sent(data1_rec, out_path+'.1'+'.rec')
    write_sent(data0_tsf, out_path + 'formal' + '.tsf')
    write_sent(data1_tsf, out_path + 'informal' + '.tsf')

  return losses


# def create_model(sess, args, vocab):
#     model = Model(args, vocab)
#     if args.load_model:
#         print 'Loading model from', args.model
#         model.saver.restore(sess, args.model)
#     else:
#         print 'Creating model with fresh parameters.'
#         sess.run(tf.global_variables_initializer())
#     return model


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


def online_transfer(neg_lines, write_file='neg.txt', style=0):
  with open(write_file, 'w') as f:
    for line in neg_lines:
      y = style
      batch = get_batch([line], [y], vocab.word2id)
      ori, tsf = decoder.rewrite(batch)
      # f.write('input: {}\n'.format(' '.join(line)))
      # f.write('original: {}\n'.format(' '.join(w for w in ori[0])))
      # f.write('transfer: {}\n'.format(' '.join(w for w in tsf[0])))
      f.write('{}\n'.format(' '.join(line)))
      f.write('{}\n'.format(' '.join(w for w in ori[0])))
      f.write('{}\n'.format(' '.join(w for w in tsf[0])))


if __name__ == '__main__':
  args = load_arguments()

  if not os.path.exists(args.model):
    os.system("mkdir -p {}".format(args.model))

  #####   data preparation   #####
  if args.train or args.latent_train:
    chosen = args.train if len(args.train) > len(args.latent_train) else \
      args.latent_train
    # train0 = load_sent(chosen + '.0', args.max_train_size)
    # train1 = load_sent(chosen + '.1', args.max_train_size)

    train0 = load_sent(chosen + 'formal', args.max_train_size)
    train1 = load_sent(chosen + 'informal', args.max_train_size)

    print('#sents of training file 0:', len(train0))
    print('#sents of training file 1:', len(train1))

    if not os.path.isfile(args.vocab):
      build_vocab(train0 + train1, args.vocab)

  vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
  print('vocabulary size:', vocab.size)

  if args.dev or args.latent_dev:
    chosen = args.dev if len(args.dev) > len(args.latent_dev) else \
      args.latent_dev
    dev0 = load_sent(chosen + 'formal')
    dev1 = load_sent(chosen + 'informal')

  if args.test or args.latent_test:
    chosen = args.test if len(args.test) > len(args.latent_test) else \
      args.latent_test
    test0 = load_sent(chosen + 'formal')
    test1 = load_sent(chosen + 'informal')

  # get condifg object and set dynamic memory aloc
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True



  with tf.Session(config=config) as sess:
    # model = create_model(sess, args, vocab)

    print("\n\n*** ALWAYS TRAIN FROM SCRATCH NOW\n\n")
    model = Model(args, vocab)
    sess.run(tf.global_variables_initializer())

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




    # set type of decoding (is this after the very last layer?)
    if args.beam > 1:
      decoder = beam_search.Decoder(sess, args, vocab, model)
    else:
      decoder = greedy_decoding.Decoder(sess, args, vocab, model)

    if args.train:
      batches, _, _ = get_batches(train0, train1, vocab.word2id,
                                  args.batch_size, noisy=True,
                                  unparallel=False,
                                  max_seq_len=args.max_seq_length)
      random.shuffle(batches)

      start_time = time.time()
      step = 0
      losses = Accumulator(args.steps_per_checkpoint,
                           ['loss', 'rec', 'adv', 'd0', 'd1', "loss_rec_cyc",
                            'loss_kld'])
      best_dev = float('inf')
      learning_rate = args.learning_rate
      rho = args.rho
      epsilon = args.epsilon
      gamma = args.gamma_init
      dropout = args.dropout_keep_prob
      anneal = args.anneal
      C = args.C

      # gradients = Accumulator(args.steps_per_checkpoint,
      #    ['|grad_rec|', '|grad_adv|', '|grad|'])
      print("***SCHEDULING C FROM 0.0 to 25.0 ***")

      C_increase = float(args.C) / (args.max_epochs * len(batches))
      C = C_increase

      for epoch in range(1, 1 + args.max_epochs):
        print('--------------------epoch %d--------------------' % epoch)
        print('learning_rate:', learning_rate, '  gamma:', gamma)

        # anneal += anneal_increase
        # anneal = min(1.0, anneal_increase)
        for batch in batches:

          # import pdb; pdb.set_trace()
          C += C_increase

          feed_dict = feed_dictionary(model, batch, rho, epsilon, gamma,
                                      dropout, learning_rate, anneal, C)
          # feed_dict order: model, batch, rho, epsilon, gamma, dropout=1, learning_rate=None, anneal=1, C=25
          loss_d0, _ = sess.run([model.loss_d0, model.optimize_d0],
                                feed_dict=feed_dict)
          loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                                feed_dict=feed_dict)

          # do not back-propagate from the discriminator
          # when it is too poor
          if loss_d0 < 1.2 and loss_d1 < 1.2:
            optimize = model.optimize_tot
          else:
            optimize = model.optimize_rec

          loss, loss_rec, loss_adv, _, loss_rec_cyc, loss_vae = sess.run(
            [model.loss,
             model.loss_rec, model.loss_adv, optimize, model.loss_rec_cyc,
             model.kld_loss],
            feed_dict=feed_dict)
          losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1, loss_rec_cyc,
                      loss_vae])

          # grad_rec, grad_adv, grad = sess.run([model.grad_rec_norm,
          #    model.grad_adv_norm, model.grad_norm],
          #    feed_dict=feed_dict)
          # gradients.add([grad_rec, grad_adv, grad])

          step += 1
          if step % args.steps_per_checkpoint == 0:
            losses.output('step %d, time %.0fs,'
                          % (step, time.time() - start_time))
            losses.clear()

            checkpoint_path = os.path.join(args.model, 'model.ckpt')
            model.saver.save(sess, args.model, global_step=step)
            print('\t\tModel saved to {}'.format(checkpoint_path))
            # gradients.output()
            # gradients.clear()

        if args.dev:
          dev_losses = transfer(model, decoder, sess, args, vocab,
                                dev0, dev1, args.output + '.epoch%d' % epoch)
          dev_losses.output('dev')

          if dev_losses.values[0] < best_dev:
            best_dev = dev_losses.values[0]
            print('saving model...')
            checkpoint_path = os.path.join(args.model, 'model.ckpt')
            model.saver.save(sess, args.model, global_step=step)
            print('\tSaved to {}'.format(checkpoint_path))

        gamma = max(args.gamma_min, gamma * args.gamma_decay)

        # we test every epoch along with validation
        if args.test:
          print("Testing in epoch {}".format(epoch))
          test_losses = transfer(model, decoder, sess, args, vocab, test0,
                                 test1, args.output + '.test.epoch%d' % epoch)
          test_losses.output('test')

    if args.test:
      print("Final test")
      test_losses = transfer(model, decoder, sess, args, vocab,
                             test0, test1, args.output)
      test_losses.output('test')

    if args.online_testing:
      while True:
        sys.stdout.write('> ')
        sys.stdout.flush()
        inp = sys.stdin.readline().rstrip()
        if inp == 'quit' or inp == 'exit':
          break
        inp = inp.split()
        y = int(inp[0])
        sent = inp[1:]

        batch = get_batch([sent], [y], vocab.word2id)
        ori, tsf = decoder.rewrite(batch)
        print('original:', ' '.join(w for w in ori[0]))
        print('transfer:', ' '.join(w for w in tsf[0]))

    if args.mass_online_testing:
      neg_lines, pos_lines = get_mass_test_lines(args.test,
                                                 n_samples=args.total_test)

      online_transfer(neg_lines,
                      write_file='cycle_{}_{}_neg.txt'.format(args.anneal,
                                                            args.C),
                      style=0)
      online_transfer(pos_lines,
                      write_file='cycle_{}_{}_pos.txt'.format(args.anneal,
                                                            args.C),
                      style=1)

    if args.latent_testing:
      train_batches, _, _ = get_batches(train0, train1, vocab.word2id,
                                        args.batch_size, noisy=True)
      val_batches, _, _ = get_batches(dev0, dev1, vocab.word2id,
                                      args.batch_size, noisy=False)

      test_batches, _, _ = get_batches(test0, test1, vocab.word2id,
                                       args.batch_size, noisy=False)

      losses = Accumulator(args.steps_per_checkpoint,
                           ['loss', 'rec', 'adv', 'd0', 'd1', "loss_rec_cyc",
                            'loss_kld'])
      best_dev = float('inf')
      learning_rate = args.learning_rate
      rho = args.rho
      epsilon = args.epsilon
      gamma = args.gamma_init
      dropout = args.dropout_keep_prob
      anneal = args.anneal
      C = args.C

      # train
      train_latent_contents, train_latent_styles, train_gt_labels = [], [], []
      c = 0
      for batch in train_batches:
        c += 1
        if c % 100 == 0: print('{} / {}'.format(c, len(train_batches)))
        feed_dict = feed_dictionary(model, batch, rho, epsilon, gamma,
                                    dropout, learning_rate, anneal, C)

        gt_labels = batch['labels']  # array of integers in {0, 1}

        latent_content, latent_style = sess.run([model.encoded_content,
                                                 model.encoded_style],
                                                feed_dict=feed_dict)

        train_gt_labels.append(gt_labels)
        train_latent_contents.append(latent_content)
        train_latent_styles.append(latent_style)

      pickle_to_disk(train_latent_contents, args.latent_path, 'train_zc.pkl')
      pickle_to_disk(train_latent_styles, args.latent_path, 'train_zy.pkl')
      pickle_to_disk(train_gt_labels, args.latent_path, 'train_y.pkl')

      # val
      val_latent_contents, val_latent_styles, val_gt_labels = [], [], []
      c = 0
      for batch in val_batches:
        c += 1
        if c % 100 == 0: print('{} / {}'.format(c, len(val_batches)))
        feed_dict = feed_dictionary(model=model,
                                    batch=batch,
                                    rho=args.rho,
                                    epsilon=args.epsilon,
                                    gamma=args.gamma_min,
                                    anneal=args.anneal,
                                    C=args.C)
        gt_labels = batch['labels']  # array of integers in {0, 1}

        latent_content, latent_style = sess.run([model.encoded_content,
                                                 model.encoded_style],
                                                feed_dict=feed_dict)

        val_gt_labels.append(gt_labels)
        val_latent_contents.append(latent_content)
        val_latent_styles.append(latent_style)

      pickle_to_disk(val_latent_contents, args.latent_path, 'val_zc.pkl')
      pickle_to_disk(val_latent_styles, args.latent_path, 'val_zy.pkl')
      pickle_to_disk(val_gt_labels, args.latent_path, 'val_y.pkl')


      # test
      test_latent_contents, test_latent_styles, test_gt_labels = [], [], []
      c = 0
      for batch in test_batches:
        c += 1
        if c % 100 == 0: print('{} / {}'.format(c, len(val_batches)))

        feed_dict = feed_dictionary(model=model,
                                    batch=batch,
                                    rho=args.rho,
                                    epsilon=args.epsilon,
                                    gamma=args.gamma_min,
                                    anneal=args.anneal,
                                    C=args.C)
        gt_labels = batch['labels']  # array of integers in {0, 1}

        latent_content, latent_style = sess.run([model.encoded_content,
                                                 model.encoded_style],
                                                feed_dict=feed_dict)

        test_gt_labels.append(gt_labels)
        test_latent_contents.append(latent_content)
        test_latent_styles.append(latent_style)

      # dump to disk
      if not os.path.exists(args.latent_path):
        os.mkdir(args.latent_path)

      pickle_to_disk(test_latent_contents, args.latent_path, 'test_zc.pkl')
      pickle_to_disk(test_latent_styles, args.latent_path, 'test_zy.pkl')
      pickle_to_disk(test_gt_labels, args.latent_path, 'test_y.pkl')
