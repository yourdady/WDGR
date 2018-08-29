''' 
@project WDGRL
@author Peng
@file WDGRL.py
@time 2018-08-28
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'


class WDGRL():
    def __init__(self, l2 = 0.001, learning_rate = 0.01, batch_size = 128, D_train_steps = 20, training_steps = 5000,
                 optimizer = 'GD', input_dim = 500, middle_dim = 100, new_dim = 50,
                 save_step = 100, print_step = 20, wd_param = 0.05, gp_param = 1,
                 learning_rate_wd = 1e-4, n_classes = 10):
        """
        
        :param l2: 
        :param learning_rate: 
        :param batch_size: 
        :param D_train_steps: 
        :param training_steps: 
        :param optimizer: 
        :param input_dim: 
        :param middle_dim: 
        :param new_dim: 
        :param save_step: 
        :param print_step: 
        :param wd_param: 
        :param gp_param: 
        :param learning_rate_wd: 
        :param n_classes: 
        """
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.D_train_steps = D_train_steps
        self.training_steps = training_steps
        self.optimizer = optimizer
        self.save_step = save_step
        self.print_step = print_step
        self.wd_param = wd_param
        self.gp_param = gp_param
        self.learning_rate_wd = learning_rate_wd
        self.input_dim = input_dim
        self.middle_dim = middle_dim
        self.new_dim = new_dim
        self.n_classes = n_classes

    def fit(self, data_src, data_tar, draw_plot=False):
        with tf.Graph().as_default() as g:
            clf_losses = None
            wd_losses = None
            iterations = None
            if draw_plot == True:
                clf_losses = []
                wd_losses = []
                iterations = []
            global_step = tf.Variable(0, trainable=False)
            X_src_placeholder = tf.placeholder(shape=[self.batch_size, self.input_dim],
                                               dtype=tf.float32, name='Xsrc')
            X_tar_placeholder = tf.placeholder(shape=[self.batch_size, self.input_dim],
                                               dtype=tf.float32, name='Xtar')
            y_src_placeholder = tf.placeholder(shape=[self.batch_size, self.n_classes],
                                               dtype=tf.float32, name='Ysrc')
            g_s, g_t = self.wdgrl_generator(X_src_placeholder, X_tar_placeholder)
            logits = self.wdgrl_classifier(g_s)
            clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=y_src_placeholder))
            critic_out, h_whole, d_s, d_t = self.wdgrl_discriminator(g_s, g_t)
            wd_loss = tf.reduce_mean(d_s) - tf.reduce_mean(d_t)
            gradients = tf.gradients(critic_out, [h_whole])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            theta_C = [v for v in tf.global_variables() if 'classifier' in v.name]
            theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
            theta_G = [v for v in tf.global_variables() if 'generator' in v.name]

            # 训练分辨器wd_loss梯度上升, 提升领域区分能力

            if self.optimizer == 'GD':
                wd_d_op = tf.train.GradientDescentOptimizer(self.learning_rate_wd).minimize(-wd_loss +
                                                                                 self.gp_param * gradient_penalty,
                                                                                 var_list=theta_D)
                                                                                 # )
            else:
                wd_d_op = tf.train.GradientDescentOptimizer(self.learning_rate_wd).minimize(-wd_loss +
                                                                                            self.gp_param * gradient_penalty,
                                                                                            var_list=theta_D)
                                                                                            # )
            all_variables = tf.trainable_variables()
            l2_loss = self.l2 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
            total_loss = clf_loss + l2_loss + self.wd_param * wd_loss
            print([v.name for v in tf.global_variables()])
            # 训练生成器wd_loss，梯度下降, 提升领域相似性
            if self.optimizer == 'GD':
                train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(total_loss,
                                                                               global_step=global_step,
                                                                               var_list=theta_G + theta_C)
                                                                               # )
            else:
                train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(total_loss,
                                                                                          global_step=global_step,
                                                                                          var_list=theta_G + theta_C)
                                                                                          # )
            saver = tf.train.Saver()
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                xs, ys = data_src.dataset.train.next_batch(self.batch_size)
                xt, _ = data_tar.dataset.train.next_batch(self.batch_size)
                for i in range(self.training_steps):
                    for j in range(self.D_train_steps):
                        _ = sess.run([wd_d_op], feed_dict={X_src_placeholder: xs,
                                                     X_tar_placeholder: xt,
                                                     y_src_placeholder: ys})


                    _, wd_loss_, clf_loss_ = sess.run([train_op, wd_loss, clf_loss], feed_dict={X_src_placeholder: xs,
                                                                                                X_tar_placeholder: xt,
                                              y_src_placeholder: ys})
                    if draw_plot == True:
                        clf_losses.append(clf_loss_)
                        wd_losses.append(wd_loss_)
                        iterations.append(i)

                    if i%self.print_step == 0:
                        print("After {} training steps\nwd_loss:{} clf_loss:{}".format(i, wd_loss_, clf_loss_))
                    if i%self.save_step == 0:
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                if draw_plot == True:
                    self.draw_plot(iterations, clf_loss=clf_losses, wd_loss=wd_losses)


    def transform(self, X):
        """
        
        :param X_tar: 
        :return: 
        """
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [
                None,
                self.input_dim],name='x-input')
            _, h_t = self.wdgrl_generator(X_tar=x)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    X_new = sess.run([h_t], feed_dict={x: X})
                    print("transform successfully, shape: ", np.array(X_new).shape)
                    return X_new



    def wdgrl_discriminator(self, h_src, h_tar):
        """
        
        :param h_src: 
        :param h_tar: 
        :return: 
        """
        with tf.variable_scope('critic'):
            weights_d = tf.get_variable(name="weights_d", shape=[self.new_dim, 1])
            biases_d = tf.get_variable(name="biases_d", shape=[1])
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            differences = h_src - h_tar
            interpolates = h_tar + (alpha * differences)
            h_st = tf.concat([h_src, h_tar], 0)
            h_whole = tf.concat([h_st, interpolates], 0)
            critic_out = self._fully_connected(h_whole, weights_d, biases_d)
            d_s = self._fully_connected(h_src, weights_d, biases_d)
            d_t = self._fully_connected(h_tar, weights_d, biases_d)
            return critic_out, h_whole, d_s, d_t

    def wdgrl_classifier(self, h_src):
        """
        
        :param h_src: 
        :return: 
        """
        with tf.variable_scope('classifier'):
            weights = tf.get_variable("weights", shape = [self.new_dim, self.n_classes],
                                      initializer=tf.truncated_normal_initializer(stddev=1))
            biases = tf.get_variable("biases", shape = [self.n_classes],
                                     initializer=tf.truncated_normal_initializer(stddev=1))
            pred_logits = self._fully_connected(h_src, weights, biases)

        return pred_logits


    def wdgrl_generator(self, X_src=None, X_tar=None):
        """
        
        :param X_src: 
        :param X_tar: 
        :param train: 
        :return: 
        """

        if X_tar is None:
            raise TypeError("X_tar shouldn't be None.")

        with tf.variable_scope("generator"):
            weights_1 = tf.get_variable("weights_1", shape=[self.input_dim, self.middle_dim],
                                        initializer=tf.truncated_normal_initializer(stddev=1))
            bias_1 = tf.get_variable("bias_1", shape=[self.middle_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=1))
            hidden_layer_src = None
            if X_src is not None:
                hidden_layer_src = tf.nn.relu(self._fully_connected(X_src, weights_1, bias_1))
            hidden_layer_tar = tf.nn.relu(self._fully_connected(X_tar, weights_1, bias_1))

        with tf.name_scope("layer2"):
            weights_2 = tf.get_variable("weights_2", shape=[self.middle_dim, self.new_dim],
                                        initializer=tf.truncated_normal_initializer(stddev=1))
            bias_2 = tf.get_variable("bias_2", shape=[self.new_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=1))
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(self.l2)(weights_2))
            h_s = None
            if X_src is not None:
                h_s = tf.nn.relu(self._fully_connected(hidden_layer_src, weights_2, bias_2))
            h_t = tf.nn.relu(self._fully_connected(hidden_layer_tar, weights_2, bias_2))
        return h_s, h_t


    def draw_plot(self, iter, clf_loss, wd_loss):
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111)
        p1, = ax1.plot(iter, np.array(wd_loss)/max(wd_loss), 'b-', label='wd_loss')
        ax1.set_ylabel('WASSERSTEIN DISTANCE')
        ax1.set_title("Iters")
        ax1.yaxis.label.set_color(p1.get_color())
        ax2 = ax1.twinx()
        p2, = ax2.plot(iter, np.array(clf_loss)/max(clf_loss), 'g--', label='src loss')
        ax2.set_ylabel("SRC TRAINING LOSS")
        ax2.yaxis.label.set_color(p2.get_color())
        plt.savefig('./demo.png')
        plt.show()

    ###############################################################################
    # Helper functions
    ################################################################################

    def _fully_connected(self, input_layer, weights, biases):
        return tf.add(tf.matmul(input_layer, weights), biases)

