''' 
@project WDGRL
@author Peng
@file WDGRL.py
@time 2018-08-28
'''

class WDGRL():
    def __init__(self, l2 = 0.001, learning_rate = 0.01, batch_size = 128, training_steps = 5000,
                 optimizer = 'GD', save_step = 100, print_step = 20, wd_param = 0.0005, gp_param = 10,
                 learning_rate_wd = 1e-4):
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.optimizer = optimizer
        self.save_step = save_step
        self.print_step = print_step
        self.wd_param = wd_param
        self.gp_param = gp_param
        self.learning_rate_wd = learning_rate_wd

    def fit(self, X_src, X_tar):
        pass

    def transfor(self, X_tar):
        pass

    def nn(self):
        pass
