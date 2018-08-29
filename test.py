''' 
@project WDGRL
@author Peng
@file test.py
@time 2018-08-28
'''
from utils import load_mnist, load_usps, parse_data
from WDGRL import WDGRL

def test():
    ux, uy = parse_data()
    usps_data = load_usps(ux, uy, validation_size=5000, test_size=0)
    mnist_data = load_mnist(one_hot=True, validation_size=5000)
    x_original = mnist_data.dataset.validation._images
    wdgrl = WDGRL(input_dim=784, gp_param=10, training_steps=2000, D_train_steps=20)
    wdgrl.fit(data_src=usps_data, data_tar=mnist_data, draw_plot=True)
    x_new = wdgrl.transform(x_original)
if __name__ == '__main__':
    test()