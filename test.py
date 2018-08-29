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
    wdgrl = WDGRL(input_dim=784)
    wdgrl.fit(usps_data, mnist_data)

if __name__ == '__main__':
    test()