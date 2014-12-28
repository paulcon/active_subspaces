import numpy as np
import gaussian_quadrature as gq



if __name__ == '__main__':
    x,w = gq.gauss_hermite([4,3,2])
    print x
    print w