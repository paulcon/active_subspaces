#!/usr/bin/env python

import matplotlib
import nose

if __name__ == '__main__':
	matplotlib.use('agg')
	nose.main(argv = ['--processes=1', '-v'])
