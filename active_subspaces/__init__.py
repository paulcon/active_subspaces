#from active_subspaces.base import *
import logging

# create logger
logger = logging.getLogger('PAUL')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.FileHandler('paul.log')
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)