from active_subspaces.base import *
import logging

# remove stderr logger
logging.getLogger(__name__).addHandler(logging.NullHandler())
