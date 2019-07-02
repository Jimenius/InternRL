import logging
import os
import sys

class Logger(object):
    '''
    Description:
        General agent super class
    '''

    def __init__(self, name, logdir = '', logfile = ''):
        self._logger = logging.getLogger(name = name)
        self._logger.setLevel(logging.INFO) # Change the logging threshold to INFO

        # StreamHandler puts message to stdout
        _sh = logging.StreamHandler(stream = sys.stdout)
        _form = logging.Formatter("%(asctime)s: %(message)s")
        _sh.setFormatter(_form)
        self._logger.addHandler(_sh)

        # FileHandler puts message to file
        if logdir and logfile:
            os.makedirs(logdir, exist_ok = True)
            _fh = logging.FileHandler(os.path.join(logdir, logfile))
            _fh.setFormatter(_form)
            self._logger.addHandler(_fh)
        
    def __call__(self, message):
        self._logger.info(message)