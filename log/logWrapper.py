import logging
import log.logConfig as config
import os

class customlogger:
	def __init__(self, logger):
		self.logger = logger
		self.logger.setLevel(logging.INFO)
        #self.logger.setLevel(config.LOG_CONFIG['LEVEL'][str(os.environ.get("LOGLEVEL","INFO"))])
		# create console handler and set level to debug
		ch = logging.StreamHandler()
		ch.setLevel(config.LOG_CONFIG['LEVEL'][str(os.environ.get("LOGLEVEL","INFO"))])
        # create formatter
		formatter = logging.Formatter(config.LOG_CONFIG['MESSAGE_FORMAT']['OUTPUT_FORMAT'], config.LOG_CONFIG['MESSAGE_FORMAT']['TIME_FORMAT'])
		# add formatter to ch
		ch.setFormatter(formatter)
        # add ch to logger
		self.logger.addHandler(ch)

	def getLogger(self):
		return self.logger