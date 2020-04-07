LOG_CONFIG = {
    'MESSAGE_FORMAT' : {
        'OUTPUT_FORMAT' : '[Date: %(asctime)s][Level: %(levelname)s][File: %(filename)s][Function: %(funcName)s][Line: %(lineno)d][MSG: %(message)s]',
        'TIME_FORMAT' : '%Y-%m-%d %H:%M:%S',
        'FORMAT': '[Date %(asctime)s][Level %(levelname)s]%(message)s'
    },
    'LOGGER' : "console_logger",
    'DEFAULT_LOG_LEVEL' : "INFO",
    'LEVEL' : {
        'CRITICAL' : 50,
        'ERROR' : 40,
        'WARNING' : 30,
        'INFO' : 20,
        'DEBUG' : 10,
        'NOTESET' : 0
    }
}