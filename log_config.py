import logging

loglevel = logging.DEBUG
logfile = '/tmp/luigi_workshop.log'

log_format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
log_date_format = '%H:%M:%S'


def add_handlers_and_init(logger, log_format, log_date_format, loglevel, logfile):
    logger.setLevel(loglevel)

    formatter = logging.Formatter(log_format, datefmt=log_date_format)

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)