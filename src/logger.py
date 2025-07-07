import logging

_PROJECT_LOGGER_NAME = "firechat"


def get_logger():
    """
    Helper function to set up logger for print-only logging.
    """
    _logger = logging.getLogger(_PROJECT_LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(filename)s: %(funcName)s %(lineno)d: %(message)s",
        datefmt="%Y/%m/%d-%I:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)

    _logger.propagate = False

    return _logger


logger = get_logger()
