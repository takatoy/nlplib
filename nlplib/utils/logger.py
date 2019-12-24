import sys
from loguru import logger


def set_logger(path=None):
    logger.remove(0)
    frmt = (
        "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | "
        "<c>{name}:{function}:{line}</c> - <level>{message}</level>"
    )
    logger.add(sys.stderr, format=frmt, level="INFO")
    if path:
        logger.add(path, format=frmt, level="INFO")
