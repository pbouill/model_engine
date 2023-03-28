import logging
import coloredlogs

DEFAULT_FORMAT = ('%(asctime)s.%(msecs)06d :: %(levelname)s :: '
                  '%(name)s :: %(module)s.%(funcName)s:%(lineno)d - %(message)s')

logging.basicConfig(format=DEFAULT_FORMAT)
logger = logging.getLogger(__name__)
coloredlogs.install(fmt=DEFAULT_FORMAT, logger=logger, level='INFO')
logger.setLevel(logging.DEBUG)