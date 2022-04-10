import traceback
import logging
import json
import logging.handlers
import os

LOG_FILE_PATH = os.path.dirname(__file__) + "/../log/deep-recommender.log"
LOG_EXCEPTION_FILE_PATH = os.path.dirname(__file__) + "/../log/deep-recommender_exceptions.log"
LOG_FILE_MAX_SIZE_BYTE = 1024 * 1024 * 1024  #1GiB
LOG_FILE_ROTATION_BACKUP = 1


class JSONFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.not_used_attrs = set(["exc_info", "exc_text", "created", "msecs", "process", "processName", "stack_info",
                                   "thread", "threadName", "levelname", "levelno", "args", "filename", "relativeCreated"])

    def format(self, record: logging.LogRecord):
        record_dict = record.__dict__
        record_dict = self._get_jsonable_dict(record_dict)

        msg_dict = {}
        for attr_name in record_dict:
            if attr_name not in self.not_used_attrs:
                msg_dict[attr_name] = record_dict[attr_name]

        if "asctime" not in msg_dict:
            msg_dict["asctime"] = self.formatTime(record)

        return json.dumps(msg_dict, default=str)

    def _get_jsonable_dict(self, record_dict):
        new_dict = {}
        for attr_name in record_dict:
            if isinstance(attr_name, str):
                new_attr_name = attr_name
            else:
                new_attr_name = str(attr_name)

            if isinstance(record_dict[attr_name], dict):
                new_dict[new_attr_name] = self._get_jsonable_dict(record_dict[attr_name])
            else:
                new_dict[new_attr_name] = record_dict[attr_name]
        return new_dict


def initialize_log(log_file_path=LOG_FILE_PATH, log_exception_file_path=LOG_EXCEPTION_FILE_PATH):
    log_dir = os.path.dirname(log_file_path)
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes=LOG_FILE_MAX_SIZE_BYTE, backupCount=LOG_FILE_ROTATION_BACKUP)
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    exp_logger = logging.getLogger('tfrs_exceptions')
    exp_logger.setLevel(logging.ERROR)
    exp_file_handler = logging.handlers.RotatingFileHandler(log_exception_file_path, maxBytes=LOG_FILE_MAX_SIZE_BYTE, backupCount=LOG_FILE_ROTATION_BACKUP)
    exp_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    exp_file_handler.setFormatter(exp_formatter)
    exp_logger.addHandler(exp_file_handler)


def log_exception():
    exp_logger = logging.getLogger('_exceptions')
    exp_logger.error(traceback.format_exc())