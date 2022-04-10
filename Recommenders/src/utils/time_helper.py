import time
import logging
from functools import wraps


def log_time_cost(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        results = f(*args, **kwds)
        end_time = time.time()
        logging.info({
            "log_type": "FUNC_TIME_COST",
            "function": f.__name__,
            "cost_sec": (end_time - start_time)
        })
        return results
    return wrapper