from functools import wraps
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("timeit")


def timeit(func):
    name = func.__name__

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {name} took {total_time:.4f} secs')
        return result
    return timeit_wrapper
