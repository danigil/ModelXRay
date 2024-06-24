import sys

def request_logger(logger_name: str = None, dump_to_sysout=True):
    if logger_name is None:
        logger_name = __name__

    import logging

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if dump_to_sysout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_it(exit_on_fail=False):
    logger = request_logger()
    def _log_it(func):
        def wrap(*args, **kwargs):
            func_name = func.__name__
            
            args_str = ', '.join([str(arg) for arg in args])
            kwargs_str = ' | '.join(map(lambda tup: f'{tup[0]}:{tup[1]}', sorted(kwargs.items())))

            run_str = f'running {func_name} with args: [{args_str}] | kwargs: {kwargs_str}'
            logger.info(run_str)

            try:
                ret = func(*args, **kwargs)
            except:
                logger.exception(f'~~failed~~ {run_str}')
                if exit_on_fail:
                    sys.exit(1)
                return None

            logger.info(f'~~finished~~ {run_str}')
            return ret
        return wrap
    return _log_it