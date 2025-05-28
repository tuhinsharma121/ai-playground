import logging
import logging.config

import uvicorn


def get_python_logger(log_level="INFO",
                      log_format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
                      log_date_format="%Y-%m-%d %H:%M:%S"):
    log_level = log_level.upper()
    logger_config = {
        'version': 1,
        'formatters': {
            "jarvis_logger": {'format': log_format,
                              'datefmt': log_date_format}
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': "jarvis_logger",
                'stream': 'ext://sys.stdout'
            },
        },
        'loggers': {
            "jarvis_logger": {
                'level': log_level,
                'handlers': ['console']
            }
        },
        'disable_existing_loggers': False
    }

    logging.config.dictConfig(config=logger_config)
    return logging.getLogger("jarvis_logger")


def get_uvicorn_log_config(log_format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
                           log_date_format="%Y-%m-%d %H:%M:%S"):
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format
    log_config["formatters"]["access"]["datefmt"] = log_date_format
    log_config["formatters"]["default"]["datefmt"] = log_date_format
    return log_config



