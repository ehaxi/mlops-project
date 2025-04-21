import logging.config
import yaml
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/log_{timestamp}.log"

    with open("config/logging.yaml", "r") as f:
        config = yaml.safe_load(f)

    config['handlers']['file']['filename'] = log_filename

    logging.config.dictConfig(config)