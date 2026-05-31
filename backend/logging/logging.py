import logging

# Create a specific logger instead of using basicConfig
logger = logging.getLogger("paper_search")
logger.setLevel(logging.DEBUG)  

# Add a file handler to this specific logger only
file_handler = logging.FileHandler("error_logs.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

# Prevent propagation to root logger to avoid capturing framework logs
logger.propagate = False
logger.addHandler(file_handler)