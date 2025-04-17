import logging
import os

from dotenv import load_dotenv

# Load the users .env file into environment variables
load_dotenv(verbose=True, override=True)

del load_dotenv

logger = logging.getLogger(__name__)
