import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("host")
user =os.getenv("user")
password = os.getenv("password")

def read_sql_data():
    logging.info("Reading sql database started")
    try:
        pass
    except Exception as ex:
        raise CustomException(ex)
