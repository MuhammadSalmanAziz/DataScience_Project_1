import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifact","train.csv")
    test_data_path:str = os.path.join("artifact","test.csv")
    raw_data_path:str = os.path.join("artifact","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading from sql database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            pass
        except Exception as e:
            raise CustomException(e,sys)