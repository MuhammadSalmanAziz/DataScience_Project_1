import sys
from src.mlproject.logger import logging 
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.mlproject.components.model_trainer import ModelTrainerConfig,ModelTrainer

if __name__ == "__main__":
    logging.info("The execution has started")
    try:
        data_ingestion = DataIngestion()
        train_df_path , test_df_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr ,_= data_transformation.initiate_data_transformation(train_df_path,test_df_path)

        #model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)