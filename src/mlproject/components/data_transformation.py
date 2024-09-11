import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os

@dataclass
class DataTransformationConfig:
    # setting preprocessor Path
    preprocesser__obj_path_file = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        """
        This function is for data transformation

        """
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy= 'median')),
                ('scalar',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder()),
                ("scalar",StandardScaler(with_mean=False))
            ])
            logging.info(f"Numerical Columns : {num_features}")
            logging.info(f"Categorical Columns : {cat_features}")


            preprocessor = ColumnTransformer([
                ("numerical Pipeline", num_pipeline, num_features),
                ('categorical Pipeline', cat_pipeline,cat_features)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and test data")

            preprocessing_obj = self.get_data_transformer_obj()
            
            target_column_name = "math_score"
            num_columns = ['reading_score', 'writing_score']

            input_features_train_df = train_df.drop(target_column_name , axis= 1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(target_column_name , axis= 1)
            target_feature_test_df = test_df[target_column_name]


            logging.info("Applying Preprocessing on training and testing data")
            input_features_train_arr= preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr= preprocessing_obj.transform(input_features_test_df)


            train_arr = np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved Preprocessing Obj")

            save_object(
                file_path= self.data_transformation_config.preprocesser__obj_path_file,
                obj= preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser__obj_path_file
            )

        except Exception as e:
            raise CustomException(e,sys)
 