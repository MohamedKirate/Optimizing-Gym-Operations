import os
from src.logging.logger import logging
import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.tools import save_pkl_file,read_csv_file
from src.entity.entity import DataTransformationConfig
from src.artifacts.artifact import DataTransformationArtifact,DataIngestionArtifact
from src.logging.logger import logging

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
    
    def categorical_pipeline_for_onehot(self,columns):
        try:
            categorical_pipeline = Pipeline(
                steps=[
            ('OneHotEncoder',OneHotEncoder(drop='first')),
                    ]
            )
            return (categorical_pipeline,columns)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
    
    def categorical_pipeline_for_tfidf(self,columns):
        try:
            categorical_pipeline = Pipeline(
                steps=[
            ('TfidfVectorizer',TfidfVectorizer(max_features=50)),
                ]
            )
            return (categorical_pipeline,columns)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
    
    def numerical_pipeline(self,columns):
        try:
            numerical_pipeline=Pipeline(
                steps=[
                    ('StandardScaler',MinMaxScaler()),
                ]
            )
            return (numerical_pipeline,columns)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
    
    def pipeline_transformation(self,data:pd.DataFrame):
        try:
            preprocessing = ColumnTransformer(
                transformers=[
                    self.numerical_pipeline(self.data_transformation_config.num_columns),
                    self.categorical_pipeline_for_tfidf(self.data_transformation_config.cat_tfidf_columns),
                    self.categorical_pipeline_for_onehot(self.data_transformation_config.cat_onehot_columns)
                ]
            )
            preprocessing.fit(data)

            save_pkl_file(preprocessing,self.data_transformation_config.preprocess_path)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

    
    def init_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_ingestion_artifact= data_ingestion_artifact

            data= read_csv_file(self.data_ingestion_artifact.data_path)

            self.pipeline_transformation(data)

            data_transformation_artifact=DataTransformationArtifact(
                preprocess_path=self.data_ingestion_artifact.train_path,
                train_path=self.data_ingestion_artifact.test_path,
                test_path=self.data_transformation_config.preprocess_path
            )
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        

        