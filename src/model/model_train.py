from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoost
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

from src.utils.tools import read_pkl_file,read_csv_file
from src.artifacts.artifact import DataTransformationArtifact,ModelTrainingArtifact
from src.entity.entity import ModelTrainConfig 
from src.logging.logger import logging

class ModelTrain:
    def __init__(self,model_train_config:ModelTrainConfig):
        self.model_train_config= model_train_config

    def evaluate(true,predict):
        try:
            r2=r2_score(true,predict)
            mse=mean_squared_error(true,predict)

            return r2,mse
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        
    def split_data(self,data:pd.DataFrame,target):
        try:
            x_train= data.drop(columns=[target],axis=1)
            y_train=data[target]

            return x_train,y_train
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        
    def training(self,x_train, y_train,models):
        try:
            model_name_list = []
            r2_list = []
            mse_list = []

            for model_name, model in models.items():
                
                model.fit(x_train, y_train)

                y_train_predict = model.predict(x_train)
                r2_train_score, mse_train_score = self.evaluate(y_train, y_train_predict)

                model_name_list.append(model_name)
                r2_list.append(r2_train_score)
                mse_list.append(mse_train_score)

            results_df = pd.DataFrame({
                'Model': model_name_list,
                'R2_Score': r2_list,
                'MSE': mse_list
            })

            return results_df
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        

    def init_model_training(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.data_transformation_artifact=data_transformation_artifact

            data=read_csv_file(data_transformation_artifact.train_path)
            x_train,y_train=self.split_data(data=data,target=self.model_train_config.target_column)

            peprocessing= read_pkl_file(self.data_transformation_artifact.preprocess_path)

            x_train_preprocess= peprocessing.transform(x_train)
            result= self.training(x_train_preprocess,y_train,self.model_train_config.models)

            model_name = result['Model'].iloc[result['MSE'].idxmin()]

            return ModelTrainingArtifact(
                model_name
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

