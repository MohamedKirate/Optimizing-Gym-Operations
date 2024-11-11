from src.entity.entity import HyperTuningConfig
from src.artifacts.artifact import HyperTuningArtifact,ModelTrainingArtifact,DataTransformationArtifact
from src.utils.tools import save_pkl_file,read_csv_file,read_pkl_file,save_hyper_params
from src.logging.logger import logging


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoost
from sklearn.metrics import mean_squared_error,r2_score
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json


class Hyper_Tuning:
    def __init__(self,hyper_tuning_config:HyperTuningConfig):
        self.hyper_tuning_config= hyper_tuning_config
    
    def get_model_params(self,model_name:str,model_params:dict,models:dict) -> dict :
        try:
            if model_name in model_params and model_name in models:
                params=model_params[model_name]
                return params
            else:
                logging.info(f"Model {model_name} is not recognized.")
                return None
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
    
    def evaluate(self,y_true,y_predict):
        try:
            mse= mean_squared_error(y_true,y_predict)
            r2= r2_score(y_true,y_predict)

            return mse,r2
        except Exception as e:
            logging.error(f'An error occurred: {e}')
            raise e
        
    def split_data(self,data:pd.DataFrame,target):
        try:
            x_train= data.drop(columns=[target],axis=1)
            y_train=data[target]

            return x_train,y_train
        except Exception as e:
            logging.error(f"An error occurred: {e} ")
            raise e
    
    def save_model_with_params(self,model,model_path,params,params_path):
        try:
            save_pkl_file(model,model_path)
            save_hyper_params(params,params_path)
            logging.info(f"Model and parameters saved successfully.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def hyper_tyning(self,data_path:str,target,model_path:str,params:dict,params_path:str):
        try:
            
            data= read_csv_file(data_path)
            x_train,y_train=self.split_data(data_path,target)

            model=read_pkl_file(model_path)

            gs=GridSearchCV(model,params,cv=3,scoring='r2', n_jobs=-1)
            gs.fit(x_train,y_train)

            y_predict=gs.best_estimator_.predict(x_train)

            mse_score,R2_score=self.evaluate(y_train,y_predict)
            
            logging.info("Hyperparameter tuning completed successfully.")

            self.save_model_with_params(gs.best_estimator_,model_path,gs.best_params_,params_path)

            return gs.best_estimator_, gs.best_params_,mse_score,R2_score


        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

    def init_hyper_tuning(self,model_training_artifact:ModelTrainingArtifact,data_transformation_artifact:DataTransformationArtifact):
        try:
            mlflow.start_run()
            params=self.get_model_params(
                model_training_artifact.model_name,
                self.hyper_tuning_config.model_params,
                self.hyper_tuning_config.models
            )

            best_model, best_params,mse_score,R2_score=self.hyper_tyning(data_transformation_artifact.train_path,
                            self.hyper_tuning_config.model_target,
                            self.hyper_tuning_config.best_model_path,
                            params,
                            self.hyper_tuning_config.best_model_params_path)
            mlflow.sklearn.load_model(best_model,'best model')
            mlflow.log_params(best_params)
            mlflow.log_metric("MSE", mse_score)
            mlflow.log_metric("R2", R2_score)

            logging.info("Hyperparameter tuning and evaluation completed successfully.")
            mlflow.end_run()
            
            return HyperTuningArtifact(
                self.hyper_tuning_config.best_model_path,
                self.hyper_tuning_config.best_model_params_path,
                mse_score,
                R2_score
            )
        except Exception as e:
            logging.error(f"An error occurred during initialization of hyperparameter tuning: {e}")
            mlflow.end_run(status="FAILED")
            raise e


