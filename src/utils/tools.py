import os
import joblib
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from src.logging.logger import logging


def read_csv_file(path:str):
    try:
        
        df=pd.read_csv(path)

        return df
    except Exception as e:
        logging.error(f"Failed to read CSV file at {path}: {e}")
        raise e

def save_csv_file(data:pd.DataFrame,path:str):
    try:
        
        data.to_csv(path,index=False,header=True)
    except Exception as e:
        logging.error(f"Failed to save CSV file at {path}: {e}")
        raise e


def read_pkl_file(path:str):
    try:
        
        with open(path,'rb') as f:
            file=joblib.load(f)
        return file
    except Exception as e:
        logging.error(f"Failed to read pickle file at {path}: {e}")
        raise e

def save_pkl_file(data,path:str):
    try:
        
        with open(path,'wb') as f:
            joblib.dump(data,f)
    except Exception as e:
        logging.error(f"Failed to save pickle file at {path}: {e}")
        raise e

def drop_columns(data:pd.DataFrame,columns:list):
    try:
        df=data.drop(columns, axis=1)
        return df
    except Exception as e:
        logging.error(f"Failed to drop columns {columns}: {e}")
        raise e


def change_df_type(data: pd.DataFrame, column: str, df_to) -> pd.DataFrame:
    try:
        
        if column not in data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        
        if df_to == 'datetime':
            data[column] = pd.to_datetime(data[column], errors='coerce')
        else:
            data[column] = data[column].astype(df_to)
        
        return data
    except Exception as e:
        logging.error(f"Failed to convert column '{column}' to {df_to}: {e}")
        raise e


def data_split(data:pd.DataFrame,split_size,random_state:int):
    try:
        train,test= train_test_split(data,test_size=split_size,random_state=random_state)
        return train,test
    except Exception as e:
        logging.error(f"Failed to split data with split size {split_size}: {e}")
        raise e




def save_hyper_params(params: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
        logging.info(f"Hyperparameter tuning parameters saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save hyperparameter tuning parameters at {file_path}: {e}")
        raise e

       
