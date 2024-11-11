import os
import pandas as pd
import numpy as np
from datetime import datetime


from src.entity.entity import DataIngestionConfig
from src.artifacts.artifact import DataIngestionArtifact

from src.utils.tools import read_csv_file,save_csv_file,drop_columns,change_df_type,data_split
from src.logging.logger import logging


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config= data_ingestion_config

    def merge_dataframes(self):
        try:
            history_df= read_csv_file(self.data_ingestion_config.history_df)
            users_df=read_csv_file(self.data_ingestion_config.users_df)
            location_df= read_csv_file(self.data_ingestion_config.location_df)
            subplan_df=read_csv_file(self.data_ingestion_config.subplan_df)

            df= users_df.merge(subplan_df,left_on='subscription_plan',right_on='subscription_plan',how='left')
            df= df.merge(history_df,left_on='user_id',right_on='user_id',how='left')
            df= df.merge(location_df,left_on=['user_location','gym_id'],right_on=['location','gym_id'],how='left')

            return df
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

    def change_df_types(self,data:pd.DataFrame,columns:dict):
        try:
            for column, change_info in columns.items():

                df_to = change_info.get('to')

                if column in data.columns:
                    data=change_df_type(data,column,df_to)
            return data
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        

        
    
    def feature_engineering(self,data:pd.DataFrame):
        try:
            data= self.change_df_types(data,self.data_ingestion_config.coulmns_changing)

            data['time_spend']= data['checkout_time']-data['checkin_time']
            def timespend(x):
                hours = x.components.hours
                minutes = x.components.minutes
                if hours > 0:
                    return f'{hours}h {minutes}m'
                else:
                    return f'{minutes}m'
                
            data['time_spend']= data['time_spend'].apply(timespend)
            data['time_spend_hours']= data['checkout_time']-data['checkin_time']
            data['time_spend_hours']= data['time_spend_hours'].dt.total_seconds() / 3600

            data['membership_duration']= datetime.now() - data['sign_up_date']
            data['membership_duration_days'] = data['membership_duration'].dt.days

            data['last_date']= datetime.now()

            data=drop_columns(data,self.data_ingestion_config.useless_columns)
            return data
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
    

    def init_data_ingestion(self):
        try:
            df=self.merge_dataframes()
            df= self.feature_engineering(df)
            save_csv_file(df,self.data_ingestion_config.data_path)

            train,test=data_split(df,self.data_ingestion_config.split_size,self.data_ingestion_config.random_state)

            save_csv_file(train,self.data_ingestion_config.train_path)
            
            save_csv_file(test,self.data_ingestion_config.test_path)

            data_ingestion_artifact=DataIngestionArtifact(
                data_path=self.data_ingestion_config.data_path,
                train_path=self.data_ingestion_config.train_path,
                test_path=self.data_ingestion_config.test_path
                                            )
            
            logging.info(f'data ingestion artifact data path{data_ingestion_artifact.data_path}')
            logging.info(f'data ingestion artifact train path{data_ingestion_artifact.train_path}')
            logging.info(f'data ingestion artifact test path{data_ingestion_artifact.test_path}')
            return data_ingestion_artifact
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e
        




