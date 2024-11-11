import os
from typing import Dict,List
from src.constants.constants import (INGESTION_INPUT_DIR,INGESTION_INPUT_HISTORY,INGESTION_INPUT_LOCATION,INGESTION_INPUT_SUBPLAN,
                                     INGESTION_INPUT_USERS,INGESTION_OUTPUT_DIR,INGESTION_OUTPUT_DATA,INGESTION_OUTPUT_TRAIN,
                                     INGESTION_OUTPUT_TEST,
                                     INGESTION_USELESS_COLUMNS,INGESTION_FEATURES_CHANGE_TYPES)
from src.constants.constants import (DATA_TRAINSFORMATION_COLUMNS_NUMERICAL,DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_TF,
                                     DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_ONEHOT,DATA_TRAINSFORMATION_PREPROCESS)
from src.constants.constants import MODEL_TRAIN_MODELS,MODEL_TRAIN_TARGET,MODEL_TRAIN_HYPERPARAMS,MODEL_TRAIN_RANDOM_STATE,MODEL_TRAIN_SPLIT_SIZE

from src.constants.constants import MODEL_TRAIN_HYPERPARAMS,HYPER_TUNING_MODEL_PATH,HYPER_TUNING_MODEL_PARAMS_PATH,HYPER_TUNING_MODEL_TARGET

class DataIngestionConfig:
    def __init__(self):
        
        self.input_dir:str= INGESTION_INPUT_DIR
        self.output_dir= INGESTION_OUTPUT_DIR
        self.history_df:str= os.path.join(self.input_dir,INGESTION_INPUT_HISTORY)
        self.users_df:str= os.path.join(self.input_dir,INGESTION_INPUT_USERS)
        self.location_df:str= os.path.join(self.input_dir,INGESTION_INPUT_LOCATION)
        self.subplan_df:str= os.path.join(self.input_dir,INGESTION_INPUT_SUBPLAN)

        self.useless_columns: List[str]= INGESTION_USELESS_COLUMNS
        self.coulmns_changing: Dict[str,str]= INGESTION_FEATURES_CHANGE_TYPES
        self.split_size:float=MODEL_TRAIN_SPLIT_SIZE
        self.random_state:int=MODEL_TRAIN_RANDOM_STATE

        self.data_path:str  =  os.path.join(self.output_dir,INGESTION_OUTPUT_DATA)
        self.train_path:str=os.path.join(self.output_dir,INGESTION_OUTPUT_TRAIN)
        self.test_path:str=os.path.join(self.output_dir,INGESTION_OUTPUT_TEST)


class DataTransformationConfig:
    def __init__(self):
        self.num_columns:List[str]=DATA_TRAINSFORMATION_COLUMNS_NUMERICAL
        self.cat_onehot_columns:List[str]= DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_ONEHOT
        self.cat_tfidf_columns:List[str]= DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_TF
        self.preprocess_path:str= DATA_TRAINSFORMATION_PREPROCESS


class ModelTrainConfig:
    def __init__(self):
        self.models:Dict[str,any]= MODEL_TRAIN_MODELS
        self.target_column:List[str]=MODEL_TRAIN_TARGET

class HyperTuningConfig:
    def __init__(self):
        self.model_params:Dict[str,any]=MODEL_TRAIN_HYPERPARAMS
        self.best_model_params_path:str=HYPER_TUNING_MODEL_PARAMS_PATH
        self.best_model_path:str= HYPER_TUNING_MODEL_PATH
        self.models:Dict[str,any]= MODEL_TRAIN_HYPERPARAMS
        self.model_target:List[str]=HYPER_TUNING_MODEL_TARGET