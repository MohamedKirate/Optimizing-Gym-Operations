# Data Ingestion

INGESTION_INPUT_DIR:str='data'
INGESTION_INPUT_HISTORY:str = 'checkin_checkout_history_updated.csv' 
INGESTION_INPUT_LOCATION:str = 'gym_locations_data.csv'
INGESTION_INPUT_SUBPLAN:str = 'subscription_plans.csv'
INGESTION_INPUT_USERS:str = 'users_data.csv'
INGESTION_USELESS_COLUMNS:list= ['user_id','gym_id','first_name','last_name','birthdate','location','gym_type','facilities','membership_duration']

INGESTION_FEATURES_CHANGE_TYPES={
    'sign_up_date':{
        'to':'datetime',
    },
    'checkout_time':{
        'to':'datetime',
    },
    'checkin_time':{
        'to':'datetime',
    },

}


INGESTION_OUTPUT_DIR:str ='artifact\data'
INGESTION_OUTPUT_DATA:str= 'data.csv'
INGESTION_OUTPUT_TRAIN:str= 'train.csv'
INGESTION_OUTPUT_TEST:str= 'test.csv'


# Data Transformation

DATA_TRAINSFORMATION_COLUMNS_NUMERICAL:list=['age','time_spend_hours','membership_duration_days']
DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_TF='features'
DATA_TRAINSFORMATION_COLUMNS_CATEGORICAL_ONEHOT=['gender', 'user_location', 'subscription_plan','workout_type']
DATA_TRAINSFORMATION_PREPROCESS:str= 'artifact/preprocess_obj/preprocess.pkl'


# Model training


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoost


MODEL_TRAIN_SPLIT_SIZE:float= 0.2
MODEL_TRAIN_RANDOM_STATE:int= 42
MODEL_TRAIN_TARGET=['calories_burned']
MODEL_TRAIN_MODELS:dict={
    'RandomForestRegressor':RandomForestRegressor(),
    'XGBRegressor':XGBRegressor(),
    'CatBoost':CatBoost(),
    'GradientBoostingRegressor':GradientBoostingRegressor()
}



MODEL_TRAIN_HYPERPARAMS:dict={
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4],
    },
    'XGBRegressor': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 6],
        'min_child_weight': [1, 3],
    },
    'CatBoost': {
        'iterations': [100, 200],
        'depth': [6, 8],
        'learning_rate': [0.01, 0.1],
        'l2_leaf_reg': [1, 3],
    },
    'GradientBoostingRegressor': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }
}
HYPER_TUNING_MODEL_PATH:str= 'model/best_model/best_model.pkl'
HYPER_TUNING_MODEL_PARAMS_PATH:str= 'model/best_model/model_params.json'
HYPER_TUNING_MODEL_TARGET=['calories_burned']
