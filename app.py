from src.artifacts.artifact import DataIngestionArtifact,DataTransformationArtifact,ModelTrainingArtifact,HyperTuningArtifact
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainConfig,HyperTuningConfig
from src.model.model_train import ModelTrain
from src.model.hyper_tuning import Hyper_Tuning


def main():
    try:
        logging.info('Data Ingestion started ...')
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        ingestion_artifact = data_ingestion.init_data_ingestion()

    
        logging.info(f"Data Ingestion completed successfully. Data saved to: {ingestion_artifact.data_path}")
        logging.info(f"Training data saved to: {ingestion_artifact.train_path}")
        logging.info(f"Testing data saved to: {ingestion_artifact.test_path}")

        logging.info('Data Transformation started ... ')
        data_transformation_config= DataTransformationConfig()
        data_transformation= DataTransformation(data_transformation_config)
        data_transformation.init_data_transformation(DataIngestionArtifact)
        logging.info('Data Transformation completed. ')

        logging.info('Model Training started ... ')
        model_training_config=ModelTrainConfig()
        model_training= ModelTrain(model_training_config)
        model_training.init_model_training(ModelTrainingArtifact)
        logging.info('Model Training completed .')

        logging.info('Hyper Tuning started ... ')
        hyper_tuning_config=HyperTuningConfig()
        hyper_tuning= Hyper_Tuning()
        hyper_tuning.init_hyper_tuning(ModelTrainingArtifact,DataTransformationArtifact)
        logging.info(f'the best model was {HyperTuningArtifact.best_model}')
        logging.info(f'the best params was {HyperTuningArtifact.good_params}')
        logging.info(f'r2 score : {HyperTuningArtifact.r2_score}')
        logging.info(f'mse score : {HyperTuningArtifact.mse_score}')
        
        logging.info('Hyper Tuning completed.')



    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
