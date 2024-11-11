from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
        data_path:str
        train_path:str
        test_path:str

@dataclass
class DataTransformationArtifact:
        preprocess_path:str
        train_path:str
        test_path:str

@dataclass
class ModelTrainingArtifact:
        model_name:str


@dataclass
class HyperTuningArtifact:
        best_model:str
        good_params:str
        mse_score:str
        r2_score:str

    
    