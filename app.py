# from src.mlproject.logger import logging

# if __name__=="__main__":
#     logging.info("The execution has started")

# from src.mlproject.exception import CustomException
# from src.mlproject.logger import logging
# import sys

# if __name__=="__main__":
#     logging.info("The execution has started")

#     try:
#         a=1/0

#     except Exception as e:
#         logging.info("Custom Exception")
#         raise CustomException(e,sys)


from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig

from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.model_tranier import ModelTrainerConfig, ModelTrainer

import sys


# import dagshub
# dagshub.init(
#     repo_owner="polashds",
#     repo_name="study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws",
#     mlflow=True
# )


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        ## Model Training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)