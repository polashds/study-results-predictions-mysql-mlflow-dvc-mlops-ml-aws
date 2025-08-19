import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import warnings

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    min_r2_threshold: float = 0.6  # Minimum acceptable R2 score

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.mlflow_tracking_uri = "https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow"

    def eval_metrics(self, actual, pred):
        """Safely calculate evaluation metrics with error handling"""
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            # Convert numpy types to native Python types for MLflow compatibility
            return float(rmse), float(mae), float(r2)
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            return float('inf'), float('inf'), -float('inf')

    def setup_mlflow(self):
        """Safely setup MLflow tracking with fallback"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logging.info(f"MLflow tracking URI set to: {self.mlflow_tracking_uri}")
            
            # Test the connection
            try:
                mlflow.get_tracking_uri()
                return True
            except Exception as conn_error:
                logging.warning(f"MLflow connection test failed: {conn_error}. Using local tracking.")
                return False
                
        except Exception as e:
            logging.warning(f"Failed to set MLflow tracking URI: {e}. Using local tracking.")
            return False

    def is_dagshub_uri(self):
        """Check if the tracking URI is DagsHub"""
        try:
            return "dagshub" in self.mlflow_tracking_uri.lower()
        except:
            return False

    def log_model_safely(self, model, model_name, X_test, y_test, best_params):
        """Safely log model to MLflow with DagsHub compatibility"""
        try:
            # Calculate metrics
            predicted_qualities = model.predict(X_test)
            rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)
            
            with mlflow.start_run() as run:
                # Log basic information
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("mlflow.runStatus", "SUCCESS")
                
                # Log parameters
                if best_params:
                    for param_name, param_value in best_params.items():
                        if isinstance(param_value, (list, tuple)) and param_value:
                            # Log the first value for hyperparameter tuning visualization
                            mlflow.log_param(param_name, param_value[0])
                        elif param_value:
                            mlflow.log_param(param_name, param_value)
                
                # Log metrics (ensure native Python types)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Log model - simplified for DagsHub compatibility
                try:
                    # Use the modern parameter naming
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        artifact_path="model",
                        registered_model_name=model_name if not self.is_dagshub_uri() else None
                    )
                except Exception as model_log_error:
                    # Fallback for any model logging issues
                    logging.warning(f"Model logging failed with modern API: {model_log_error}")
                    try:
                        # Try the older API
                        mlflow.sklearn.log_model(model, "model")
                    except Exception as fallback_error:
                        logging.error(f"Fallback model logging also failed: {fallback_error}")
                
                logging.info(f"Model logged successfully: {model_name}")
                
                return rmse, mae, r2
                
        except Exception as e:
            logging.error(f"Failed to log model to MLflow: {e}")
            # Still calculate metrics even if MLflow fails
            predicted_qualities = model.predict(X_test)
            return self.eval_metrics(y_test, predicted_qualities)

    def validate_data(self, train_array, test_array):
        """Validate input data before processing"""
        if train_array is None or test_array is None:
            raise CustomException("Train or test array is None")
        
        if len(train_array) == 0 or len(test_array) == 0:
            raise CustomException("Train or test array is empty")
        
        if train_array.shape[1] < 2 or test_array.shape[1] < 2:
            raise CustomException("Arrays don't have enough features")

    def initiate_model_trainer(self, train_array, test_array):
        """Main method to initiate model training with comprehensive error handling"""
        try:
            # Validate input data
            self.validate_data(train_array, test_array)
            
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Validate split data
            if len(X_train) == 0 or len(X_test) == 0:
                raise CustomException("Empty training or test features after split")
            
            if len(y_train) == 0 or len(y_test) == 0:
                raise CustomException("Empty training or test targets after split")

            # Define models with safe parameters
            models = {
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "XGBRegressor": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(random_state=42, verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.8, 0.9, 1.0],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5],
                    'n_estimators': [50, 100, 200],
                }
            }

            # Setup MLflow
            mlflow_setup_success = self.setup_mlflow()
            if mlflow_setup_success:
                logging.info("MLflow tracking enabled")
            else:
                logging.info("Using local execution without MLflow tracking")

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if not model_report:
                raise CustomException("No models were successfully evaluated")

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            print(f"This is the best model: {best_model_name} with score: {best_model_score:.4f}")

            # Get best parameters (use empty dict if none found)
            best_params = params.get(best_model_name, {})

            # Log to MLflow if setup was successful
            if mlflow_setup_success:
                rmse, mae, r2 = self.log_model_safely(best_model, best_model_name, X_test, y_test, best_params)
            else:
                # Calculate metrics without MLflow
                predicted = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted)

            # Validate model performance
            if r2 < self.model_trainer_config.min_r2_threshold:
                raise CustomException(f"No adequate model found. Best R2 score: {r2:.4f} "
                                    f"(minimum required: {self.model_trainer_config.min_r2_threshold})")

            logging.info(f"Best model found: {best_model_name} with R2 score: {r2:.4f}")

            # Save model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Model saved successfully to {self.model_trainer_config.trained_model_file_path}")

            # Return results with native Python types
            return {
                "model_name": best_model_name,
                "r2_score": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "model_path": self.model_trainer_config.trained_model_file_path,
                "mlflow_tracking": mlflow_setup_success
            }

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            # Ensure MLflow run is marked as failed if it was started
            try:
                mlflow.set_tag("mlflow.runStatus", "FAILED")
            except:
                pass
            raise CustomException(f"Model training failed: {e}")

# Optional: Add a main guard for testing
if __name__ == "__main__":
    # Example usage for testing
    try:
        trainer = ModelTrainer()
        # You would normally pass actual data here
        # result = trainer.initiate_model_trainer(train_data, test_data)
        # print(result)
    except Exception as e:
        print(f"Error: {e}")