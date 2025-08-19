# study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws
A end to end production-ready data science and machine learning projects with mysql, mlflow, dvc, mlops, aws functionalites




# go to python mode
import dagshub
dagshub.init(repo_owner='polashds' repo_name='study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

# now run
python app.py

2nd method"
export MLFLOW_TRACKING_URI="https://dagshub.com/polashds /study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow"
export MLFLOW_TRACKING_USERNAME="polashds"
export MLFLOW_TRACKING_PASSWORD=9f981e71d2db0a11c625ea5f111a1f6c597256dd