from datetime import datetime 
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

def check_drift():
    import sys
    sys.path.insert(0, "/home/aryan/ml_flow")
    from src.drift import should_retrain

    if should_retrain(2):
        return "preprocess_data"

    else:
        return "skip_retraining"

my_dag= DAG(
    dag_id='churn_prediction_pipeline',
    start_date=datetime(2026,9,26),
    schedule=None,
    catchup=False
)

check_drift_task = BranchPythonOperator(
    task_id="check_drift",
    python_callable= check_drift,
    dag=my_dag
)

preprocess_task = BashOperator(
    task_id= "preprocess_data",
    bash_command="cd /home/aryan/ml_flow && export PYTHONPATH=/home/aryan/ml_flow && python /home/aryan/ml_flow/src/preprocess.py 1",
    dag = my_dag
)

train_task = BashOperator(
    task_id= "train_model",
    bash_command="cd /home/aryan/ml_flow && export PYTHONPATH=/home/aryan/ml_flow && python /home/aryan/ml_flow/src/train.py",
    dag= my_dag
)

skip_task= DummyOperator(
    task_id="skip_retraining",
    dag= my_dag
)

check_drift_task >> [preprocess_task, skip_task]
preprocess_task >> train_task

