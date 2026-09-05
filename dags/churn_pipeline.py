from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# 1. Define the DAG
my_dag = DAG(
    dag_id='churn_prediction_pipeline',
    start_date=datetime(2026, 8, 15),
    schedule=None,
    catchup=False
)

# 2. Task 1: Preprocess Data
preprocess_task = BashOperator(
    task_id="preprocess_data",
    bash_command="cd /home/aryan/ml_flow && export PYTHONPATH=/home/aryan/ml_flow && python /home/aryan/ml_flow/src/preprocess.py 1",
    dag=my_dag
)

# 3. Task 2: Train Model
train_task = BashOperator(
    task_id="train_model",
    bash_command= "cd /home/aryan/ml_flow && export PYTHONPATH=/home/aryan/ml_flow && python /home/aryan/ml_flow/src/train.py",
    dag=my_dag
)

# 4. Set Task Dependencies
preprocess_task >> train_task
