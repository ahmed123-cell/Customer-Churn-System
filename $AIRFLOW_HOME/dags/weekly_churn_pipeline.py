"""
dags/weekly_churn_pipeline.py

Weekly Airflow DAG for the Telco Customer Churn project:

    pull_data
        -> validate_data          (tests/test_data.py)
        -> train_models           (train.py: trains + logs every model to MLflow)
        -> select_best_model      (reads artifacts/model_comparison.csv)
        -> export_best_model      (save_model.py: best model -> ONNX + scaler + feature_names)
        -> run_integration_tests  (tests/integration_test.py, real pipeline end-to-end)
        -> version_artifacts      (dvc add + dvc push on the new model artifacts)
        -> restart_api            (restart the serving container so it loads the new model)

Each stage after `pull_data` runs inside the same `churn-api` Docker image
built by the project's Dockerfile, so the Airflow worker itself only needs
`docker`, `dvc`, and `pandas` installed — not the full ML stack (xgboost,
onnxruntime, etc.). If you'd rather run these as plain BashOperators against
a local virtualenv instead of Docker, swap DockerOperator for BashOperator
using the same commands; everything else (ordering, XCom, schedule) stays
the same.

Configuration is read from Airflow Variables (Admin -> Variables in the
UI, or `airflow variables set <key> <value>`), so nothing here needs
editing per-environment:

    churn_project_dir     Absolute path to the project root on the Airflow
                           worker host (default: /opt/airflow/projects/churn)
    churn_docker_image     Image built from the project's Dockerfile
                           (default: churn-api:latest)
    churn_api_container    Name of the *running* serving container to
                           restart after a successful export
                           (default: churn-api)

Requires (on the Airflow worker, not inside the pipeline's own image):
    pip install apache-airflow-providers-docker pandas

Place this file in $AIRFLOW_HOME/dags/, then enable "weekly_churn_pipeline"
in the Airflow UI.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


PROJECT_DIR = Variable.get("churn_project_dir", default_var="/opt/airflow/projects/churn")
DOCKER_IMAGE = Variable.get("churn_docker_image", default_var="churn-api:latest")
API_CONTAINER = Variable.get("churn_api_container", default_var="churn-api")

DATA_CSV = "data/Telco-Customer-Churn.csv"

# Mounts shared by every DockerOperator task below: the pipeline's own
# container paths (/app/...) map onto the same host directories the
# project's `docker run` commands already use, so artifacts, MLflow runs,
# and configs persist across tasks and across weekly runs.
PROJECT_MOUNTS = [
    Mount(source=f"{PROJECT_DIR}/data", target="/app/data", type="bind"),
    Mount(source=f"{PROJECT_DIR}/artifacts", target="/app/artifacts", type="bind"),
    Mount(source=f"{PROJECT_DIR}/configs", target="/app/configs", type="bind"),
    Mount(source=f"{PROJECT_DIR}/mlruns", target="/app/mlruns", type="bind"),
]

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _docker_task(task_id: str, command: str, **kwargs) -> DockerOperator:
    """Shared factory for the pipeline stages that run inside churn-api's
    own image, so every task gets identical mounts/image/cleanup behavior
    without repeating them at each call site.
    """
    return DockerOperator(
        task_id=task_id,
        image=DOCKER_IMAGE,
        command=command,
        mounts=PROJECT_MOUNTS,
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        **kwargs,
    )


def _select_best_model() -> str:
    """Read the comparison table train.py just wrote and return the model
    name with the highest test ROC-AUC. The return value is automatically
    pushed to XCom (key='return_value'), so downstream tasks can pull it
    via Jinja templating.
    """
    results = pd.read_csv(f"{PROJECT_DIR}/artifacts/model_comparison.csv", index_col="model")
    best_model = results["test_roc_auc"].idxmax()
    print(f"Best model this run: '{best_model}' (test_roc_auc={results.loc[best_model, 'test_roc_auc']:.3f})")
    return best_model


with DAG(
    dag_id="weekly_churn_pipeline",
    description="Weekly retrain + export + redeploy of the Telco churn model",
    default_args=default_args,
    schedule="@weekly",  # every Sunday at midnight; use a cron string (e.g. "0 3 * * 1" for Monday 3am) to customize
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["churn", "ml-pipeline"],
) as dag:

    pull_data = BashOperator(
        task_id="pull_data",
        bash_command=f"cd {PROJECT_DIR} && dvc pull",
    )

    validate_data = _docker_task(
        task_id="validate_data",
        command="pytest tests/test_data.py -v",
    )

    train_models = _docker_task(
        task_id="train_models",
        command=f"python train.py --data {DATA_CSV}",
    )

    select_best_model = PythonOperator(
        task_id="select_best_model",
        python_callable=_select_best_model,
    )

    export_best_model = _docker_task(
        task_id="export_best_model",
        command=(
            f"python save_model.py --data {DATA_CSV} "
            "--model {{ ti.xcom_pull(task_ids='select_best_model') }}"
        ),
    )

    run_integration_tests = _docker_task(
        task_id="run_integration_tests",
        command="pytest tests/integration_test.py -v -m integration",
    )

    version_artifacts = BashOperator(
        task_id="version_artifacts",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "dvc add artifacts/*.onnx artifacts/scaler.joblib artifacts/feature_names.json && "
            "dvc push && "
            "git add artifacts/*.dvc && "
            "git commit -m \"Weekly retrain: $(date +%Y-%m-%d)\" && "
            "git push"
        ),
    )

    restart_api = BashOperator(
        task_id="restart_api",
        bash_command=f"docker restart {API_CONTAINER}",
    )

    (
        pull_data
        >> validate_data
        >> train_models
        >> select_best_model
        >> export_best_model
        >> run_integration_tests
        >> version_artifacts
        >> restart_api
    )