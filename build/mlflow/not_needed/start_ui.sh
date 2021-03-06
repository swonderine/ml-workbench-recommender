#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# mlflow db upgrade $DB_URI

mlflow ui \
    --backend-store-uri $DB_URI \
    --host 0.0.0.0 \
    --port 81 \
    --default-artifact-root s3://mlflow/


# --default-artifact-root $MLFLOW_ARTIFACT_ROOT
