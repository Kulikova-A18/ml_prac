#!/bin/bash

deactivate || echo skip deactivate

python3 -m venv myenv

source myenv/bin/activate

pip uninstall pandas matplotlib seaborn scikit-learn tensorflow -y

pip install -r requirements.txt

python3 anomaly_detection.py
