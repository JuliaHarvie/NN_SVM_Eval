#!/bin/sh

python3 -m pip install -r requirements.txt
python3 FoldsCreation.py Images 15 jpg
python3 NNEvaluation.py 10
python3 SVMEvaluation.py 10
