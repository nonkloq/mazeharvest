#!/bin/bash
# Remote VM Model Training Env Setup Script

pip install tensorboard tqdm numpy
cd homegym/ 
pip install -e . 
cd ..
cd experiments/
