#!/usr/bin/env bash


python run_experiment.py mnist wb
python run_experiment.py mnist wb_kernel
python run_experiment.py mnist kernel
python run_experiment.py mnist nn
python plotting.py
