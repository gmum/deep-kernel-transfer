#!/usr/bin/env bash
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

unzip household_power_consumption.zip -d .
mv household_power_consumption.txt household_power_consumption.csv