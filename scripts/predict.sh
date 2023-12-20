#!/bin/bash

# Define the list of date strings
dates=('20220420' '20220601' '20220608' '20220629' '20220714')

# Iterate through the list and perform actions for each date
for date in "${dates[@]}"; do
    echo "Processing date: $date"
    python3 predict.py -seq $date
done
