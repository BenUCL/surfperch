#!/bin/bash
# This script runs the cpu_execution_time.py script with different numbers of threads

# Define the array of thread numbers
THREADS=(1 4 8 12 16)

# Loop through each thread count
for num_threads in "${THREADS[@]}"
do
    echo "" 
    python cpu_execution_time.py --num_threads $num_threads
done
