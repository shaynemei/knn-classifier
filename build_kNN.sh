#!/bin/sh

if [ -z "$1" ]; then
	echo "#Usage: build_kNN.sh training_data test_data k_val similarity_func sys_output > acc_file"
else
	/opt/python-3.6/bin/python3 build_knn.py $1 $2 $3 $4 $5 $6
	#python3 build_knn.py $1 $2 $3 $4 $5 $6
fi