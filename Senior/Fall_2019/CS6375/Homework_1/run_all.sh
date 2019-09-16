#!/bin/bash

declare -a C=("c300" "c500" "c1000" "c1500" "c1800")
declare -a D=("d100" "d1000" "d5000")

for c in ${C[@]}
do
    for d in ${D[@]}
    do
	python Homework_1_Code.py ${c}_${d} dt e rep,dbp,n
	python Homework_1_Code.py ${c}_${d} dt vi rep,dbp,n
	python Homework_1_Code.py ${c}_${d} rf
    done
done
