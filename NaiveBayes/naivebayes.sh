#!/usr/bin/env bash
if [ "$#" -eq 2 ];then

    python naivebayes.py $1 $2
else
    echo "invalid input"
fi