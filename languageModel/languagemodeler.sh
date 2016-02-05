#!/usr/bin/env bash
if [ "$#" -eq 6 ];then

    python perplexityCalculator.py $1 $2 $3 $4 $5 $6
else
    echo "invalid input"
fi