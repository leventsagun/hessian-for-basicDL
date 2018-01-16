#!/bin/bash
for number in {1..50}
do
python runexperiment.py
echo $number
done

