#!/bin/sh
python two_recurrent_layers.py 55 1 1 2

for i in 1 2 3
do
    for j in 1 2 4 8 16 24 32 40 48
    do
        for k in 1 2
        do
            python two_recurrent_layers.py 55 $k $i $j
        done 
    done
done
