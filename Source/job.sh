#!/bin/sh

#Files

for file in 11 99
do
   #layer
   for i in 1 2 
   do
      #units
      for j in 2 4 8
      do
         #folds
         for k in 1 2
         do
            python two_recurrent_layers.py $file $k $i $j
         done 
      done
   done
done


