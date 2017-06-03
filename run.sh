#!/bin/bash

file="train"

if [ -f "$file" ]
then
	echo "remove $file"
	rm $file
fi

echo "build train"

g++ -I../../tiny-dnn/ -I/usr/local/include/ -L/usr/local/lib/ -pthread -ltbb -lopencv_core -lopencv_highgui -std=c++11 -O2 -mavx -DCNN_USE_AVX -DCNN_USE_TBB train.cpp -o train

if [ -f "$file" ]
then
	echo "build train exe finished" 
fi
