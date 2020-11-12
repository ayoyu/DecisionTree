#!/bin/sh
g++ -Wall -o multi_class_exec multiclass_classification_tree.cpp -L../build -Wl,-rpath=../build -lAlgo
