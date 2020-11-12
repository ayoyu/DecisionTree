#!/bin/sh
g++ -Wall -o binary_class_exec binary_classification_tree.cpp -L../build -Wl,-rpath=../build -lAlgo
