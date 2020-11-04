#!/bin/sh
g++ -Wall -o exec tree_examples.cpp -L../build -Wl,-rpath=../build -lAlgo