#!/bin/bash

cd ../..

python test/parent.py

if [ $? -eq 0 ]
then
	exit 0
else
	exit 1
fi