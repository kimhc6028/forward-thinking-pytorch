#!/bin/bash

python backpropagation.py
python forward_thinking.py

python backpropagation.py --deep --epoch 200
python forward_thinking.py --deep --epoch 200

python plotter.py
