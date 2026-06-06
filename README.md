# Assets Optimizer Module

This module is used to process and automatically compress fbx assets using AI and blender

## Pipeline

1. **AI prediction.** The fbx model is loaded at the input, its data is read and transferred to the neural network. At the output, the neural network produces three values: the compression ratio, the quality criterion and classification of model
2. **Calculation of the best parameters.** The algorithm is given coefficients from a neural network. Through an iterative approach, the algorithm calculates the best values for compressing and protecting the topology
3. **Optimization.** The algorithm takes the best coefficients and outputs of the neural network and compresses the model into the final version

## Requirements

1. Python 3.12

## Getting Started

1. Clone the repo
2. Create `config/config.json` from `config/config.example.json`
3. ```bash
   python3 -m venv .venv
   . ./venv/bin/activate
   pip install -r ./pyproject.toml
   ```
4. ```bash
   python3 main.py
   ```