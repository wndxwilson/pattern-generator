# Reaction-Diffusion Simulation README

## Overview

This Python script simulates a reaction-diffusion system, a fundamental concept in physical chemistry and pattern formation. It uses numpy for numerical computations, matplotlib for visualization, scipy for convolution operations, and click for command-line interface functionality. The simulation generates patterns based on user-defined parameters and saves the results as images.

## Features

- **Reaction-Diffusion Simulation**: Models the interaction between two chemical substances.
- **Image Generation**: Creates visual representations of the reaction-diffusion process.
- **Interactive Parameters**: Allows customization of simulation parameters.
- **Batch Image Generation**: Generates and saves images for a range of simulation steps.

## Installation

Setup virtual environment 
```
python -m venv venv
```

Activate virtual environment
```
source venv/bin/activate
```

Before running the script, ensure you have Python installed on your system along with the following packages:
- numpy
- matplotlib
- scipy
- click

You can install these packages using pip:
```bash
pip install numpy matplotlib scipy click
```

## Usage

The script can be used in two main ways:

1. **As a Command-Line Tool**: 

   Run the script from the command line with optional arguments to specify the range and interval of simulation steps.

   ```bash
   python pattern.py --min_step [MIN_STEPS] --max_step [MAX_STEPS] --interval [INTERVAL]
   ```

## Functions

- `reaction_diffusion(a, b, da, db, feed, k, steps)`: Performs the reaction-diffusion process.
- `interactive_simulation_steps(steps, da, db, feed, k, colormap)`: Runs the interactive simulation for a given number of steps.
- `steps_generator(min_step, max_step, interval)`: Generates images for a range of steps at specified intervals.

## Parameters

- `a`, `b`: Numpy arrays representing the chemical substances.
- `da`, `db`: Diffusion rates of substances `a` and `b`.
- `feed`: Feed rate.
- `k`: Kill rate.
- `steps`: Number of simulation steps.
- `colormap`: Colormap for the resulting image (in `interactive_simulation_steps`).
