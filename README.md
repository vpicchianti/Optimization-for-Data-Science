# Optimization for Data Science 
Repository dedicated to the Quadratic Min Cost Flow Optimization project, implemented using the Frank-Wolfe algorithm. 

## Optimization Problem

Consider the optimization problem:

\[ \min \{x^T Q x + q^T x : Ex = b, 0 \leq x \leq u\} \quad  \]

where:
- \( x, q \in \mathbb{R}^m \)
- \( Q \in \mathbb{R}^{m \times m} \) is a positive semidefinite diagonal matrix
- \( E \in \mathbb{R}^{n \times m} \) is the node-arc incidence matrix of a connected graph with \( n \) nodes and \( m \) edges
- \( b \in \mathbb{R}^n \) is the node balance vector of the graph
- \( u \in \mathbb{R}^m \) is a vector of positive values representing the upper capacity of each edge in the graph.



We aim at implementing an **optimization algorithm** A of the class of **Frank-Wolfe (FW) methods**, based on the conditional gradient (CG), designed to solve constrained and convex optimization problems.


## In this repository
This folder contains the following files:

- _funzioni.py_: This file includes the definition of all the functions used to implement the algorithms described in the report, as well as the corresponding solving algorithms (both FW and AFW).

- _prove_algoritmi.py:_ This file contains the code to test the functionality of the two algorithms, with arbitrary parameter values. At each iteration, the current iteration number, the function value, and the gap value are printed. Upon completion, the status (found optimum/stopped for max_iter/..), the final stepsize value, gap value, function value, total number of iterations, and running time are also printed.

- _plot_esperimenti.py_: This file contains scripts used for the plots reported in the .pdf and scripts for implementing the code with the Gurobi solver.

- Picchianti_Scotti_report.pdf: This file contains the descriptive report of the project, divided into sections:
      _Introduction
      Frank-Wolfe Algorithm
      Algorithm Implementation
      Description of Data Used in Experiments
      Experiments
      Conclusions
      Scotti_Picchianti_Optimization32.ipynb: This notebook is the actual one used for the project._

- folder 1000 and 2000: Folders containing the instances of the problem, with 1000 and 2000 nodes (excluding Q). It was generated on https://commalab.di.unipi.it/datasets/mcf/, where other instances are also available.

- Scotti_Picchianti_PROG32: This file is the notebook file we originally worked on
