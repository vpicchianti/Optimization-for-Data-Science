from funzioni import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import time
from datetime import datetime
from time import process_time
from scipy.optimize import *
from scipy.optimize import minimize_scalar
from pyMCFSimplex import *
import gurobipy as gp
import cvxpy as cp


# Frank Wolfe con trust region ex post e FW con away step 
func = lambda x: np.dot(x, np.dot(Q, x))+np.dot(q,x)
nome_file_dmx ='1000/netgen-1000-1-2-a-b-s.dmx'
n, numero_archi, u, b, q, _, from_ , to = leggi_file_dimacs(nome_file_dmx)
a = 100
p = 0.3
Q = genera_Q(a, u, q, numero_archi, p)
x_0 = u/2   
f_tol = 1e-9
time_tol = np.inf
epsilon = 1e-4
max_iter = 1000
tau = 0.0
x, function_value, elapsed_time, dual_gap, k, tempi_per_it, tempo_per_it_MCF, status = FW(b,n,from_,to,f_tol, time_tol, epsilon, Q, q, u,
                                                                                                 max_iter, numero_archi, x_0, tau)

f_values, k, tempo_tot, tempi_per_it, x, dual_gap, tempo_per_it_MCF, elapsed_time = AFW(epsilon, max_iter, Q, q, numero_archi,func, x,  n, u, b, to, from_, visualize_res=True)


