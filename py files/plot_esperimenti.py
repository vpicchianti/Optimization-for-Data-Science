
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


''' questo script è stato progettato per condurre una serie di esperimenti e 
    plot per i due algoritmi implementati (FW e AFW), come riportati e commentati nel report '''


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




# Esperimento per valutare l'impatto di tau - trust region sulla convergenza
# PLOT FIGURA 4 DEL REPORT
func = lambda x: np.dot(x, np.dot(Q, x))+np.dot(q,x)
nome_file_dmx ='1000/netgen-1000-1-2-a-b-s.dmx'

n, numero_archi, u, b, q, _, from_ , to = leggi_file_dimacs(nome_file_dmx)
a = 100
Q = genera_Q(a, u, q, numero_archi, 0.5)
x_0 = u/2   
f_tol = 1e-9
time_tol = np.inf
epsilon = 1e-4
max_iter = 2000


tau_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

gap_results = []
function_values = []
tempi_iter_results = []
best_function_values = []

for tau in tau_values:
    x, function_value, elapsed_time, dual_gap, k, tempi_per_it, tempo_per_it_MCF, status = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, max_iter, numero_archi, x_0, tau, visualize_res=False)
    gap_results.append(dual_gap)
    best_function_value = function_value[-1]
    function_values.append(function_value)  
    best_function_values.append(best_function_value)
    tempi_iter_results.append(tempi_per_it)


np.save('Q.npy', Q)
k_summary = [len(el)-1 for el in tempi_iter_results]
tempo_tot = [sum(lista) for lista in tempi_iter_results]
d = {'tau': tau_values, 'k tot': k_summary, 'best f value': best_function_values, 'tempo_tot': tempo_tot}
results_df = pd.DataFrame(data = d)


plt.style.use('ggplot')
for i, tau in enumerate(tau_values):
    plt.plot(gap_results[i], label=f'tau = {tau}, fbest = {best_function_values[i]:.4e}')

plt.yscale('log')
plt.xlabel('iter (k)', fontsize=14)
plt.ylabel('gap', fontsize=14)
plt.title('Convergence of Dual Gap for Different Tau Values', fontsize=14)
plt.legend()
plt.show()





# PLOT DELLO STEPSIZE in f(k) FIGURA 2a del REPORT 

# risultati dell'algoritmo quando stepsize_ottimo = False
x_nonopt, function_value_nonopt, elapsed_time_nonopt, dual_gap_nonopt, k_nonopt, tempi_per_it_nonopt, tempo_per_it_MCF_nonopt, status_nonopt = FW(b,n,from_,to,f_tol, time_tol, epsilon, Q, q, u,
                                                                                                 max_iter, numero_archi, x_0, tau, False, True)

plt.style.use('ggplot')
plt.plot(list(range(k+1)), dual_gap, label = 'stepsize ottimo', color = 'blue')
plt.plot(list(range(k_nonopt+1)), dual_gap_nonopt, label='alpha = 2/k+2', color = 'red')

plt.ylabel ('gap',  fontsize = 14)
plt.xlabel('iter(k)',  fontsize = 14)
plt.yscale('log')
plt.title('Andamento del gap per i due diversi stepsize', fontsize = 14)
plt.legend()
plt.show()




# ANALISI DEL TEMPO PER OGNI ITERAZIONE FIGURA 2b del report 
plt.style.use('ggplot')
plt.plot(list(range(k+1)), tempi_per_it, label = 'stepsize ottimo', color = 'blue')
plt.plot(list(range(k_nonopt+1)), tempi_per_it_nonopt, label='alpha = 2/k+2', color = 'red')

plt.ylabel ('tempo per iterazione in secondi',  fontsize = 14)
plt.xlabel('iter (k)',  fontsize = 14)
# plt.yscale('log')
plt.ylim(-0.1, 0.30)
plt.title('Tempi per iter per i due diversi stepsize', fontsize = 14)
plt.legend()
plt.show()




# TABELLA 1: Impatto dei diversi stepsize (ottimo/non) sul numero totale di iterazioni e sul tempo 
folder_path = r'C:\Users\Valeria\Documents\GitHub\Optimization\1000' # path assoluto della cartella 1000
esperimenti = []
file_names = os.listdir(folder_path)

for file_name in file_names:
    if file_name.endswith('.dmx'):
        file_path = os.path.join(folder_path, file_name)
        relative_path = os.path.relpath(file_path, os.getcwd())
        esperimenti.append(relative_path)


data = {
    'nome_file_dmx': [],
    'iter_optimal': [],
    'iter_non_optimal': [],
    'time_optimal': [],
    'time_non_optimal': []
}

a = 1000
p = 0.5
Q = genera_Q(a, u, q, numero_archi, p)

for nome_file_dmx in esperimenti[:14]:
    func = lambda x: np.dot(x, np.dot(Q, x)) + np.dot(q, x)

    n, numero_archi, u, b, q, _, from_, to = leggi_file_dimacs(nome_file_dmx)
    x_0 = u / 2
    f_tol = 1e-9
    time_tol = np.inf
    epsilon = 1e-3
    tau1 = 0.0
    max_iter = 2000

    try:
        x, _, elapsed_time, _, k, _, _, _ = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=True, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    try:
        x_nonopt, _, elapsed_time_nonopt, _, k_nonopt, _, _, _ = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=False, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    updated_nome_file_dmx = nome_file_dmx.replace('1000\\', '')
    data['nome_file_dmx'].append(updated_nome_file_dmx)
    data['iter_optimal'].append(k)
    data['iter_non_optimal'].append(k_nonopt)
    data['time_optimal'].append(elapsed_time[-1])
    data['time_non_optimal'].append(elapsed_time_nonopt[-1])

df = pd.DataFrame(data)




# PLOT FIGURA 1 DEL REPORT : convergenza effettiva con i diversi stepsize (ottimo/non)
a = 100
p = 0.5
Q = genera_Q(a, u, q, numero_archi, p)

for nome_file_dmx in esperimenti[4:9]:
    func = lambda x: np.dot(x, np.dot(Q, x)) + np.dot(q, x)

    n, numero_archi, u, b, q, _, from_, to = leggi_file_dimacs(nome_file_dmx)
    x_0 = u / 2
    f_tol = 1e-9
    time_tol = np.inf
    epsilon = 1e-3
    tau1 = 0.0
    max_iter = 2000

    try:
        x, function_value, elapsed_time, _, k, _, _, _ = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=True, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    try:
        x_nonopt, function_value_nonopt, elapsed_time_nonopt, _, k_nonopt, _, _, _ = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=False, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue
    
    fstar_opt = func(x)
    fstar_nonopt = func(x_nonopt)
    _, _, u, _, _, _, _ , _ = leggi_file_dimacs(nome_file_dmx)
    Q_diag = np.diag(Q)

    convergence_values = []  # valori di convergenza teorica
    convergence_values_nonopt = []

    # simulo i dati teorici (convergenza teorica)
  
    plt.plot(list(range(k)), function_value[:k] - fstar_opt, linestyle='-', color = 'blue')
    plt.yscale("log") 
    plt.xlabel("iter  (k)", fontsize = 14)
    plt.ylabel(r'Valore di convergenza = $|f(x) - f(x*)|$', fontsize = 13)
    plt.title('Convergenza effettiva con stepsize ottimo', fontsize = 14)
    plt.grid(True)

    plt.show()
    
    plt.plot(list(range(k_nonopt)), function_value_nonopt[:k_nonopt] - fstar_nonopt, label="Convergenza Effettiva" , linestyle='-', color = 'red')
    plt.yscale("log") 
    # plt.ylim(1e-1, 1e13)
    plt.xlabel("iter  (k)", fontsize = 14)
    plt.ylabel(r'Valore di convergenza = $|f(x) - f(x*)|$', fontsize = 13)
    
 
    plt.title("Convergenza effettiva con stepsize 2/2+k", fontsize = 14)
    plt.grid(True)

    plt.show()

    
# PLOT DI FIGURA 6 e 7: andamento del gap relativo e del primal gap 
x, function_value, elapsed_time, rel_gap, k, tempi_per_it, tempo_per_it_MCF, status = FW(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, max_iter, numero_archi, x_0)
x_afw,function_value_afw, k_afw, tempo_tot_afw, tempi_per_it_afw, x_old_afw, found_optimal_afw,rel_gap_afw,tempi_per_it_afw, tempo_per_it_MCF_afw,elapsed_time_afw=AFW(nome_file_dmx, epsilon, max_iter, Q, q, numero_archi,func, 1, x_0, visualize_res=False)

tempi_per_it_afw_final = []
accumulo = 0
for elemento in tempi_per_it_afw:
    accumulo += elemento
    tempi_per_it_afw_final.append(accumulo)
    
tempi_per_it_final = []
accumulo = 0
for elemento in tempi_per_it:
    accumulo += elemento
    tempi_per_it_final.append(accumulo)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.style.use('ggplot')
axs[0].plot(tempi_per_it_final, rel_gap, label='FW tradizionale', color='red')
axs[0].plot(tempi_per_it_afw_final[:-1], rel_gap_afw, label='Away Step FW', color='green')
axs[0].set_xlabel('tempo [s]')
axs[0].set_ylabel('Gap Relativo')
axs[0].set_title('Andamento del Gap Relativo')
axs[0].legend()
axs[0].grid(True)
axs[0].set_yscale('log')
optimal_value_fw = function_value[-1]
primal_gap_fw = [abs(val - optimal_value_fw) for val in function_value]

optimal_value_afw = function_value_afw[-1]
primal_gap_afw = [abs(val - optimal_value_afw) for val in function_value_afw]
axs[1].plot(tempi_per_it_final, primal_gap_fw, label='FW tradizionale', color='red')
axs[1].plot(tempi_per_it_afw_final, primal_gap_afw, label='Away Step FW', color='green')
axs[1].set_xlabel('tempo [s]')
axs[1].set_ylabel('Primal Gap')
axs[1].set_title('Andamento del Primal Gap per i Due Algoritmi')
axs[1].legend()
axs[1].grid(True)
axs[1].set_yscale('log')

plt.tight_layout()
plt.show()





# Esperimenti con solver Gurobi 

func = lambda x: np.dot(x, np.dot(Q, x))+np.dot(q,x)
nome_file = '1000/1000/netgen-1000-1-2-a-b-s.dmx'
n, m, u, b, q, _, from_ , to= leggi_file_dimacs(nome_file)
Q=genera_Q(100, u, q, m, 0.6)
np.save('Q_1000-1-2-a-b-s.npy', Q)
#Q=np.load('Q_1000-1-1-a-a-ns.npy')
E = np.zeros((n, m))

for j in range(m):
    i_entrante = to[j]
    i_uscente = from_[j]
    E[i_entrante - 1, j] = -1  # L'arco j è entrante nel nodo i_entrante
    E[i_uscente - 1, j] = 1    # L'arco j è uscente dal nodo i_uscente

model = gp.Model()

# variabili di decisione + vincolo sulla x 
x = []
for i in range(m):
    x.append(model.addVar(lb=0.0, ub=u[i], name=f"x_{i}"))

# funzione obiettivo
model.setObjective(x @ Q @ x + q @ x, sense=gp.GRB.MINIMIZE)

# vincoli 
for i in range(n):
   
    model.addConstr(sum(E[i][j] * x[j] for j in range(m)) == b[i], name=f"flow_balance_{i}")
    
model.optimize()
if model.status == gp.GRB.OPTIMAL:
    print("Soluzione ottima trovata:")
   # for i in range(m):
  #      print(f"x[{i}] = {x[i].x}")
    print(f"Valore della funzione obiettivo: {model.objVal}")
else:
    print("Il problema non ha una soluzione ottimale.")

model.dispose()



