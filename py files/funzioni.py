
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
import cvxpy as cp

'''  Generazione delle istanze del problema
 La prima funzione *genera_Q(alpha, u, q, dimensione, p)* serve per generare i costi quadratici 
 che si aggiungono ai costi lineari dell'istanza di MCF;
 la stategia di generazione prevede di generare randomicamente (uniformemente) costi Qij "piccoli" o "grandi" rispetto ai costi lineari, 
 come specificato dal parametro *alpha*. 
 
 Permette inoltre di definire - tramite il parametro *p* - la proporzione di archi che hanno costo quadratico Qij nullo, 
 abilitando la definizione e l'analisi anche di matrici semi-definite positive (0 <=p <= 1). 
 
 La seconda funzione *leggi_file_dimacs(nome_file)* serve per estrarre le seguenti quantità: 
  - u, b, q
  - numero nodi
  - numero archi 
 
 dal relativo file *.dmx*, generato [qui](https://commalab.di.unipi.it/datasets/mcf/). 
'''

def genera_Q(alpha, u, q, dimensione, p):
    ''' 
    Genera una matrice quadratica Q con elementi casuali

    Parametri:
    - alpha (float): Fattore di scala per la generazione casuale degli elementi di Q
    - u (list): Lista dei valori ui, max capacity 
    - q (list): Lista dei costi lineari
    - dimensione (int): Dimensione della matrice Q
    - p (float): percentuale di elementi della diagonale principale di Q da impostare a zero 

    Ritorno:
    - Q (numpy.ndarray): Matrice quadratica generata, con elementi casuali e alcuni elementi posti a zero in base a p.
    '''

    Q_diag = []

    for i in range(dimensione):
        Q_i = abs(random.uniform((-q[i] / u[i] * alpha), (q[i] / u[i] * alpha))) # generazione randomica 
        Q_diag.append(Q_i)

    Q = np.zeros((dimensione, dimensione))
    np.fill_diagonal(Q, Q_diag) # creazione matrice 

    num_entrate_zero = int(p * dimensione) 
    indici_zeri = np.random.choice(dimensione, num_entrate_zero, replace=False) # trova p*dimensione indici dei costi quadratici da mettere a zero 

    for idx in indici_zeri:
        Q[idx, idx] = 0

    return Q 


def leggi_file_dimacs(nome_file):

    '''
    Legge un file in formato DIMACS e restituisce informazioni sull'istanza

    Parametri:
    - nome_file (str): Il percorso del file DIMACS da leggere

    Ritorno:
    - numero_nodi (int): Il numero totale di nodi 
    - numero_archi (int): Il numero totale di archi 
    - u (numpy.ndarray): Un array numpy contenente le capacità massime degli archi
    - b (list): Una lista contenente i valori di supply per ciascun nodo
    - q (numpy.ndarray): Un array numpy contenente i costi associati agli archi
    - edges (list): Una lista di tuple rappresentanti gli archi del grafo
    - from_ (list): Una lista contenente i nodi di partenza degli archi
    - to_ (list): Una lista contenente i nodi di destinazione degli archi
    '''
        
    numero_nodi = 0
    numero_archi = 0
    u = []
    b = []
    q = []
    from_= []
    to_= []
    edges = []

    with open(nome_file, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) > 0:
                if parts[0] == 'p':  # parts[0] è il primo carattere di ogni riga - nel formato standard DIMACS può essere c, p, n, a
                    # legge il numero di nodi e archi dal problema
                    numero_nodi = int(parts[2])
                    numero_archi = int(parts[3])
                    # inizializza il vettore di supply con zeri
                    b = [0] * numero_nodi
                elif parts[0] == 'n':
                    # legge i valori di supply per i nodi
                    nodo_id = int(parts[1])
                    supply = int(parts[2])
                    # assegna il valore di supply al nodo corrispondente
                    b[nodo_id - 1] = supply
                elif parts[0] == 'a':
                    # leggi l'arco e il suo cotso
                    from_node = int(parts[1])
                    to_node = int(parts[2])
                    max_capacity = int(parts[4])
                    costo = int(parts[5]) 
                    from_.append(from_node)
                    to_.append(to_node)
                    u.append(max_capacity)
                    q.append(costo)
                    edges.append((from_node , to_node ))

    return numero_nodi, numero_archi, np.array(u), b, np.array(q), edges,from_, to_



def showModuleFunctionality(mcf): 
    ''' '''
    vettore_soluzione = {}  
    nmx = mcf.MCFnmax()
    mmx = mcf.MCFmmax()
    pn = mcf.MCFnmax()
    pm = mcf.MCFmmax()

    pU = []
    caps = new_darray(mmx)
    mcf.MCFUCaps(caps)
    for i in range(0, mmx):
        pU.append(darray_get(caps, i))
    
    pC = []
    costs = new_darray(mmx)
    mcf.MCFCosts(costs)
    for i in range(0, mmx):
        pC.append(darray_get(costs, i))

    pDfct = []
    supply = new_darray(nmx)
    mcf.MCFDfcts(supply)
    for i in range(0, nmx):
        pDfct.append(darray_get(supply, i))

    pSn = []
    pEn = []
    startNodes = new_uiarray(mmx)
    endNodes = new_uiarray(mmx)
    mcf.MCFArcs(startNodes, endNodes)
    for i in range(0, mmx):
        pSn.append(uiarray_get(startNodes, i) + 1)
        pEn.append(uiarray_get(endNodes, i) + 1)

    #print("arc flow")
    length = mcf.MCFm()
    flow = new_darray(length)
    length = mcf.MCFn()
    nms = new_uiarray(length)
    mcf.MCFGetX(flow, nms)

   

    for i in range(0, length):
        if uiarray_get(nms, i)== 4294967295:
            break
        else:
       # print("flow", darray_get(flow, i), "arc", uiarray_get(nms, i))
            vettore_soluzione[uiarray_get(nms, i)] = darray_get(flow, i)

    return vettore_soluzione  # restituisce il vettore_soluzione alla fine della funzione



def line_search(x, d, gamma_max,func):
    '''
    Esegue una line search per trovare la lunghezza del passo ottimale lungo la direzione 

    Parametri:
    - x (numpy.ndarray): Punto di partenz
    - d (numpy.ndarray): Direzione di ricerca
    - gamma_max (float): Limite superiore per la lunghezza del passo
    - func (callable): Funzione obiettivo da minimizzare

    Ritorno:
    - ls (numpy.ndarray): Punto ottenuto dopo l'esecuzione della ricerca lineare
    - gamma (float): Lunghezza del passo ottimale
    '''
        
    def fun(gamma):
        ls = x + gamma*d
        return func(ls)

    res = minimize_scalar(fun, bounds=(0, gamma_max), method='bounded')

    gamma = res.x
    ls = x + gamma*d        
    return ls, gamma



class ProblemUnfeasibleError(Exception):
    pass


# away vertex
def away_step(grad, S):
    '''
    Esegue uno "step away" nella direzione del gradiente massimo rispetto a S

    Parametri:
    - grad (numpy.ndarray): Vettore gradiente
    - S (dict): Dizionario rappresentante active set

    Ritorno:
    - vertex (numpy.ndarray): Vertice di S corrispondente al gradiente massimo
    - alpha (float): Coefficiente associato al gradiente massimo
    '''

    costs = {}
    
    for k,v in S.items():
        cost = np.dot(k,grad)
        costs[cost] = [k,v]
    vertex, alpha = costs[max(costs.keys())]  
    return vertex, alpha




def update_S(S,gamma, Away, vertex):
    
    '''
    Aggiorna active set

    Parametri:
    - S (dict): dizionario rappresentante S
    - gamma (float): stepsize 
    - Away (bool): flag che indica se lo step è di tipo "away"
    - vertex (numpy.ndarray): vertice del poliedro S coinvolto nell'aggiornamento

    Ritorno:
    - S_updated (dict): Dizionario aggiornato rappresentante il poliedro S

    '''

    S = S.copy()
    vertex = tuple(vertex)
    
    if not Away:
        if vertex not in S.keys():
            S[vertex] = gamma
        else:
            S[vertex] *= (1-gamma)
            S[vertex] += gamma
            
        for k in S.keys():
            if k != vertex:
                S[k] *= (1-gamma)
    else:
        for k in S.keys():
            if k != vertex:
                S[k] *= (1+gamma)
            else:
                S[k] *= (1+gamma)
                S[k] -= gamma
    return {k:v for k,v in S.items() if np.round(v,3) > 0}




def FW(b,n,from_,to,f_tol, time_tol, 
              epsilon, Q, q, u, max_iter, numero_archi, x, tau,  
              step_size_ottimo=True, visualize_res=True):
    
    '''
    EImplementa Frank-Wolfe con trust region ex post

    Parametri:
    - b (list): Lista dei valori di supply per i nodi
    - n (int): Numero totale di nodi nel problema
    - from_ (list): Lista dei nodi di partenza degli archi
    - to (list): Lista dei nodi di destinazione degli archi
    - f_tol (float): Tolleranza sulla variazione della funzione obiettivo per la terminazione.
    - time_tol (float): Tolleranza sul tempo di esecuzione totale per la terminazione.
    - epsilon (float): Tolleranza per la definizione di ottimalità (gap duale).
    - Q (numpy.ndarray): Matrice quadratica associata al problema.
    - q (numpy.ndarray): Vettore dei costi associato al problema.
    - u (numpy.ndarray): Vettore delle capacità massime degli archi.
    - max_iter (int): Numero massimo di iterazioni consentite.
    - numero_archi (int): Numero totale di archi nel problema.
    - x (numpy.ndarray): Punto di partenza.
    - tau (float): Parametro di controllo della regione di fiducia "ex post".
    - step_size_ottimo (bool): Flag che indica se utilizzare una dimensione passo ottimale.
    - visualize_res (bool): Flag che indica se stampare i risultati durante l'esecuzione.

    Ritorno:
    - x (numpy.ndarray): Soluzione finale 
    - function_value (list): Lista dei valori della funzione obiettivo ad ogni iterazione
    - elapsed_time (list): Lista dei tempi trascorsi ad ogni iterazione
    - dual_gap (list): Lista dei gap duali ad ogni iterazione
    - k (int): Numero totale di iterazioni effettuate
    - tempi_per_it (list): Lista dei tempi trascorsi ad ogni iterazione
    - tempo_per_it_MCF (list): Lista dei tempi trascorsi nella risoluzione del sottoproblema MCF ad ogni iterazione.
    - status (str): Stato finale dell'algoritmo

    '''

    ## inizializzazione delle variabili 
    k = 0
    alpha = 1
    dual_gap = []
    function_value = [func(x)]
    elapsed_time = [0]  
    tempo_per_it_MCF = [0]     
    tempi_per_it = [0] 
    f_improv = np.inf
    status = 'processing'
    bestlb = float('-inf')
    gap = float('-inf')
  
  
    while abs(f_improv) > f_tol and elapsed_time[-1] < time_tol:

        start = time.perf_counter() #contatore tempo 

        ## risoluzione del sottoproblema lineare usando solver MCF 
        
        gradient = (2 * np.dot(Q, x)) + q
        start_MCF = time.perf_counter()
        mcf = MCFSimplex()
        nmx     = n
        mmx     = numero_archi
        pn      = n
        pm      = numero_archi
        pU      = u.tolist()
        pC      = gradient.tolist()
        pDfct   = b
        pSn     = to
        pEn     = from_

        mcf.LoadNet(nmx, mmx, pn, pm, CreateDoubleArrayFromList(pU), CreateDoubleArrayFromList(pC),
                    CreateDoubleArrayFromList(pDfct), CreateUIntArrayFromList(pSn),
                    CreateUIntArrayFromList(pEn))
        
        mcf.SolveMCF()

        end_MCF = time.perf_counter()  
        tempo_per_it_MCF.append(end_MCF - start_MCF)  
        ## fine della risoluzione del sottoproblema lineare

        if mcf.MCFGetStatus() != 0:
            raise ProblemUnfeasibleError("The problem is unfeasible!")

        # recupero x ottima lineare (che chiamerò v)
        vettore_soluzione = showModuleFunctionality(mcf)
        sol_x = np.zeros(numero_archi)
        for key, value in vettore_soluzione.items():
            sol_x[key] = value
        v = sol_x.copy()
        

        # TRUST REGION STABILIZATION EX POST
        u = np.array(u)
        if tau > 0 and k > 1: 
            # calcolo lower bound e upper bound della trust region 
            ub_trust = x + tau * u
            lb_trust = x - tau * u 
            v = np.maximum(lb_trust, np.minimum(v, ub_trust))

        d = v - x  # direzione 

        lb = func(x) + np.dot(gradient, d)   # lower bound 
        if lb > bestlb:
            bestlb = lb
        gap = (func(x) - bestlb) / max(abs(func(x)), 1)  # relative gap
        dual_gap.append(gap)


        if visualize_res and k > 0:
            print(f'iter: {k}, f value = {function_value[-1]}, fbest = {func(x):.8e}, gap = {dual_gap[-1]:.4e}')

        # vari check terminazione 
        if abs(gap) <= epsilon:  
            status = 'found optimal'
            break
        
        if k == max_iter:
            status = 'stopped (max_iter)'
            print(status)
            break
        
        # aggiornamento della posizione 
        if step_size_ottimo:
            x, alpha = line_search(x, d, 1, func)
        else:
            alpha = 2 / (2 + k)
            x = x + alpha * d

        if not np.all((0 <= x) & (x <= u)):
            status = 'unfeasible because of trust region'
            print(status)
            break

        end = time.perf_counter()
        tempo_it = end - start
        tempi_per_it.append(tempo_it)
        elapsed_time.append(elapsed_time[k] + tempo_it) 
        
        f_improv = function_value[-1] - func(x) # inutile non guardare 
        function_value.append(func(x))
        
        k += 1 # aggiorno k 

        if abs(f_improv) < f_tol:  # aggiornamento status
            status = 'stopped (f_tol)'
        if elapsed_time[-1] > time_tol:
            status = 'stopped (time_tol)'

    print('Status:', status)
    print('Final gap duale: {:>10.6f}  Final step size: {:>10.8f}  Total number iterations: {:>4}'.format(dual_gap[-1], alpha, k))
    print('fbest:', func(x))
    print('Total running time: ', sum(tempi_per_it))
    
    return x, function_value, elapsed_time, dual_gap, k, tempi_per_it, tempo_per_it_MCF, status





def FW_exante(b,n,from_,to,f_tol, time_tol, epsilon, Q, q, u, tau, max_iter, numero_archi, x, step_size_ottimo=True, visualize_res=True):
    
    '''
    Frank-Wolfe con trust region ex ante

    Parametri:
    - b (list): Lista dei valori di supply per i nodi.
    - n (int): Numero totale di nodi nel problema.
    - from_ (list): Lista dei nodi di partenza degli archi.
    - to (list): Lista dei nodi di destinazione degli archi.
    - f_tol (float): Tolleranza sulla variazione della funzione obiettivo per la terminazione.
    - time_tol (float): Tolleranza sul tempo di esecuzione totale per la terminazione.
    - epsilon (float): Tolleranza per la definizione di ottimalità (gap duale).
    - Q (numpy.ndarray): Matrice quadratica associata al problema.
    - q (numpy.ndarray): Vettore dei costi associato al problema.
    - u (numpy.ndarray): Vettore delle capacità massime degli archi.
    - tau (float): Parametro di controllo della regione di fiducia "ex ante".
    - max_iter (int): Numero massimo di iterazioni consentite.
    - numero_archi (int): Numero totale di archi nel problema.
    - x (numpy.ndarray): Punto di partenza.
    - step_size_ottimo (bool): Flag che indica se utilizzare una dimensione passo ottimale.
    - visualize_res (bool): Flag che indica se stampare i risultati durante l'esecuzione.

    Ritorno:
    - x (numpy.ndarray): soluzione finale 
    - function_value (list): Lista dei valori della funzione obiettivo ad ogni iterazione.
    - elapsed_time (list): Lista dei tempi trascorsi ad ogni iterazione.
    - dual_gap (list): Lista dei gap duali ad ogni iterazione.
    - k (int): Numero totale di iterazioni effettuate.
    - tempi_per_it (list): Lista dei tempi trascorsi ad ogni iterazione.
    - tempo_per_it_MCF (list): Lista dei tempi trascorsi nella risoluzione del sottoproblema MCF ad ogni iterazione.
    - status (str): Stato finale dell'algoritmo.
    '''


    k = 0
    dual_gap = []
    function_value = [func(x)]
    elapsed_time = [0]  
    tempo_per_it_MCF = [0]     
    tempi_per_it = [0] 
    f_improv = np.inf
    status = 'processing'
    bestlb = float('-inf')
    lb_trust = np.zeros(numero_archi).astype('float64')
    ub_trust = np.array(u).astype('float64')
    alpha = 1
    pDfct   = b

    while abs(f_improv) > f_tol and elapsed_time[-1] < time_tol:

        if tau > 0 and k >1: 
            lb_trust = np.maximum(0, sol_x - tau * u).astype('float64')
            ub_trust = np.minimum(u,sol_x + tau * u).astype('float64')
        start = time.perf_counter()

        gradient = (2 * np.dot(Q, x)) + q
        start_MCF = time.perf_counter()
        mcf = MCFSimplex()
        nmx     = n
        mmx     = numero_archi
        pn      = n
        pm      = numero_archi
        pU      = ub_trust
        pC      = gradient 
        #pDfct   = b
        pSn     = to
        pEn     = from_
        pU      -= lb_trust
        pU      = pU.tolist()

        for index, el in enumerate(pEn):
            pDfct[el-1] = pDfct[el-1] - lb_trust[index]
        for index, el in enumerate(pSn):
             pDfct[el-1] = pDfct[el-1] + lb_trust[index]

        mcf.LoadNet(nmx, mmx, pn, pm, CreateDoubleArrayFromList(pU), CreateDoubleArrayFromList(pC),
                    CreateDoubleArrayFromList(pDfct), CreateUIntArrayFromList(pSn),
                    CreateUIntArrayFromList(pEn))

        mcf.SolveMCF()

        end_MCF = time.perf_counter()  
        tempo_per_it_MCF.append(end_MCF - start_MCF)  

        if mcf.MCFGetStatus() != 0:
            raise ProblemUnfeasibleError("The problem is unfeasible!")

        vettore_soluzione = showModuleFunctionality(mcf)
        sol_x = [0] * numero_archi
  
        for key in vettore_soluzione:
            sol_x[key] = vettore_soluzione[key]
        sol_x = sol_x - lb_trust
        sol_x -= lb_trust
        v = np.array(sol_x)
        d = v - x

        if visualize_res and k > 0:
            print(f'iter: {k}, f value = {function_value[-1]}, fbest = {func(x):.8e}, gap = {dual_gap[-1]:.4e}')


        lb = func(x)+ np.dot(gradient, v - x)  
        if lb > bestlb:
            bestlb = lb
        gap = (func(x) - bestlb) / max(abs(func(x)), 1)  # relative gap

        dual_gap.append(gap)

        if abs(gap) <= epsilon:  
            status = 'found optimal'
            break
        
        if k == max_iter:
            status = 'stopped (max_iter)'
            print(status)
            break
        
        if step_size_ottimo:
            x, alpha = line_search(x, d, 1, func)
        else:
            alpha = 2 / (2 + k)
            x = x + alpha * d

        end = time.perf_counter()
        tempo_it = end - start
        tempi_per_it.append(tempo_it)
        elapsed_time.append(elapsed_time[k] + tempo_it) 
        
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        
        k += 1

        if abs(f_improv) < f_tol:
            status = 'stopped (f_tol)'
            print(status)
            break
        if elapsed_time[-1] > time_tol:
            status = 'stopped (time_tol)'
            print(status)
            break

    print('Status:', status)
    print('Final gap duale: {:>10.6f}  Final step size: {:>10.8f}  Total number iterations: {:>4}'.format(dual_gap[-1], alpha, k))
    print('fbest:', func(x))
    print('Total running time: ', sum(tempi_per_it))

    return x, function_value, elapsed_time, dual_gap, k, tempi_per_it, tempo_per_it_MCF, status




def AFW(epsilon, max_iter, Q, q, numero_archi,func, x,  n, u, b, to, from_, visualize_res=True):
    
    '''
    FW con away step 

    Parametri:
    - epsilon (float): tolleranza su gap
    - max_iter (int): Numero massimo di iterazioni consentite
    - Q (numpy.ndarray): Matrice quadratica associata al problema
    - q (numpy.ndarray): Vettore dei costi associato al problema
    - numero_archi (int): Numero totale di archi nel problema
    - func (callable): Funzione obiettivo del problema
    - x (numpy.ndarray): Punto di partenza
    - n (int): Numero totale di nodi nel problema
    - u (numpy.ndarray): Vettore delle capacità massime degli archi
    - b (list): Lista dei valori di supply per i nodi
    - to (list): Lista dei nodi di destinazione degli archi
    - from_ (list): Lista dei nodi di partenza degli archi
    - visualize_res (bool): Flag che indica se stampare i risultati durante l'esecuzione

    Ritorno:
    - f_values (list): Lista dei valori della funzione obiettivo ad ogni iterazione.
    - k (int): Numero totale di iterazioni effettuate.
    - tempo_tot (float): Tempo totale di esecuzione.
    - tempi_per_it (list): Lista dei tempi trascorsi ad ogni iterazione.
    - x (numpy.ndarray): Soluzione finale 
    - dual_gap (list): Lista dei gap duali ad ogni iterazione.
    - tempo_per_it_MCF (list): Lista dei tempi trascorsi nella risoluzione del sottoproblema MCF ad ogni iterazione.
    - elapsed_time (list): Lista dei tempi totali trascorsi ad ogni iterazione.
    '''
        

    k = 0    
    bestlb = -np.inf
    gamma = 1
    tempo_tot = 0
    status = 'processing'
    f_values = []
    tempi_per_it = [0]
    gap = []
    dual_gap = []
    elapsed_time = [0]  
    tempo_per_it_MCF = [0]  
    f_improv = np.inf
    tempi_per_it = [0]  
  
    S = {tuple(np.array(x)): 1} # active set inizializzazione
   

    while abs(f_improv) > f_tol and elapsed_time[-1] < time_tol:

        start = time.perf_counter()

        f_values.append(func(x))

        # STEP 1 : calcolo gradiente
        gradient = (2 * np.dot(Q, x)) + q

        mcf = MCFSimplex()
        start_MCF = time.perf_counter()  
        nmx     = n
        mmx     = numero_archi
        pn      = n
        pm      = numero_archi
        pU      = u.tolist()
        pC      = gradient.tolist()
        pDfct   = b
        pSn     = to
        pEn     = from_
     
        mcf.LoadNet(nmx, mmx, pn, pm, CreateDoubleArrayFromList(pU), CreateDoubleArrayFromList(pC),
            CreateDoubleArrayFromList(pDfct), CreateUIntArrayFromList(pSn),
            CreateUIntArrayFromList(pEn))
        
        mcf.SolveMCF()

        end_MCF = time.perf_counter() 
        tempo_per_it_MCF.append(end_MCF - start_MCF)  

        if mcf.MCFGetStatus() != 0:
            raise ProblemUnfeasibleError("The problem is unfeasible!")

        vettore_soluzione = showModuleFunctionality(mcf)
        sol_x = [0] * numero_archi

        for key in vettore_soluzione:
            sol_x[key] = vettore_soluzione[key]

        # calcolo delle x_bar e determinazione della direzione di ricerca
        v = sol_x.copy()
        d_FW = v - x
        
        # calcolo away vertex e direzione d_A 
        a, alpha_a = away_step(gradient, S)
        d_A = x - a
       
        # check se FW gap è maggiore dell'away gap --> per capire quale passo usare 
        if np.dot(-gradient, d_FW) >= np.dot(-gradient, d_A):
            # scegliamo FW direction
            d = d_FW
            vertex = v
            gamma_max = 1
            Away = False
        else:
            # scegliamo Away direction
            d = d_A
            vertex = a
            gamma_max = alpha_a/(1-alpha_a)

        if visualize_res and k > 0:
            #visualize_bis(k, function_value[-1], alpha, primal_gap[-1], k_granularity=1)
            print(f'iter: {k} iter, fval = {f_values[-1]}, fbest = {func(x):.8e}, gap = {dual_gap[-1]:.4e}')
        
        lb = func(x)+ np.dot(np.array(gradient), np.array(d))  # Compute the lower bound
       
        if lb > bestlb:
            bestlb = lb
        
        gap = (func(x) - bestlb)/ max(abs(func(x)), 1)  # relative gap
        dual_gap.append(gap)
        
        if gap <= epsilon:  
            status = 'found optimal'
            print(status)
            break
        if k == max_iter:
            status = 'stopped (max_iter)'
            print(status)
            break

        # ricerca di x_new e gamma usando line search 
        x, gamma = line_search(x,d, gamma_max, func)

        # aggiornamento active set
        S = update_S(S,gamma, Away, vertex)

   
        end = time.perf_counter()
        tempo_it = end - start
        tempi_per_it.append(tempo_it)
        elapsed_time.append(elapsed_time[k] + tempo_it)  
        f_improv = f_values[-1] - func(x)
        f_values.append(func(x))
        
        k+=1

        if abs(f_improv) < f_tol:
            status = 'stopped (f_tol)'
            print(status)
            break

        if elapsed_time[-1] > time_tol:
            status = 'stopped (time_tol)'
            print(status)
            break
        
         

    print('Status:', status)
    print()
    print('Final gap duale: {:>10.6f}  Final step size: {:>10.8f}  Total number iterations: {:>4}'.format(dual_gap[-1], gamma, k))
    if status == 'found optimal':
        ('Optimal solution:', func(x))
    print('Total running time: ', sum(tempi_per_it))

    return f_values, k, tempo_tot, tempi_per_it, x, dual_gap, tempo_per_it_MCF, elapsed_time



func = lambda x: np.dot(x, np.dot(Q, x))+np.dot(q,x)
nome_file_dmx ='1000/netgen-1000-1-2-a-b-s.dmx'
n, numero_archi, u, b, q, _, from_ , to = leggi_file_dimacs(nome_file_dmx)
a = 100
Q = genera_Q(a, u, q, numero_archi, 0.3)
x_0 = u/2   
f_tol = 1e-9
time_tol = np.inf
epsilon = 1e-4
max_iter = 1000
tau = 0.0
x, function_value, elapsed_time, dual_gap, k, tempi_per_it, tempo_per_it_MCF, status = FW(b,n,from_,to,f_tol, time_tol, epsilon, Q, q, u,
                                                                                                 max_iter, numero_archi, x_0, tau)
f_values, k, tempo_tot, tempi_per_it, x, dual_gap, tempo_per_it_MCF, elapsed_time = AFW(epsilon, max_iter, Q, q, numero_archi,func, x,  n, u, b, to, from_, visualize_res=True)


func = lambda x: np.dot(x, np.dot(Q, x))+np.dot(q,x)
#nome_file_dmx ='1000/netgen-1000-1-2-a-b-s.dmx'
nome_file_dmx = '1000/netgen-1000-1-3-a-b-s.dmx'
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


# %%
np.save('Q.npy', Q)
k_summary = [len(el)-1 for el in tempi_iter_results]
tempo_tot = [sum(lista) for lista in tempi_iter_results]
d = {'tau': tau_values, 'k tot': k_summary, 'best f value': best_function_values, 'tempo_tot': tempo_tot}
results_df = pd.DataFrame(data = d)
results_df

# %%
plt.style.use('ggplot')
for i, tau in enumerate(tau_values):
    plt.plot(gap_results[i], label=f'tau = {tau}, fbest = {best_function_values[i]:.4e}')

plt.yscale('log')

plt.xlabel('iter (k)', fontsize=14)
plt.ylabel('gap', fontsize=14)
plt.title('Convergence of Dual Gap for Different Tau Values', fontsize=14)
plt.legend()

plt.show()

# %%
# confronto con altro solver (cvxpy)

nome_file_dmx = '1000/netgen-1000-1-3-a-b-s.dmx'
n, numero_archi, u, b, q, _, from_ , to = leggi_file_dimacs(nome_file_dmx)
Q = np.load('Q.npy')
E = np.zeros((n, numero_archi))

for j in range(numero_archi):
    i_entrante = to[j]
    i_uscente = from_[j]
    E[i_entrante - 1, j] = -1  
    E[i_uscente - 1, j] = 1    

n = len(q)
x = cp.Variable(n)

objective = cp.Minimize(cp.quad_form(x, Q) + q @ x) # funzione obiettivo 

constraints = [E @ x == b, 0 <= x, x <= u]  #definizione vincoli 

problem = cp.Problem(objective, constraints)

problem.solve()

if problem.status == cp.OPTIMAL:
    print('soluzione ottimale trovata!')
    print('valore ottimo della funzione obiettivo =', problem.value)
    # print('valori delle variabili x:')
    # print(x.value)
else:
    print('nessuna soluzione ottimale trovata.')



# %%
# PLOT DELLO STEPSIZE in f(time)
plt.style.use('ggplot')
plt.plot(elapsed_time, dual_gap, label = 'stepsize ottimo', color = 'blue')
plt.plot(elapsed_time_nonopt, dual_gap_nonopt, label='alpha = 2/k+2', color = 'red')

plt.ylabel ('gap',  fontsize = 14)
plt.xlabel('time',  fontsize = 14)
plt.yscale('log')
plt.title('Andamento del gap per i due diversi stepsize', fontsize = 14)
plt.legend()
plt.show()

# %%
# PLOT DELLO STEPSIZE in f(k)
plt.style.use('ggplot')
plt.plot(list(range(k+1)), dual_gap, label = 'stepsize ottimo', color = 'blue')
plt.plot(list(range(k_nonopt+1)), dual_gap_nonopt, label='alpha = 2/k+2', color = 'red')

plt.ylabel ('gap',  fontsize = 14)
plt.xlabel('iter(k)',  fontsize = 14)
plt.yscale('log')
plt.title('Andamento del gap per i due diversi stepsize', fontsize = 14)
plt.legend()
plt.show()

# %%
# ANALISI DEL TEMPO PER OGNI ITERAZIONE 
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

# %%
# DUAL E PRIMAL GAP : plot per mostrare che il primo è un UB per il secondo 
plt.plot(list(range(k+1)), dual_gap, label = 'dual gap', color = 'blue')
fstar = func(x)
primal = function_value - fstar
plt.plot(list(range(k+1)), primal, label='primal_gap', color = 'red')

plt.ylabel ('gap',  fontsize = 14)
plt.xlabel('iter (k)',  fontsize = 14)
plt.yscale('log')
plt.title('Andamento dei gap in funzione di k', fontsize = 14)
plt.legend()
plt.show()

# %%
# TABELLA DIFFERENZA STEPSIZES
folder_path = r'C:\Users\Valeria\Documents\GitHub\Optimization\1000' # path assoluto della cartella 1000
esperimenti = []
file_names = os.listdir(folder_path)

for file_name in file_names:
    if file_name.endswith('.dmx'):
        file_path = os.path.join(folder_path, file_name)
        relative_path = os.path.relpath(file_path, os.getcwd())
        esperimenti.append(relative_path)

# %%

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
        x, _, elapsed_time, _, k, _, _, _ = FW_trad_modificato(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=True, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    try:
        x_nonopt, _, elapsed_time_nonopt, _, k_nonopt, _, _, _ = FW_trad_modificato(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=False, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    updated_nome_file_dmx = nome_file_dmx.replace('1000\\', '')
    data['nome_file_dmx'].append(updated_nome_file_dmx)
    data['iter_optimal'].append(k)
    data['iter_non_optimal'].append(k_nonopt)
    data['time_optimal'].append(elapsed_time[-1])
    data['time_non_optimal'].append(elapsed_time_nonopt[-1])

df = pd.DataFrame(data)

# %%
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
        x, function_value, elapsed_time, _, k, _, _, _ = FW_trad_modificato(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=True, visualize_res=False)
    except ProblemUnfeasibleError as e:
        continue

    try:
        x_nonopt, function_value_nonopt, elapsed_time_nonopt, _, k_nonopt, _, _, _ = FW_trad_modificato(b, n, from_, to, f_tol, time_tol, epsilon, Q, q, u, tau1, max_iter, numero_archi, x_0, step_size_ottimo=False, visualize_res=False)
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
    # plt.ylim(1e-1, 1e13)
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

    
    

# %%
import gurobipy as gp
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

#ariabili di decisione + vincolo sulla x 
x = []
for i in range(m):
    x.append(model.addVar(lb=0.0, ub=u[i], name=f"x_{i}"))


# funzione obiettivo
model.setObjective(x @ Q @ x + q @ x, sense=gp.GRB.MINIMIZE)

# Vincoli 


for i in range(n):
   
    model.addConstr(sum(E[i][j] * x[j] for j in range(m)) == b[i], name=f"flow_balance_{i}")
    

# Risoluzione del problema
model.optimize()

# Stampare il risultato
if model.status == gp.GRB.OPTIMAL:
    print("Soluzione ottima trovata:")
   # for i in range(m):
  #      print(f"x[{i}] = {x[i].x}")
    print(f"Valore della funzione obiettivo: {model.objVal}")
else:
    print("Il problema non ha una soluzione ottimale.")

# Cleanup
model.dispose()


# %% [markdown]
# ###  Funzioni per PLOT
# 

# %%
# queste due funzioni plottano il FW (dual) gap e il primal gap 

def plot_dual_gap(gap_fw, gap_afw, axes):
    
    gap_fw = np.abs(gap_fw)
    gap_afw = np.abs(gap_afw)
    iterazioni_fw = list(range(1, len(gap_fw) + 1))
    iterazioni_afw = list(range(1, len(gap_afw) + 1))

    plt.style.use('ggplot')
    # plt.figure(figsize=(8, 6))
    axes[0].plot(iterazioni_fw, gap_fw, color = 'lightseagreen', label = 'FW')
    axes[0].plot(iterazioni_afw, gap_afw, color = 'darkred', label = 'AFW')
    axes[0].set_xlabel('Iterazioni (k)', fontsize= 10)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'dual gap = $|<grad(f), d>|$', fontsize=10)
    
    axes[0].set_title('Andamento del Dual Gap (aka FW Gap) in funzione di k', fontsize=11)
    axes[0].legend(fontsize='small', loc='upper right')
    plt.grid(True)

    # plt.show()


def plot_primal_gap(f_values_fw, xstar_fw, found_optimal_fw, f_values_afw, xstar_afw, found_optimal_afw, axes):   
    # f(x) - f(x*)

    # check iniziale:  
    # se l'algoritmo ha trovato x ottima (ie il criterio di stop verificato è quello sul gap, non su max iter)
    # allora ha senso calcolare f(x*)
    if found_optimal_fw and found_optimal_afw:

        xstar_fw = np.array(xstar_fw)
        xstar_afw = np.array(xstar_afw)
        fxstar_fw = xstar_fw.T @ Q @ xstar_fw + q @ xstar_fw
        fxstar_afw = xstar_afw.T @ Q @ xstar_afw + q @ xstar_afw
        gap_fw = np.abs(f_values_fw - fxstar_fw)
        gap_afw = np.abs(f_values_afw - fxstar_afw)
        iterazioni_fw = list(range(1, len(f_values_fw) + 1))
        iterazioni_afw = list(range(1, len(f_values_afw) + 1))
        
        plt.style.use('ggplot')
        #plt.figure(figsize=(8, 6))
        axes[1].plot(iterazioni_fw, gap_fw, color = 'lightseagreen', label = 'FW')
        axes[1].plot(iterazioni_afw, gap_afw, color = 'darkred', label = 'AFW')
        axes[1].set_xlabel('Iterazioni (k)', fontsize=10)
        axes[1].set_yscale('log')
        axes[1].set_ylabel(r'primal gap = $|f(x) - f(x*)|$', fontdict={'style': 'italic', 'fontsize':10})
        
        axes[1].set_title('Andamento del Primal Gap in funzione di k', fontsize=11)
        axes[1].legend(fontsize='small', loc='upper right')
        
        plt.grid(True)

        # plt.show()
        
    else: 
        ('primal gap non disponibile')

# %%
def plot_dual_gap_bis(gap_fw, gap_afw, p, alpha): # senza axes 
    
    gap_fw = np.abs(gap_fw)
    gap_afw = np.abs(gap_afw)
    iterazioni_fw = list(range(1, len(gap_fw) + 1))
    iterazioni_afw = list(range(1, len(gap_afw) + 1))

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    plt.plot(iterazioni_fw, gap_fw, color = 'lightseagreen', label = 'FW')
    plt.plot(iterazioni_afw, gap_afw, color = 'darkred', label = 'AFW')
    # axes[0].axhline(y = 0.0001, color='red', linestyle='--', label='Valore stop epsilon')
    plt.xlabel('Iterazioni (k)', fontsize= 12)
    plt.yscale('log')
    plt.ylabel(r'dual gap = $|<grad(f), d>|$', fontsize=12)
    # plt.xticks(iterazioni_fw)
    plt.title('Andamento del FW Gap in funzione di k - p={}, alpha={}'.format(p, alpha), fontsize=14)
    plt.legend(fontsize='small', loc='upper right')
    plt.grid(True)

    plt.show()


# %%
# l'obiettivo di questa funzione è mostrare che effettivamente il FW gap (dual gap) costituisce un upper bound al primal gap


def plot_dual_and_primal_gap(f_values, dual_gap, xstar):
    dual_gap = np.abs(dual_gap)
    iterazioni = list(range(1, len(dual_gap) + 1))
    
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    
    plt.plot(iterazioni, dual_gap, color='lightseagreen', label='Dual Gap (aka FW Gap)')
    
    xstar = np.array(xstar)
    fxstar = xstar.T @ Q @ xstar + q @ xstar
    primal_gap = np.abs(f_values - fxstar)
    plt.plot(iterazioni, primal_gap, color='dodgerblue', label='Primal Gap')
    
    plt.xlabel('Iterazioni (k)')
    plt.ylabel('Gap', fontdict={'style': 'italic'})
    
    plt.title('Andamento del Dual Gap e del Primal Gap in funzione di k - 1000 nodi', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()



