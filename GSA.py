import random
import numpy
import math
from solution import solution
import time

def gConstant(l,iters):
    alfa = 20
    G0 = 100
    Gimd = numpy.exp(-alfa*float(l)/iters)
    G = G0 * Gimd
    return G


def gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower):
    final_per = 2
    if ElitistCheck == 1:
        kbest = final_per + (1-l/iters)*(100-final_per)
        kbest = round(PopSize*kbest/100)
    else:
        kbest = PopSize
            
    kbest = int(kbest)
    ds = sorted(range(len(M)), key=lambda k: M[k],reverse=True)
        
    Force = numpy.zeros((PopSize,dim))
    # Force = Force.astype(int)
    
    for r in range(0,PopSize):
        for ii in range(0,kbest):
            z = ds[ii]
            R = 0
            if z != r:                    
                x=pos[r,:]
                y=pos[z,:]
                esum=0
                imval = 0
                for t in range(0,dim):
                    imval = ((x[t] - y[t])** 2)
                    esum = esum + imval
                    
                R = math.sqrt(esum)
                
                for k in range(0,dim):
                    randnum=random.random()
                    Force[r,k] = Force[r,k]+randnum*(M[z])*((pos[z,k]-pos[r,k])/(R**Rpower+numpy.finfo(float).eps))
                    
    acc = numpy.zeros((PopSize,dim))
    for x in range(0,PopSize):
        for y in range (0,dim):
            acc[x,y]=Force[x,y]*G
    return acc

def massCalculation(fit,PopSize,M):
    Fmax = max(fit)
    Fmin = min(fit)
    Fsum = sum(fit)        
    Fmean = Fsum / len(fit)
        
    if Fmax == Fmin:
        M = numpy.ones(PopSize)
    else:
        best = Fmin
        worst = Fmax
        
        for p in range(0,PopSize):
           M[p] = (fit[p] - worst) / (best - worst)
            
    Msum = sum(M)
    for q in range(0, PopSize):
        M[q] = M[q] / Msum
            
    return M

def move(PopSize,dim,pos, vel, acc):
    for i in range(0, PopSize):
        for j in range (0,dim):
            r1 = random.random()
            vel[i, j] = r1 * vel[i, j] + acc[i, j]
            pos[i, j] = pos[i, j] + vel[i, j]
    
    return pos, vel

        
def GSA(objf, lb, ub, dim, args, df, clf, metric):
    """ Main GSA function for feature selection """
    
    # GSA parameters
    ElitistCheck = 1  # Use elitist strategy (focus on top-performing agents)
    Rpower = 1  # Power of distance in the force equation

    #
    PopSize, iters = args.pop_size, args.iterations

    #
    patience, early_stop = args.patience, args.early_stop
    
    # Initialize solution object to store the results
    s = solution()
    
    # Initialize velocities, fitness, masses, and best solution
    vel = numpy.zeros((PopSize, dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest = numpy.zeros(dim)  # Best feature subset found
    gBestScore = float("inf")  # Best score (minimized objective)
    
    # Initialize positions of agents (random feature subsets)
    pos = numpy.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    
    # Track convergence (best score at each iteration)
    convergence_curve = numpy.zeros(iters)

    # Initialize patience variables
    best_score_so_far = gBestScore
    no_improvement_counter = 0
    
    # Start the timer
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop for GSA iterations
    for l in range(iters):
        iter_start_time = time.time()
        improvement = False 
        # Evaluate fitness for each agent (feature subset)
        for i in range(PopSize):
            pos[i, :] = numpy.clip(pos[i, :], lb, ub)  # Keep positions within bounds
            fitness = -objf(pos[i, :], df, clf, metric, args.ignore_first)  # Evaluate fitness of the feature subset
            fit[i] = fitness  # Store fitness
            
            # Update the best solution found so far
            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :]
                improvement = True

        # Update masses based on fitness
        M = massCalculation(fit, PopSize, M)
        
        # Compute gravitational constant for this iteration
        G = gConstant(l, iters)
        
        # Calculate gravitational forces and update accelerations
        acc = gField(PopSize, dim, pos, M, l, iters, G, ElitistCheck, Rpower)
        
        # Update positions and velocities based on forces
        pos, vel = move(PopSize, dim, pos, vel, acc)
        
        # Store convergence data
        convergence_curve[l] = gBestScore

        # Check if an improvement was made in this iteration
        if improvement:
            no_improvement_counter = 0  # Reset counter if there was an improvement
        else:
            no_improvement_counter += 1  # Increment counter if no improvement

        # Print progress
        iter_time_taken = time.time() - iter_start_time
        print(f"Iteration {l+1}: \tthe best fitness is {-gBestScore:.6f}\t[{iter_time_taken:.2f}s]\t(*{no_improvement_counter}*)")

        # Early stopping check based on patience
        if early_stop and no_improvement_counter >= patience:
            print(f"Early stopping at iteration {l+1} due to no improvement for {patience} iterations.")
            break
    
    # Record final times and results
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.gBest = gBest
    s.Algorithm = "GSA"
    s.objectivefunc = objf.__name__

    return s
