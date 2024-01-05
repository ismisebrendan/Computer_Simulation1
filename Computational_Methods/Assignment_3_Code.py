from ising_lattice_2d_Brendan_Watters import IsingLattice2D
import numpy as np
import matplotlib.pyplot as plt

#
# Please note that this code takes a while to run due to the large number of calculations needed in each step, as well as the repetitions I added which aren't strictly necessary, but did help to reduce the effects of the randomness of the system
#

# Setting up the initial lattice parameters
L_x, L_y = 10, 10

############### EXERCISE 1 ###############
# Compute observables as a function of magnetic field
# for fixed temperature, k_B T/J = 1.0 and k_B T/J = 4.0

# The two temperatures to be investigated
Ts = [1.0, 4.0]

# Array of h values
h_arr = np.arange(0, 2, 0.01)

for T in Ts:
    # Blank arrays for the saved values
    M_arr = np.zeros_like(h_arr)
    E_arr = np.zeros_like(h_arr)
    chi_arr = np.zeros_like(h_arr)
    C_arr = np.zeros_like(h_arr)
    
    # Loop some number of times and then get the average
    n_loops = 10
    for i in range(n_loops):
        # Start from the same initial state in each loop
        S = np.random.choice([-1,1],(L_x,L_y))

        temp_M = np.array([])
        temp_E = np.array([])
        temp_chi = np.array([])
        temp_C = np.array([])
        
        # Find the values after subjecting the system to a metropolis algorithm for all h values
        for h in h_arr:
            ising = IsingLattice2D(L_x, L_y, T, h)
            ising.set_state(S)

            M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis()

            temp_M = np.append(temp_M, M)
            temp_E = np.append(temp_E, E)
            temp_chi = np.append(temp_chi, chi)
            temp_C = np.append(temp_C, C)
        
        M_arr = M_arr + temp_M
        E_arr = E_arr + temp_E
        chi_arr = chi_arr + temp_chi
        C_arr = C_arr + temp_C
    
    M_arr = M_arr/n_loops
    E_arr = E_arr/n_loops
    chi_arr = chi_arr/n_loops
    C_arr = C_arr/n_loops
    
    # Plot them
    fig, ax = plt.subplots(4,1)
    ax[0].plot(h_arr, M_arr)
    ax[0].set(ylabel=r'$\langle$M$\rangle$')
    ax[1].plot(h_arr, E_arr)
    ax[1].set(ylabel=r'$\langle$E$\rangle/J$')
    ax[2].plot(h_arr, chi_arr)
    ax[2].set(ylabel=r'$\chi/k_B$')
    ax[3].plot(h_arr, C_arr)
    ax[3].set(ylabel=r'$C/k_B$', xlabel=r'$h$')
    fig.suptitle(r'Average magnetisation ($\langle$M$\rangle$), energy ($\langle$E$\rangle$), magnetic susceptibility ($\chi$) and''\n'r'heat capacity (C) against magnetic field strength (h) for $k_BT/J$ = '+str(T))
    plt.tight_layout()
    plt.show()

############### EXERCISE 2 ###############

# Compute observables as a function of temperature
# for fixed magnetic field, h = 0.0

h = 0

# Array of T values
T_arr = np.arange(0.01, 4, 0.01)

# Start from the same initial state in each loop
S = np.random.choice([-1,1],(L_x,L_y))

M_arr = np.array([])
E_arr = np.array([])
chi_arr = np.array([])
C_arr = np.array([])
    
# Find the values after subjecting the system to a metropolis algorithm for all h values
for T in T_arr:

    ising = IsingLattice2D(L_x, L_y, T, h)
    ising.set_state(S)

    M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis()

    M_arr = np.append(M_arr, M)
    E_arr = np.append(E_arr, E)
    chi_arr = np.append(chi_arr, chi)
    C_arr = np.append(C_arr, C)

# Plot them
fig, ax = plt.subplots(4,1)
ax[0].plot(T_arr, M_arr)
ax[0].set(ylabel=r'$\langle$M$\rangle$')
ax[1].plot(T_arr, E_arr)
ax[1].set(ylabel=r'$\langle$E$\rangle/J$')
ax[2].plot(T_arr, chi_arr)
ax[2].set(ylabel=r'$\chi/k_B$')
ax[3].plot(T_arr, C_arr)
ax[3].set(ylabel=r'$C/k_B$', xlabel=r'$k_B T/J$')
fig.suptitle(r'Average magnetisation ($\langle$M$\rangle$), energy ($\langle$E$\rangle$), magnetic susceptibility ($\chi$) and''\n'r'heat capacity (C) against temperature (T)')
plt.tight_layout()
plt.show()

############### EXERCISE 3 ###############

# Compute magnetic susceptibility and heat capacity as a function of different things
# for fixed magnetic field, h = 0.0
# and fixed temperature T = T_C = 2.00

h = 0
T = 2.00

###########################
# Function of system size #
###########################

# Arrays of dimensions
Lx_arr = np.arange(1,11,1)
Ly_arr = np.arange(1,11,1)


states = ['random', 'ones', '-ones', 'rows', 'checked']
titles = ['random initial states', 'an initial grid of 1s', 'an initial grid of -1s', 'alternating rows of 1, -1', 'a checkerboard of 1 and -1']

# Loop over different starting states
for s in range(len(states)):

    # Blank arrays for the saved values
    chi_arr = np.array([])
    C_arr = np.array([])
    
    # Find these values after subjecting the system to a metropolis algorithm
    for L_y in Ly_arr:
        for L_x in Lx_arr:

            temp_chi = np.array([])
            temp_C = np.array([])

            # Repeat each 10 times and get an average
            for i in range(10):
                # Choose the initial state
                if states[s] == 'random':
                    # Random state for each iteration
                    S = np.random.choice([-1,1],(L_x,L_y))
                elif states[s] == 'ones':
                    # Start with all 1
                    S = np.ones((L_x,L_y))
                elif states[s] == '-ones':
                    # Start with all -1
                    S = np.ones((L_x,L_y))*-1
                elif states[s] == 'rows':
                    # Start with alternating rows of -1 and 1
                    S = np.resize([1,-1], (L_x,L_y))
                elif states[s] == 'checked':
                    # Start with 'checkerboard' of 1 and -1
                    S = np.ones((L_x,L_y))
                    for x in range(1, len(S[0])+1):
                        for y in range(1, len(S[:,0])+1):
                            S[y-1,x-1] = (-1)**(x+y)

                ising = IsingLattice2D(L_x, L_y, T, h)
                ising.set_state(S)

                M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis()

                temp_chi = np.append(temp_chi, chi)
                temp_C = np.append(temp_C, C)

            chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
            C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))

    # Reshape them as L_y x L_x arrays
    chi_arr = np.reshape(chi_arr, (len(Lx_arr), len(Ly_arr)))
    C_arr = np.reshape(C_arr, (len(Lx_arr), len(Ly_arr)))

    # Make 2D colour maps
    plt.pcolormesh(Lx_arr, Ly_arr, chi_arr, cmap='plasma')
    plt.xlabel(r'$L_x$')
    plt.ylabel(r'$L_y$')
    plt.title(r'A colour map of the magnetic susceptibility ($\chi$)''\nfor different system sizes, for '+titles[s])
    plt.colorbar()
    plt.show()

    plt.pcolormesh(Lx_arr, Ly_arr, C_arr, cmap='plasma')
    plt.xlabel(r'$L_x$')
    plt.ylabel(r'$L_y$')
    plt.title(r'A colour map of the heat capacity ($C$)''\nfor different system sizes, for '+titles[s])
    plt.colorbar()
    plt.show()

#################################
# Now function of initial state #
#################################

# Blank arrays for the saved values
chi_arr = np.array([])
C_arr = np.array([])

L_x, L_y = 10, 10

# Loop over different starting states
for s in range(len(states)):

    temp_chi = np.array([])
    temp_C = np.array([])

    # Repeat each 10000 times and get an average
    for i in range(10000):
        if states[s] == 'random':
            # Random state for each iteration
            S = np.random.choice([-1,1],(L_x,L_y))
        elif states[s] == 'ones':
            # Start with all 1
            S = np.ones((L_x,L_y))
        elif states[s] == '-ones':
            # Start with all -1
            S = np.ones((L_x,L_y))*-1
        elif states[s] == 'rows':
            # Start with alternating rows of -1 and 1
            S = np.resize([1,-1], (L_x,L_y))
        elif states[s] == 'checked':
            # Start with 'checkerboard' of 1 and -1
            S = np.ones((L_x,L_y))
            for x in range(1, len(S[0])+1):
                for y in range(1, len(S[:,0])+1):
                    S[y-1,x-1] = (-1)**(x+y)

        ising = IsingLattice2D(L_x, L_y, T, h)
        ising.set_state(S)

        M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis()

        temp_chi = np.append(temp_chi, chi)
        temp_C = np.append(temp_C, C)

    chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
    C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))

# Plot
plt.bar(['Random', 'Grid of 1s', 'Grid of -1s', 'Rows of\n1s and-1s', 'Checkerboard of\n1s and-1s'], chi_arr)
plt.title(r'The magnetic susceptibility ($\chi$) for different initial states')
plt.ylabel(r'$\chi/k_B$')
plt.show()

plt.bar(['Random', 'Grid of 1s', 'Grid of -1s', 'Rows of\n1s and-1s', 'Checkerboard of\n1s and-1s'], C_arr)
plt.title(r'The heat capacity ($C$) for different initial states')
plt.ylabel(r'$C/k_B$')  
plt.show()


#####################################
# Random sweeps v sequential sweeps #
#####################################

# Blank arrays for the saved values
chi_arr = np.array([])
C_arr = np.array([])

## Sequential sweeps
temp_chi = np.array([])
temp_C = np.array([])

# Repeat each 100 times and get an average
for i in range(100):
    # Random initial states
    S = np.random.choice([-1,1],(L_x,L_y))

    ising = IsingLattice2D(L_x, L_y, T, h)
    ising.set_state(S)

    M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis()

    temp_chi = np.append(temp_chi, chi)
    temp_C = np.append(temp_C, C)

chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))

## Random sweeps
temp_chi = np.array([])
temp_C = np.array([])

# Repeat each 100 times and get an average
for i in range(100):
    # Random initial states
    S = np.random.choice([-1,1],(L_x,L_y))

    ising = IsingLattice2D(L_x, L_y, T, h)
    ising.set_state(S)

    M, E, M2, E2, var_M, var_E, chi, C = ising.rand_metropolis()

    temp_chi = np.append(temp_chi, chi)
    temp_C = np.append(temp_C, C)

chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))


# Plot
plt.bar(['Sequential sweeps', 'Random sweeps'], chi_arr)
plt.title(r'The magnetic susceptibility ($\chi$) for sequential and random sweeps')
plt.ylabel(r'$\chi/k_B$')
plt.show()

plt.bar(['Sequential sweeps', 'Random sweeps'], C_arr)
plt.title(r'The heat capacity ($C$) for sequential and random sweeps')
plt.ylabel(r'$C/k_B$')
plt.show()

###############################################
# Now function of number of metropolis sweeps #
###############################################

# Blank arrays for the saved values
chi_arr = np.array([])
C_arr = np.array([])

L_x, L_y = 10, 10

# Steps of 100
sweeps = np.arange(100, 1001, 100)

# Loop over different starting states
for s in sweeps:

    temp_chi = np.array([])
    temp_C = np.array([])

    # Repeat each 100 times and get an average
    for i in range(100):
        S = np.random.choice([-1,1],(L_x,L_y))

        ising = IsingLattice2D(L_x, L_y, T, h)
        ising.set_state(S)

        M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis(n_sweeps=s)

        temp_chi = np.append(temp_chi, chi)
        temp_C = np.append(temp_C, C)

    chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
    C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))

# Plot
fig, ax = plt.subplots(2,1)
ax[0].plot(sweeps, chi_arr)
ax[0].set(ylabel=r'$\chi/k_B$')
ax[1].plot(sweeps, C_arr)
ax[1].set(ylabel=r'$C/k_B$', xlabel='Number of Metropolis sweeps')
fig.suptitle(r'The magnetic susceptibility ($\chi$) and heat capacity ($C$)''\nfor different numbers of metropolis sweeps')
plt.tight_layout()
plt.show()

# Steps of 1000
sweeps = np.arange(1000, 10001, 1000)

# Loop over different starting states
for s in sweeps:

    temp_chi = np.array([])
    temp_C = np.array([])

    # Repeat each 100 times and get an average
    for i in range(100):
        S = np.random.choice([-1,1],(L_x,L_y))

        ising = IsingLattice2D(L_x, L_y, T, h)
        ising.set_state(S)

        M, E, M2, E2, var_M, var_E, chi, C = ising.metropolis(n_sweeps=s)

        temp_chi = np.append(temp_chi, chi)
        temp_C = np.append(temp_C, C)

    chi_arr = np.append(chi_arr, np.sum(temp_chi)/len(temp_chi))
    C_arr = np.append(C_arr, np.sum(temp_C)/len(temp_C))

# Plot
fig, ax = plt.subplots(2,1)
ax[0].plot(sweeps, chi_arr)
ax[0].set(ylabel=r'$\chi/k_B$')
ax[1].plot(sweeps, C_arr)
ax[1].set(ylabel=r'$C/k_B$', xlabel='Number of Metropolis sweeps')
fig.suptitle(r'The magnetic susceptibility ($\chi$) and heat capacity ($C$)''\nfor different numbers of metropolis sweeps')
plt.tight_layout()
plt.show()
