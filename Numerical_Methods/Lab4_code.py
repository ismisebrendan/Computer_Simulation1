import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

print("---------------------")
print("-                   -")
print("-   ismisebrendan   -")
print("-  2 November 2023  -")
print("-                   -")
print("---------------------")

# The Poisson distribution function
def poisson(n, mean):
    return mean ** n / factorial(n) * np.e ** (-mean)

n_list = np.arange(0,50,0.1)

plt.rcParams.update({'font.size': 15})

# Plot the Poisson distribution for different mean values
plt.plot(n_list, poisson(n_list, 1), label=r'$\langle n \rangle=1$')
plt.plot(n_list, poisson(n_list, 5), label=r'$\langle n \rangle=5$')
plt.plot(n_list, poisson(n_list, 10), label=r'$\langle n \rangle=10$')
plt.title('The Poisson distribution for the different mean values')
plt.xlabel('n')
plt.ylabel('P(n)')
plt.legend()
plt.show()

# The functions for the different sums
def sum(N, mean):
    tot = 0
    for i in range(0, N+1):
        tot += poisson(i, mean)
    return tot

def sumn(N,mean):
    tot = 0
    for i in range(0, N+1):
        tot += i * poisson(i, mean)
    return tot

def sumn2(N,mean):
    tot = 0
    for i in range(0, N+1):
        tot += i**2 * poisson(i, mean)
    return tot

print("The sum of P(n) for 0 <= n <= 50, for n an integer and for different mean values of n.")
print("mean n = 1, sum = " + str(sum(50,1)))
print("mean n = 5, sum = " + str(sum(50,5)))
print("mean n = 10, sum = " + str(sum(50,10)))

print("The sum of nP(n) for 0 <= n <= 50, for n an integer and for different mean values of n.")
print("mean n = 1, sum = " + str(sumn(50,1)))
print("mean n = 5, sum = " + str(sumn(50,5)))
print("mean n = 10, sum = " + str(sumn(50,10)))

print("The sum of n^2 P(n) for 0 <= n <= 50, for n an integer and for different mean values of n.")
print("mean n = 1, sum = " + str(sumn2(50,1)))
print("mean n = 5, sum = " + str(sumn2(50,5)))
print("mean n = 10, sum = " + str(sumn2(50,10)))

#####################
#                   #
#   Dart throwing   #
#                   #
#####################

n_list = np.arange(0, 50, 1)

def dart_throwing(N,L,T):
    h = np.zeros([N])

    # Do this all T times
    for x in range(T):
        # Throw darts
        dart_locations = np.random.randint(0, L, N)

        bins = np.zeros([L])
        # Loop over all the values [0-L)
        for i in range(L):
            # Loop over all elements of dart_locations
            for j in dart_locations:
                # Count every occurrence of each element
                if j == i:
                    bins[i] += 1

        # Now have an array with the number of times a dart lands in each bin
        # A dictionary with key = number of darts in each bin, value = number of occurrences of each value
        h_dict = dict(zip(np.unique(bins, return_counts=True)[0], np.unique(bins, return_counts=True)[1]))

        # Convert this to an array
        hi = np.array([])
        for i in range(N):
            if i in h_dict:
                hi = np.append(hi, h_dict[i])
            else:
                hi = np.append(hi, 0)

        # Add this to the running total h
        h = h + hi
    
    # find the mean of the random distribution
    h_mean = 0
    for i in range(len(h)):
        h_mean += i*h[i]
    print('The mean of darts in each region for L = '+str(L)+', N = '+str(N)+' and T = '+ str(T) + ' trials is ' + str(h_mean))

    # Normalise h and the mean of h
    h = h/(L*T)

    h_mean = h_mean/(L*T)

    print('The mean of the Poisson distribution for L = '+str(L)+', N = '+str(N)+' and T = '+ str(T) + ' trials is ' + str(h_mean))

    # Linear graph
    plt.plot(n_list, poisson(n_list, h_mean), label=r'$P(n)$ for $\langle n \rangle=$' + str(np.round(h_mean, 2)), marker='+', mec='red')
    plt.plot(n_list, h, label=r'$P_{sim}(n)$, Random data', marker='+', mec='black')
    plt.xlabel('n')
    plt.ylabel('$P(n)$')
    plt.title('The distribution of the dart locations for L = '+str(L)+', N = '+str(N)+' and T = '+ str(T) + ' trials')
    plt.legend()
    plt.show()

    # Log graph
    plt.plot(n_list, poisson(n_list, h_mean), label=r'$P(n)$ for $\langle n \rangle=$' + str(np.round(h_mean, 2)), marker='+', mec='red')
    plt.plot(n_list, h, label=r'$P_{sim}(n)$, Random data', marker='+', mec='black')
    plt.xlabel('n')
    plt.ylabel('$P(n)$')
    plt.yscale('log')
    plt.title('The distribution of the dart locations for L = '+str(L)+', N = '+str(N)+' and T = '+ str(T) + ' trials')
    plt.legend()
    plt.show()

# L = 100
dart_throwing(50,100,10)
dart_throwing(50,100,100)
dart_throwing(50,100,1000)
dart_throwing(50,100,10000)

# L = 5
dart_throwing(50,5,10)
dart_throwing(50,5,1000)
dart_throwing(50,5,10000)

