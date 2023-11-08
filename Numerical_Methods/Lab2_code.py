import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

print("---------------------")
print("-                   -")
print("-   ismisebrendan   -")
print("-  12 October 2023  -")
print("-                   -")
print("-----------------=---")

# Load data
t, cases = np.loadtxt("COVIDData.dat", skiprows=1, unpack=True)

# Plot initial graph
plt.subplots(figsize=(14, 7))
plt.scatter(t, cases, s=5, label='Cases reported each day')
plt.title('Graph of cases reported each day (n) against days since 3 March 2020 (t), up to 30 September 2021')
plt.ylabel('n')
plt.xlabel('t')
plt.annotate("First wave peak", xy=(53,936), xytext=(20, 4000), arrowprops=dict(arrowstyle="->")) 
plt.annotate("Second wave peak", xy=(231, 1283), xytext=(200, 4000), arrowprops=dict(arrowstyle="->"))
plt.annotate("Third wave peak", xy=(313, 8248), xytext=(350, 6000), arrowprops=dict(arrowstyle="->"))
plt.annotate("Fourth wave peak", xy=(538, 2125), xytext=(500, 4000), arrowprops=dict(arrowstyle="->"))
plt.legend()
plt.savefig('linear.png')
plt.show()

# Plot ln(n) v t
lncases = np.log(cases)

plt.subplots(figsize=(14, 7))
plt.scatter(t, lncases, s=5, label='Cases reported each day')
plt.title('Graph of natural log of cases reported each day (ln(n)) against days since 3 March 2020 (t), up to 30 September 2021')
plt.ylabel('ln(n)')
plt.xlabel('t')
plt.legend()
plt.savefig('log.png')
plt.show()

#################################
#                               #
#   Fit the first three waves   #
#                               #
#################################

# Fitting function
def func(x, a, b):
    return a + b*x

# Plot ln(n) v t
plt.subplots(figsize=(14, 7))
plt.scatter(t, lncases, s=5, label='Cases reported each day')
plt.title('Graph of natural log of cases reported each day (ln(n)) against days since 3 March 2020 (t), up to 30 September 2021')
plt.ylabel('ln(n)')
plt.xlabel('t')

# Define first wave
wave1 = np.arange(0,50,1)                                                   # range of days
fit_wave1 = opt.curve_fit(func, t[wave1], lncases[wave1])[0]                # linear regression
plt.plot(wave1, fit_wave1[0] + fit_wave1[1]*wave1, label="Fit of wave 1")   # plot the line

# Define first wave decay
wave1_end = np.arange(50,120,1)
fit_wave1_end = opt.curve_fit(func, t[wave1_end], lncases[wave1_end])[0]
plt.plot(wave1_end, fit_wave1_end[0] + fit_wave1_end[1]*wave1_end, label="Fit of wave 1's decay")

# Define second wave
wave2 = np.arange(120,234,1)
fit_wave2 = opt.curve_fit(func, t[wave2], lncases[wave2])[0]
plt.plot(wave2, fit_wave2[0] + fit_wave2[1]*wave2, label="Fit of wave 2")

# Define second wave decay
wave2_end = np.arange(234,278,1)
fit_wave2_end = opt.curve_fit(func, t[wave2_end], lncases[wave2_end])[0]
plt.plot(wave2_end, fit_wave2_end[0] + fit_wave2_end[1]*wave2_end, label="Fit of wave 2's decay")

# Define third wave
wave3 = np.arange(278,314,1)
fit_wave3 = opt.curve_fit(func, t[wave3], lncases[wave3])[0]
plt.plot(wave3, fit_wave3[0] + fit_wave3[1]*wave3, label="Fit of wave 3")

# Define third wave decay
wave3_end = np.arange(314,374,1)
fit_wave3_end = opt.curve_fit(func, t[wave3_end], lncases[wave3_end])[0]
plt.plot(wave3_end, fit_wave3_end[0] + fit_wave3_end[1]*wave3_end, label="Fit of wave 3's decay")

# Show the graph and legend and save
plt.legend()
plt.savefig('log_fit.png')
plt.show()

# Print the values of a and b for each fit
print("For wave 1: a = "+str(fit_wave1[0])+" b = "+str(fit_wave1[1]))
print("For wave 1's decay: a = "+str(fit_wave1_end[0])+" b = "+str(fit_wave1_end[1]))
print("For wave 2: a = "+str(fit_wave2[0])+" b = "+str(fit_wave2[1]))
print("For wave 2's decay: a = "+str(fit_wave2_end[0])+" b = "+str(fit_wave2_end[1]))
print("For wave 3: a = "+str(fit_wave3[0])+" b = "+str(fit_wave3[1]))
print("For wave 3's decay: a = "+str(fit_wave3_end[0])+" b = "+str(fit_wave3_end[1]))

##########################
#                        #
#   Linear Graph again   #
#                        #
##########################

# Assign variables for exponential fits
t01 = 1
n01 = np.exp(fit_wave1[0] + fit_wave1[1] * t01)
t01_end = 50
n01_end = np.exp(fit_wave1_end[0] + fit_wave1_end[1] * t01_end)
t02 = 199
n02 = np.exp(fit_wave2[0] + fit_wave2[1] * t02)
t02_end = 233
n02_end = np.exp(fit_wave2_end[0] + fit_wave2_end[1] * t02_end)
t03 = 277
n03 = np.exp(fit_wave3[0] + fit_wave3[1] * t03)
t03_end = 314
n03_end = np.exp(fit_wave3_end[0] + fit_wave3_end[1] * t03_end)

# Print out n0 for each (lambda = b so it is known)
print("For wave 1: n0 = "+str(n01))
print("For wave 1's deacy: n0 = "+str(n01_end))
print("For wave 2: n0 = "+str(n02))
print("For wave 2's deacy: n0 = "+str(n02_end))
print("For wave 3: n0 = "+str(n03))
print("For wave 3's deacy: n0 = "+str(n03_end))

# Plot linearly again
plt.subplots(figsize=(14, 7))
plt.scatter(t, cases, s=5, label='Cases reported each day')
plt.title('Graph of cases reported each day (n) against days since 3 March 2020 (t), up to 30 September 2021')
plt.ylabel('n')
plt.xlabel('t')
plt.yscale('linear')

# Plot fits
plt.plot(wave1, n01*np.exp(fit_wave1[1]*(wave1-t01)), label="Fit of wave 1")
plt.plot(wave1_end, n01_end*np.exp(fit_wave1_end[1]*(wave1_end-t01_end)), label="Fit of wave 1's decay")
plt.plot(wave2, n02*np.exp(fit_wave2[1]*(wave2-t02)), label="Fit of wave 2")
plt.plot(wave2_end, n02_end*np.exp(fit_wave2_end[1]*(wave2_end-t02_end)), label="Fit of wave 2's decay")
plt.plot(wave3, n03*np.exp(fit_wave3[1]*(wave3-t03)), label="Fit of wave 3")
plt.plot(wave3_end, n03_end*np.exp(fit_wave3_end[1]*(wave3_end-t03_end)), label="Fit of wave 3's decay")
plt.legend()
plt.savefig('linear_fit.png')
plt.show()

####################################
#                                  #
#   Gaussian fit for fourth wave   #
#                                  #
####################################

# The fourth wave along with the lead up to it appear to follow a Gaussian curve

# Fitting function
def gaussian(x, a, b, c, d):
    return a * np.exp(-((x-b)/c)**2) + d

# Define fourth wave
wave4 = np.arange(374,575,1)                                                                # range of days
fit_wave4 = opt.curve_fit(gaussian, t[wave4], cases[wave4], p0=[2000, 538, 30, 300])[0]     # fitting with initial guesses

# Scatter data
plt.subplots(figsize=(14, 7))
plt.scatter(t, cases, s=5, label='Cases reported each day')
plt.title('Graph of cases reported each day (n) against days since 3 March 2020 (t), up to 30 September 2021')
plt.ylabel('n')
plt.xlabel('t')
plt.yscale('linear')

# Assign wave 4 variables and print them 
a, b, c, d = fit_wave4
print("For wave 4:\na = "+str(a)+"\nb = "+str(b)+"\nc = "+str(c)+"\nd = "+str(d))

# Plot all fits, show and save the graph
plt.plot(wave1, n01*np.exp(fit_wave1[1]*(wave1-t01)), label="Fit of wave 1")
plt.plot(wave1_end, n01_end*np.exp(fit_wave1_end[1]*(wave1_end-t01_end)), label="Fit of wave 1's decay")
plt.plot(wave2, n02*np.exp(fit_wave2[1]*(wave2-t02)), label="Fit of wave 2")
plt.plot(wave2_end, n02_end*np.exp(fit_wave2_end[1]*(wave2_end-t02_end)), label="Fit of wave 2's decay")
plt.plot(wave3, n03*np.exp(fit_wave3[1]*(wave3-t03)), label="Fit of wave 3")
plt.plot(wave3_end, n03_end*np.exp(fit_wave3_end[1]*(wave3_end-t03_end)), label="Fit of wave 3's decay")
plt.plot(wave4, a * np.exp(-((wave4-b)/c)**2) + d, label="Fit of wave 4")
plt.legend()
plt.savefig('linear_fit_all.png')
plt.show()
