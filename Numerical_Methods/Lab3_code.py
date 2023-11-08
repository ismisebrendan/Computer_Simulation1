import numpy as np
import matplotlib.pyplot as plt

print("---------------------")
print("-                   -")
print("-   ismisebrendan   -")
print("-  25 October 2023  -")
print("-                   -")
print("---------------------")

# Producing direction field
t = np.linspace(0,5,25)
x = np.linspace(-3,3,25)

T, X = np.meshgrid(t,x)

# The colours of the quiver plot
plt.rcParams['image.cmap'] = 'plasma'

########################
#                      #
#   Define functions   #
#                      #
########################

# f(x,t) = dx/dt = (1+t)*x + 1 - 3*t + t**2
def f(x,t):
    return (1+t)*x + 1 - 3*t + t**2

dx = (1+T)*X + 1 - 3*T + T**2
dt = np.ones(dx.shape)

# Quiver plot and limits (these elements are in every plot)
def plots():
    plt.subplots(figsize=(7,7))
    Q = plt.quiver(T, X, dt, dx, np.hypot(T, X))
    qk = plt.quiverkey(Q, 0.9, 0.9, 10, r'$\frac{dx}{dt}$', labelpos='E', coordinates='figure')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xlim([-0.5, 5.5])
    plt.ylim([-3.5, 3.5])

# Simple Euler Method
def simple(x, t, deltat, maxt): # x values, t values, step size, max value of t to evaluate to
    x_list_simple = np.array([x])
    t_list_simple = np.array([t])

    while t <= maxt:
        x_i = x
        t_i = t

        x = x_i + f(x_i, t_i) * deltat
        t += deltat

        x_list_simple = np.append(x_list_simple, x)
        t_list_simple = np.append(t_list_simple, t)
    
    return x_list_simple, t_list_simple

# Improved Euler Method
def improved(x, t, deltat, maxt): # x values, t values, step size, max value of t to evaluate to
    x_list_improved = np.array([x])
    t_list_improved = np.array([t])

    while t <= maxt:
        x_i = x
        t_i = t

        x = x_i + ( f(x_i, t_i) + f(x_i + f(x_i, t_i) * deltat, t_i + deltat) ) * deltat/2  
        t += deltat

        x_list_improved = np.append(x_list_improved, x)
        t_list_improved = np.append(t_list_improved, t)

    return x_list_improved, t_list_improved

# Runge-Kutta method
def rk_4(x, t, deltat, maxt): # x values, t values, step size, max value of t to evaluate to
    x_list_rk = np.array([x])
    t_list_rk = np.array([t])

    while t <= maxt:
        x_i = x
        t_i = t

        x1 = x_i
        t1 = t_i
        x2 = x_i + 1/2 * f(x_i, t_i) * deltat
        t2 = t_i + deltat/2
        x3 = x_i + 1/2 * f(x2, t2) * deltat
        t3 = t_i + deltat/2
        x4 = x_i + f(x3, t3) * deltat
        t4 = t_i + deltat

        x = x1 + deltat/6 * (f(x1, t1) + 2*f(x2, t2) + 2*f(x3, t3) + f(x4, t4))
        t += deltat

        x_list_rk = np.append(x_list_rk, x)
        t_list_rk = np.append(t_list_rk, t)
        
    return x_list_rk, t_list_rk

##################################
#                                #
#   Plot direction field alone   #
#                                #
##################################

plots()
plt.title('The direction field of the differential equation')
plt.show()
plt.close()

##########################
#                        #
#   For delta t = 0.04   #
#                        #
##########################

###########################
#   Simple Euler method   #
###########################

x_list, t_list = simple(0.0655, 0, 0.04, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the Simple Euler Method for $\Delta$t = 0.04')
plt.plot(t_list, x_list, label='Simple Euler Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

#############################
#   Improved Euler method   #
#############################

x_list, t_list = improved(0.0655, 0, 0.04, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the Improved Euler Method for $\Delta$t = 0.04')
plt.plot(t_list, x_list, label='Improved Euler Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

#######################################
#   Fourth Order Runge-Kutta method   #
#######################################

x_list, t_list = rk_4(0.0655, 0, 0.04, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the 4$^{th}$ order Runge-Kutta Method for $\Delta$t = 0.04')
plt.plot(t_list, x_list, label=r'4$^{th}$ order RK Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

##########################
#                        #
#   For delta t = 0.02   #
#                        #
##########################

###########################
#   Simple Euler method   #
###########################

x_list, t_list = simple(0.0655, 0, 0.02, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the Simple Euler Method for $\Delta$t = 0.02')
plt.plot(t_list, x_list, label='Simple Euler Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

#############################
#   Improved Euler method   #
#############################

x_list, t_list = improved(0.0655, 0, 0.02, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the Improved Euler Method for $\Delta$t = 0.02')
plt.plot(t_list, x_list, label='Improved Euler Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

#######################################
#   Fourth Order Runge-Kutta method   #
#######################################

x_list, t_list = rk_4(0.0655, 0, 0.02, 5)

# Plot direction field with solution
plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the 4$^{th}$ order Runge-Kutta Method for $\Delta$t = 0.02')
plt.plot(t_list, x_list, label=r'4$^{th}$ order RK Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

################
#              #
#   Together   #
#              #
################

# Delta t = 0.04

x_simple, t_simple = simple(0.0655, 0, 0.04, 5)
x_improved, t_improved = improved(0.0655, 0, 0.04, 5)
x_rk, t_rk = rk_4(0.0655, 0, 0.04, 5)

plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the three methods for $\Delta$t = 0.04')
plt.plot(t_simple, x_simple, label='Simple Euler Method Solution')
plt.plot(t_improved, x_improved, label='Improved Euler Method Solution')
plt.plot(t_rk, x_rk, label=r'4$^{th}$ order RK Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

# Delta t = 0.02

x_simple, t_simple = simple(0.0655, 0, 0.02, 5)
x_improved, t_improved = improved(0.0655, 0, 0.02, 5)
x_rk, t_rk = rk_4(0.0655, 0, 0.02, 5)

plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the three methods for $\Delta$t = 0.02')
plt.plot(t_simple, x_simple, label='Simple Euler Method Solution')
plt.plot(t_improved, x_improved, label='Improved Euler Method Solution')
plt.plot(t_rk, x_rk, label=r'4$^{th}$ order RK Method Solution')
plt.legend(loc='lower right')
plt.show()
plt.close()

# Both together

x_simple, t_simple = simple(0.0655, 0, 0.04, 5)
x_improved, t_improved = improved(0.0655, 0, 0.04, 5)
x_rk, t_rk = rk_4(0.0655, 0, 0.04, 5)

plots()
plt.title('The direction field of the differential equation\nwith the solution plotted as calculated\nby the three methods')
plt.plot(t_simple, x_simple, label=r'Simple Euler Method Solution for $\Delta t=0.04$')
plt.plot(t_improved, x_improved, label=r'Improved Euler Method Solution for $\Delta t=0.04$')
plt.plot(t_rk, x_rk, label=r'4$^{th}$ order RK Method Solution for $\Delta t=0.04$')

x_simple, t_simple = simple(0.0655, 0, 0.02, 5)
x_improved, t_improved = improved(0.0655, 0, 0.02, 5)
x_rk, t_rk = rk_4(0.0655, 0, 0.02, 5)

plt.plot(t_simple, x_simple, label=r'Simple Euler Method Solution for $\Delta t=0.02$')
plt.plot(t_improved, x_improved, label=r'Improved Euler Method Solution for $\Delta t=0.02$')
plt.plot(t_rk, x_rk, label=r'4$^{th}$ order RK Method Solution for $\Delta t=0.02$')
plt.legend(loc='lower right',fontsize='5')
plt.show()
plt.close()
