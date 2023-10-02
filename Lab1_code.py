# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

while True:
    print("-----------------------")
    print("-                     -")
    print("-    ismisebrendan    -")
    print("-   02 October 2023   -")
    print("-                     -")
    print("-----------------------")
    program = input("What part of the lab would you like to run?\nHeron's method: H\nCustom Heron's method: C\nUnderflow of integers: I\nOver/underflow of floats: F\nMachine precision of floats and complex numbers: P\nDerivatives: D\nEnd program: E\n--> ").upper()

    if program == 'E':
        # End program
        break
    elif program == 'H':
        ######################
        #   Heron's method   #
        ######################
        def heron_root2(x0, iterations):
            x = x0
            roots = np.array([x])
            n = np.array([0])

            i = 0            
            while i < iterations:
                x = 1/2 * (x + 2/x)
                roots = np.append(roots,x)
                i += 1
                n = np.append(n, i)
            return np.array([n, roots])
            
        # Get the root for different x0s
        x0_2 = heron_root2(2,10)
        x0_1 = heron_root2(1,10)
        x0_neg1 = heron_root2(-1,10)
        x0_half = heron_root2(0.5,10)
        x0_10 = heron_root2(10,10)

        # Plot xn v n
        plt.plot(x0_2[0], x0_2[1], label="$x_0$ = 2")
        plt.plot(x0_1[0], x0_1[1], label="$x_0$ = 1")
        plt.plot(x0_neg1[0], x0_neg1[1], label="$x_0$ = -1")
        plt.plot(x0_half[0], x0_half[1], label="$x_0$ = 0.5")
        plt.plot(x0_10[0], x0_10[1], label="$x_0$ = 10")

        plt.title("The estimations for the square root of 2\nagainst the iterations (n) for various initial values ($x_0$)")
        plt.xlabel("n")
        plt.ylabel("x$_n$")
        plt.hlines(np.sqrt(2), 0, 10, linestyles='dotted', colors='black', label="$\sqrt{2}$")
        plt.hlines(-np.sqrt(2), 0, 10, linestyles='dotted', colors='black', label="$-\sqrt{2}$")
        plt.legend()
        plt.savefig("roots.png")
        plt.show()
        plt.close()

        # Plot error in xn^2 v n 
        error_2 = (2 - x0_2[1]**2)/2
        error_1 = (2 - x0_1[1]**2)/2
        error_neg1 = (2 - x0_neg1[1]**2)/2
        error_half = (2 - x0_half[1]**2)/2
        error_10 = (2 - x0_10[1]**2)/2

        plt.plot(x0_2[0], error_2, label="$x_0$ = 2")
        plt.plot(x0_1[0], error_1, label="$x_0$ = 1")
        plt.plot(x0_neg1[0], error_neg1, label="$x_0$ = -1")
        plt.plot(x0_half[0], error_half, label="$x_0$ = 0.5")
        plt.plot(x0_10[0], error_10, label="$x_0$ = 10")

        plt.legend()
        plt.title("The relative error in x$_n^2$ ($\epsilon$)\nagainst the iterations (n) for various initial values ($x_0$)")
        plt.xlabel("n")
        plt.ylabel("$\epsilon$")
        plt.ylim([-3,1])
        plt.savefig("errors_heron.png")
        plt.show()
        plt.close()

        print("\n\n\nHeron's method has been run \n\n\n")
    elif program == 'C':
        #####################################
        #   Heron's method for any number   #
        #####################################
        def heron_root(a ,x0, iterations):
            x = x0
            roots = np.array([x])
            n = np.array([0])

            i = 0            
            while i < iterations:
                x = 1/2 * (x + a/x)
                roots = np.append(roots,x)
                i += 1
                n = np.append(n, i)
            return np.array([n, roots])
        
        a = np.abs(float(input("What positive number do you want to find the square root of? ")))
        x0 = float(input("What value do you want for the initial estimation? "))
        it = np.abs(np.round(float(input("How many iterations do you want to try? "))))
        
        root = heron_root(a, x0, it)

        # Plot xn v n
        plt.plot(root[0], root[1], label="$x_0$ = "+str(x0))
        plt.title("The estimations for the square root of "+str(a)+"\nagainst the iterations (n) for initial value $x_0$ = "+str(x0))
        plt.xlabel("n")
        plt.ylabel("x$_n$")
        if x0 < 0:
            plt.hlines(-np.sqrt(a), 0, it, linestyles='dotted', colors='black', label="$-\sqrt{"+str(a)+"}$")
        else:
            plt.hlines(np.sqrt(a), 0, it, linestyles='dotted', colors='black', label="$\sqrt{"+str(a)+"}$")
        plt.legend()
        plt.show()
        plt.close()

        # Plot error in xn^2 v n 
        error = (a - root[1]**2)/a

        plt.plot(root[0], error, label="$x_0$ = "+str(x0))
        plt.legend()
        plt.title("The relative error in x$_n^2$ ($\epsilon$)\nagainst the iterations (n) for initial value $x_0$ = "+str(x0))
        plt.xlabel("n")
        plt.ylabel("$\epsilon$")
        plt.show()
        plt.close()

        print("\n\n\nHeron's method has been run for a = \n\n\n"+str(a))
    elif program == 'I':
        ##################################
        #   Check 'underflow for ints'   #
        ##################################
        under = 1
        for i in range(100000):
            print('Iteration: %d Under: %s' % (i, str(under)))
            under = under/2
            if under == 0:
                output_file = open('under_int.txt','w')
                output_file.write("'Integer underflow' at iteration %d  with the number 2^-%d" % (i+1, i+1))
                output_file.close()
                print("'Integer underflow' at iteration %d  with the number 2^-%d" % (i+1, i+1))
                break
        print("\n\n\n'Underflow of integers' found \n\n\n")    
    elif program == 'F':
        #######################################
        #   Check over/underflow for floats   #
        #######################################
        under = 1.0
        # Under loop
        for i in range(100000):
            print('Iteration: %d Under: %s' % (i, str(under)))
            under = under/2
            if under == 0:
                output_file = open('under_float.txt','w')
                output_file.write("Float underflow at iteration %d  with the number 2^-%d" % (i+1, i+1))
                output_file.close()
                print("Float underflow at iteration %d  with the number 2^-%d" % (i+1, i+1))
                break
        
        # Wait for an input
        wait = input("Press enter to continue")
        
        over = 1.0
        # Over loop
        for i in range(10000):
            print('Iteration: %d Over: %s' % (i, str(over)))
            over = over*2
            if over == float('inf'):
                output_file = open('over_float.txt','w')
                output_file.write("Float overflow at iteration %d  with the number 2^%d" % (i+1, i+1))
                output_file.close()
                print("Float overflow at iteration %d  with the number 2^%d" % (i+1, i+1))
                break
        print("\n\n\nOver and underflow of floats found \n\n\n")
    elif program == 'P':
        ##################################
        #   Check precision for floats   #
        ##################################
        precision_float = float(1)
        for i in range(10000):
            tot = 1.0 + precision_float
            print(tot)
            if tot == 1:
                output_file = open('precision_float.txt','w')
                output_file.write("The machine precision for floats is "+ str(old_pf))
                output_file.close()
                print("The machine precision for floats is ", old_pf)
                break
            old_pf = precision_float # Store former value in case next value is loss of precision
            precision_float = precision_float/2
        # Wait for an input
        wait = input("Press enter to continue")

        ###########################################
        #   Check precision for complex numbers   #
        ###########################################
        precision_comp = 1j
        for i in range(10000):
            tot = 1+0j + precision_comp
            print(tot)
            if tot == 1+0j:
                output_file = open('precision_comp.txt','w')
                output_file.write("The machine precision for complex numbers is "+ str(old_pc))
                output_file.close()
                print("The machine precision for complex numbers is ", old_pc)
                break
            old_pc = precision_comp # Store former value in case next value is loss of precision
            precision_comp = precision_comp/2
        print("\n\n\nMachine precision for floats and complex numbers found \n\n\n")
    elif program == 'D':
        # Forward difference
        def FD(f, h, t):
            return ( f(t+h) - f(t) )/h

        # Central difference
        def CD(f, h, t):
            return ( f(t+h/2) - f(t-h/2) )/h

        # To repeat derivatives of e^t for different t
        def e_derivatives(t):
            h = 1

            h_array = np.array([])
            FD_array = np.array([])
            EFD_array = np.array([])
            CD_array = np.array([])
            ECD_array = np.array([])

            while h >= 2e-16:
                derivative_FD = FD(np.exp, h, t)
                derivative_CD = CD(np.exp, h, t)
                error_FD = np.abs((np.exp(t)-derivative_FD)/np.exp(t))
                error_CD = np.abs((np.exp(t)-derivative_CD)/np.exp(t))
                
                h_array = np.append(h_array, h)
                FD_array = np.append(FD_array, derivative_FD)
                CD_array = np.append(CD_array, derivative_CD)
                EFD_array = np.append(EFD_array, error_FD)
                ECD_array = np.append(ECD_array, error_CD)

                print('\nh = ' + str(h))
                print('Derivative (FD) = '+str(derivative_FD))
                print('Error (FD) = '+str(error_FD))
                print('Derivative (CD) = '+str(derivative_CD))
                print('Error (CD) = '+str(error_CD))

                h = h/2

            plt.plot(h_array, FD_array, label='Forward Difference Method')
            plt.plot(h_array, CD_array, label='Central Difference Method')
            plt.xlabel('h')
            plt.ylabel("Derivative")
            plt.xscale('log')
            plt.title("The graph of the derivative of $e^t$ against step size (h) for t = " + str(t))
            plt.legend()
            plt.savefig('derivative_e'+str(t)+'.png')
            plt.show()
            plt.close()

            plt.plot(h_array, EFD_array, label='Forward Difference Method Error')
            plt.plot(h_array, ECD_array, label='Central Difference Method Error')
            plt.xlabel('h')
            plt.ylabel('$\epsilon$')
            plt.title('The absolute relative error between the calculated derivative of $e^t$ \n and the analytic derivative ($\epsilon$) against step size (h) for t = '+str(t))
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('errors_derivatives_log_e'+str(t)+'.png')
            plt.show()
            plt.close()

        # To repeat derivatives of cos(t) for different t
        def cos_derivatives(t):
            h = 1

            h_array = np.array([])
            FD_array = np.array([])
            EFD_array = np.array([])
            CD_array = np.array([])
            ECD_array = np.array([])

            while h >= 2e-16:
                derivative_FD = FD(np.cos, h, t)
                derivative_CD = CD(np.cos, h, t)
                error_FD = np.abs((-np.sin(t)-derivative_FD)/-np.sin(t))
                error_CD = np.abs((-np.sin(t)-derivative_CD)/-np.sin(t))
                
                h_array = np.append(h_array, h)
                FD_array = np.append(FD_array, derivative_FD)
                CD_array = np.append(CD_array, derivative_CD)
                EFD_array = np.append(EFD_array, error_FD)
                ECD_array = np.append(ECD_array, error_CD)

                print('\nh = ' + str(h))
                print('Derivative (FD) = '+str(derivative_FD))
                print('Error (FD) = '+str(error_FD))
                print('Derivative (CD) = '+str(derivative_CD))
                print('Error (CD) = '+str(error_CD))

                h = h/2

            plt.plot(h_array, FD_array, label='Forward Difference Method')
            plt.plot(h_array, CD_array, label='Central Difference Method')
            plt.xlabel('h')
            plt.ylabel("Derivative")
            plt.xscale('log')
            plt.title("The graph of the derivative of $cos(t)$ against step size (h) for t = " + str(t))
            plt.legend()
            plt.savefig('derivative_cos'+str(t)+'.png')
            plt.show()
            plt.close()

            plt.plot(h_array, EFD_array, label='Forward Difference Method Error')
            plt.plot(h_array, ECD_array, label='Central Difference Method Error')
            plt.xlabel('h')
            plt.ylabel('$\epsilon$')
            plt.title('The absolute relative error between the calculated derivative of $cos(t)$ \n and the analytic derivative ($\epsilon$) against step size (h) for t = '+str(t))
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('errors_derivatives_log_cos'+str(t)+'.png')
            plt.show()
            plt.close()

        e_derivatives(0.1)
        e_derivatives(1)
        e_derivatives(100)

        cos_derivatives(0.1)
        cos_derivatives(1)
        cos_derivatives(100)

        print("\n\n\nDerivatives and their errors found for e^t and cos(t) for t=0.1, 1, 100 \n\n\n")
    else:
        # Leave space and start again.
        print('\n\n\n')
