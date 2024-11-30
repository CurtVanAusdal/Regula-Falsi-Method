import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

base = sys.float_info.radix  # base two
prec = sys.float_info.mant_dig  # precision
inf = math.inf
eps = sys.float_info.epsilon  # my Machine Epsilon

def ulps_diff(num1, num2):
    '''finds the difference in ulps between two numbers '''
    number1 = num1
    number2 = num2
    absolute_difference = abs(number1 - number2)
    epsilon = sys.float_info.epsilon
    difference_in_ulp = absolute_difference / epsilon
    #
    print("Difference in terms of ULPs:", difference_in_ulp)
    # returns no value, just prints the difference in ulps

def reg_fals(xLower, xUpper, func, es=eps, maxit=100000):
    '''Regula falsi method.'''
    if func(xLower) == 0 or func(xUpper) == 0:
        # Condition: If upper or lower bound is zero
        return 0, 0, 0, 0  # return zero for all values including flag
    
    if func(xLower) * func(xUpper) > 0: 
        # If two numbers are the same sign, it doesn't bracket a solution
        stringy = 'ERROR: Does not Bracket Solution, try different parameters for xl and xu'
        flag = -1  # flag goes to -1
        return stringy, flag
    
    flag = 0
    xvals = []
    for i in range(maxit): 
        # Condition: Go until 100000 iterations or less if loop breaks
        xnew = xLower - func(xLower) * ((xUpper - xLower) / (func(xUpper) - func(xLower)))
        fxnew = func(xnew)  # should get super close to zero
        xvals.append(xnew)
        ea = abs((xnew - xLower) / xnew)  # error approximate
        
        if ea < es:
            # CONDITION: If error is less than machine epsilon can handle, break the loop
            break
        elif func(xnew) <= 0:
            # if the new value is negative, new lower bound = new value
            xold = xLower
            xLower = xnew
            absoluteDiff = abs((xnew - xold))  # stops if ulps is less than 1
            if absoluteDiff / eps < 1:
                break
        elif func(xnew) >= 0:
            # if the new value is positive, new upper bound = new value
            xold = xUpper
            xUpper = xnew
            absoluteDiff = abs((xnew - xold))  # stop if ulps is less than 1
            if absoluteDiff / eps < 1:
                break
        
        if i == 100000 and (func(xnew) > 0.5 or func(xnew) < -0.5):
            # If iterations go all the way to maxit, set return values
            flag = -1
            xnew = xnew
            fxnew = fxnew
            ea = ea
            iterations = 100000
            xvals = None
            return xnew, flag, fxnew, iterations
    
    if flag is None: 
        # if flag made it through without any other assignment
        return 0  # (root found)
    
    flag = 0
    return xnew, flag, fxnew, i + 1  # (root, flag, function value, iterations)

def fun1(x):
    """function 1 handle"""
    result = x**4 - 6*x**3 + 12*x**2 - 10*x + 3
    return result

def fun2(x):
    """function 2 handle """
    result = x**3 - 7*x**2 + 15*x - 9
    return result

def plot_it(lb, up, fun, steps=100):
    '''plotting function for graphical method. '''
    x_vals = np.linspace(lb, up, steps)
    fx_vals = fun(x_vals)
    plt.plot(x_vals, fx_vals)
    plt.xlabel('x ')
    plt.ylabel('y')
    plt.title('Plot of the function f(x)')
    plt.grid(True)
    plt.show()

function1string = ' 0 = x^4 - 6*x^3 + 12*x^2 - 10*x +3'
function2string = '0 = x^3 - 7*x^2 + 15* x - 9'

fun1results = reg_fals(1.5, 2.5, fun1)
print(fun1results)
print('\n')

fun1results2 = reg_fals(0, 1.5, fun1)
print(fun1results2[0::])
print(f"Root is at {fun1results2[0]}")
print(f"Rounded root at {round(fun1results2[0], 2)}")
print('\n')

fun2results = reg_fals(1.5, 2.5, fun2)
print(fun1results)
print('\n')

fun2results2 = reg_fals(0, 1.5, fun2)
print(fun2results2[0::])
print(f"Root is at {fun2results2[0]}")
print(f"Rounded root at {round(fun2results2[0], 2)}")
