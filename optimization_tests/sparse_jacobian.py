import numpy as np
import sys
import os
sys.path.append(os.path.join('cs207-FinalProject'))
import autodiff as ad

#function that takes in 1000 vals and returns a function that maps from R1000 -> R1000, so it returns a vector of 1000 values. This function should generate a sparse jacobian matrix to test newton's method on.
def sparse_jacob(x_vals):
    if len(x_vals) != 1000: #check to make sure we generate a large sparse matrix
        raise Exception('Please enter a vector of at least 1000 values or Autodiff objects');
        
    f_vals = [];
    for i, x_val in enumerate(x_vals):
        if i == 1: #second element is log(x1)
            func = ad.log(x_val);
            f_vals.append(func);
        elif i == 100: # x3 + 2*x100*exp(x100) + 4*x34
            func = x_vals[3] + 2*x_val*ad.exp(x_val) + 4*x_vals[34];
            f_vals.append(func);
        elif i == 445: #x445**2 - x23
            func = x_val**2 - x_vals[23];
            f_vals.append(func);
        elif i == 690: #x690 * x450 - x888
            func = x_val * x_vals[450] - x_vals[888];
            f_vals.append(func);
        elif i == 887: #x887**-1 * x200/4
            func = x_val**-1 * x_vals[200]/4;
            f_vals.append(func);
        elif i == 247: #x247**3 * ad.sin(x211)
            func = x_val**3 * ad.sin(x_vals[211]);
            f_vals.append(func);
        elif i == 998: #x998 + x997 - x127 / (x348 + 2)
            func = x_val + x_vals[997] - x_vals[127] / (x_vals[348] + 2);
            f_vals.append(func);
        elif i == 798: #x798 / (x0 + x7) - ad.cos(x454) * x6
            func = x_val / (x_vals[0] + x_vals[7]) - ad.cos(x_vals[454]) * x_vals[6];
            f_vals.append(func)
        else:
            f_vals.append(x_val);
            
    return np.array(f_vals);