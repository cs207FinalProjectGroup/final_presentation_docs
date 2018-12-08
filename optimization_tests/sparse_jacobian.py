import numpy as np
import sys
import os
sys.path.append(os.path.join('..','..','cs207-FinalProject'))
import autodiff as ad

#function that takes in 1000 vals and returns a function that maps from R1000 -> R1000, so it returns a vector of 1000 values. This function should generate a sparse jacobian matrix to test newton's method on.
def sparse_jacob(x_vals):
    if len(x_vals) != 1000: #check to make sure we generate a large sparse matrix
        raise Exception('Please enter a vector of at least 1000 values or Autodiff objects');
        
    f_vals = [];
    for i, x_val in enumerate(x_vals):
        if i == 1: #second element is ln(x1)
            func = ad.ln(x_val);
            f_vals.append(func);
        elif i == 100: # x3 + 2*x100*exp(x100) + 4*x34
            func = x_vals[3] + 2*x_val*ad.exp(x_val) + 4*x_vals[34];
            f_vals.append(func);
        elif i == 445: #x445**2 - x23
            func = x_val**2 - x_vals[23];
            f_vals.append(func);
        elif i == 690: #x690 - x451 - x888
            func = x_val - x_vals[451] - x_vals[888];
            f_vals.append(func);
        elif i == 885: #x887 - x200
            func = x_val - x_vals[200];
            f_vals.append(func);
        elif i == 998: #x998 + x997 - x127
            func = x_val + x_vals[997] - x_vals[127];
            f_vals.append(func);
        else:
            f_vals.append(x_val);
            
    return np.array(f_vals);