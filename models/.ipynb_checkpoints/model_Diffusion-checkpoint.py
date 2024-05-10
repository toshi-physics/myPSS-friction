import numpy as np
from tqdm import tqdm
import os

from src.functions import logistic
from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *


def main():

    model = "diffusion"

     # Define the grid size.
    grid_size = (100, 100)
    dr = (1.0, 1.0)
    tol = 0.001
    x   = np.linspace(0+tol, 1-tol, grid_size[0])
    y   = np.linspace(0+tol, 1-tol, grid_size[1])
    xv, yv  = np.meshgrid(x,y)

    D = 1
    rho_end = 5
    dt_rho = 10

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields
    system.create_field('rho', k_list, k_grids, dynamic=True)

    # Initial Conditions
    cx = 0.5; cy = 0.5; r = 0.2
    distance = np.sqrt((xv-cx)**2 + (yv-cy)**2)
    rhoinit = np.where(distance<r, 0.1, 0)
    #set init condition and synchronize momentum with the init condition, important!!
    system.get_field('rho').set_real(rhoinit)
    system.get_field('rho').synchronize_momentum()

    #create equation, if no function of rho, write None. If function and argument, supply as a tuple. 
    #Write your own functions in the function library or use numpy functions
    #if using functions that need no args, like np.tanh, write (np.tanh, None).
    system.create_term("rho", [("rho", None)], [-D, 1, 0, 0, 0])
    system.create_term("rho", [("rho", (logistic, rho_end))], [1/dt_rho, 0, 0, 0, 0])
    #system.create_term("rho", [("rho", (np.tanh, None))], [1/dt_rho, 0, 0, 0, 0])
    
    rho = system.get_field("rho")
    savedir = 'data/'+model
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #save init condition
    np.savetxt(savedir+'/rho.csv.'+ str(0), rho.get_real(), delimiter=',')
    for t in tqdm(range(4000)):

        system.update_system(0.1)

        if (t+1)%100 == 0:
            np.savetxt(savedir+'/rho.csv.'+ str(t+1), rho.get_real(), delimiter=',')


def momentum_grids(grid_size, dr):
    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.

    k_grids = np.meshgrid(*k_list, indexing='ij')
    #k_grids = np.meshgrid(*k_list, indexing='ij', sparse=True)
    # k_grids is now a list of 2D sparse arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)
    #k_squared = k_grids[0]**2 + k_grids[1]**2
    k_abs = np.sqrt(k_squared)
    inv_kAbs = np.divide(1.0, k_abs, where=k_abs!=0)

    k_power_arrays = [k_squared, 1j*k_grids[0], 1j*k_grids[1], inv_kAbs]

    return k_power_arrays


if __name__=="__main__":
    main()
