import numpy as np
from tqdm import tqdm
import json, argparse, os

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *


def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/parameters.json"):
	    with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
    Gamma0    = parameters["Gamma0"]   # rate of Q alignment with mol field H
    gammaxx   = parameters["gammaxx"]
    gammayy   = parameters["gammayy"]
    alpha     = parameters["alpha"]    # active contractile stress
    chi       = parameters["chi"]      # coefficient of density gradients in Q's free energy
    a         = parameters["a"]
    d         = parameters["d"]
    lambd     = parameters["lambda"]   # flow alignment parameter
    p0        = parameters["p0"]       # pressure when cells are close packed, should be very high
    rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
    rho_c     = parameters["rho_c"] /rho_in
    rho_m     = 0.05 #minimum density to be reached while phase separating
    rho_seed  = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
    rho_iso   = parameters["rhoisoend"] /rho_in   # jamming density
    rho_nem   = parameters["rhonemend"] /rho_in   # jamming density max for nematic substrate
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])

    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    dn_dump   = round(n_steps / n_dump)
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=np.array([dx, dy])

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields that undergo timestepping
    system.create_field('rho', k_list, k_grids, dynamic=True)
    system.create_field('Qxx', k_list, k_grids, dynamic=True)
    system.create_field('Qxy', k_list, k_grids, dynamic=True)
    # Create fields that don't timestep
    system.create_field('Hxx', k_list, k_grids, dynamic=False)
    system.create_field('Hxy', k_list, k_grids, dynamic=False)

    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('S2', k_list, k_grids, dynamic=False)

    system.create_field('iqxQxx', k_list, k_grids, dynamic=False)
    system.create_field('iqyQxx', k_list, k_grids, dynamic=False)
    system.create_field('iqxQxy', k_list, k_grids, dynamic=False)
    system.create_field('iqyQxy', k_list, k_grids, dynamic=False)
    system.create_field('q2Qxx', k_list, k_grids, dynamic=False)
    system.create_field('q2Qxy', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0, 0, 0, 0])
    # Define S2
    system.create_term("S2", [("Qxx", (np.square, None))], [1.0, 0, 0, 0, 0])
    system.create_term("S2", [("Qxy", (np.square, None))], [1.0, 0, 0, 0, 0])
    # Define iqxQxx and so on
    system.create_term("iqxQxx", [("Qxx", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyQxx", [("Qxx", None)], [1, 0, 0, 1, 0])
    system.create_term("iqxQxy", [("Qxy", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyQxy", [("Qxy", None)], [1, 0, 0, 1, 0])
    system.create_term("q2Qxx", [("Qxx", None)], [1, 1, 0, 0, 0])
    system.create_term("q2Qxy", [("Qxy", None)], [1, 1, 0, 0, 0])
    # Define Hxx
    system.create_term("Hxx", [("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("S2", None), ("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("q2Qxx", None)], [-K, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None)], [chi/2, 0, 2, 0, 0])
    system.create_term("Hxx", [("rho", None)], [-chi/2, 0, 0, 2, 0])
    # Define Hxy
    system.create_term("Hxy", [("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("S2", None), ("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("q2Qxy", None)], [-K, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None)], [chi, 0, 1, 1, 0])

    # Create terms for rho timestepping

    # Create terms for Qxx timestepping
    system.create_term("Qxx", [("Hxx", None)], [Gamma0, 0, 0, 0, 0])

    # Create terms for Qxy timestepping
    system.create_term("Qxx", [("Hxy", None)], [Gamma0, 0, 0, 0, 0])

    rho     = system.get_field('rho')
    Qxx     = system.get_field('Qxx')
    Qxy     = system.get_field('Qxy')

    # set init condition and synchronize momentum with the init condition, important!!
    #set_rho_islands(rho, int(100*rho_seed/0.16), rho_seed, grid_size)
    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, 1)
    y   = np.arange(0+tol, grid_size[1]-tol, 1)
    r   = np.meshgrid(x,y)
    rhoinit = np.zeros(grid_size)
    distance = np.sqrt((r[0]-50)**2+(r[1]-50)**2)
    rhoseeds = np.ones([mx,my])
    rhoinit = np.where(distance<25, rho_seed*(30-distance)/30, 0.1)
    meanrho = np.average(rhoinit)
    rhoinit = rhoinit * rho_seed / meanrho
    rho.set_real(rhoinit)
    rho.synchronize_momentum()

    # Initialise Qxx and Qxy
    itheta = 2 * np.pi * np.random.rand(mx, my)
    iS = 0.1*np.random.rand(mx, my)
    Qxx.set_real(iS*(np.cos(itheta)))
    Qxx.synchronize_momentum()
    Qxy.set_real(iS*(np.sin(itheta)))
    Qxy.synchronize_momentum()

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):

        system.update_system(dt)

        if t % dn_dump == 0:
            np.savetxt(savedir+'/data/'+'rho.csv.'+ str(t//dn_dump), rho.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxx.csv.'+ str(t//dn_dump), Qxx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxy.csv.'+ str(t//dn_dump), Qxy.get_real(), delimiter=',')

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

def set_rho_islands(rhofield, ncluster, rhoseed, grid_size):
    centers = grid_size[0]*np.random.rand(ncluster,2); radii = 0.1*np.random.rand(ncluster)*grid_size[0]
    mean = rhoseed; std = rhoseed/2

    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, 1)
    y   = np.arange(0+tol, grid_size[1]-tol, 1)
    r   = np.meshgrid(x,y)

    rhoinit = np.zeros(grid_size)
    for i in np.arange(ncluster):
        distance = np.sqrt((r[0]-centers[i,0])**2+(r[1]-centers[i,1])**2)
        rhoseeds = np.abs(np.random.normal(mean, std, size=np.shape(distance)))
        rhoinit += np.where(distance < radii[i], rhoseeds*(radii[i]-distance)/radii[i], 1e-3)

    meanrho = np.average(rhoinit)

    rhoinit = rhoinit * rhoseed / meanrho

    print("Average rho at start is", np.average(rhoinit), " for rhoseed=", rhoseed)
    
    rhofield.set_real(rhoinit)
    rhofield.synchronize_momentum()


if __name__=="__main__":
    main()
