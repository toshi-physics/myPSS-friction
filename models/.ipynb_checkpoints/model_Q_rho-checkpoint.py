import numpy as np
from tqdm import tqdm
import json, argparse, os

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *


def main():

    initParser = argparse.ArgumentParser(description='Run Model_Q_rho')
    initParser.add_argument('-j','--jsonfile', help='parameter file of type json', default='parameters.json')
    initargs = initParser.parse_args()  
    if os.path.isfile("%s" %(initargs.jsonfile)):
	    with open(initargs.jsonfile) as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
    Gamma0    = parameters["Gamma0"]   # rate of Q alignment with mol field H
    gamma     = parameters["gammaf"]   # traction coefficient
    alpha     = parameters["alpha"]    # active contractile stress
    chi       = parameters["chi"]      # coefficient of density gradients in Q's free energy
    D         = parameters["D"]        # Density diffusion coefficient in density dynamics
    #lambd     = parameters["lambda"]   # flow alignment parameter
    p0        = parameters["p0"]       # pressure when cells are close packed, should be very high
    #r_p       = parameters["r_p"]      # rate of pressure growth equal to rate of growth of cells
    Pii       = parameters["Pi"]       # strength of alignment
    rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
    rho_seed  = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
    rho_iso   = parameters["rhoisoend"] /rho_in   # jamming density
    rho_nem   = parameters["rhonemend"] /rho_in   # jamming density max for nematic substrate
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    pxx       = 1.0

    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    dn_dump   = round(n_steps / n_dump)

    savedir     = "data/model_Q_rho/gamma0_{:.2f}_rhoseed_{:.2f}_pi_{:.3f}/run_{}/".format(Gamma0, rho_seed, Pii, run)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=(1.0, 1.0)

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
    system.create_field('rho_end', k_list, k_grids, dynamic=False)
    system.create_field('pressure', k_list, k_grids, dynamic=False)
    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('S2', k_list, k_grids, dynamic=False)
    system.create_field('Gamma', k_list, k_grids, dynamic=False)
    system.create_field('rho_end_rho', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0, 0, 0, 0])
    # Define S2
    system.create_term("S2", [("Qxx", (np.square, None))], [1.0, 0, 0, 0, 0])
    system.create_term("S2", [("Qxy", (np.square, None))], [1.0, 0, 0, 0, 0])
    # Define RhoEnd
    system.create_term("rho_end", [("Ident", None)], [rho_iso, 0, 0, 0, 0])
    system.create_term("rho_end", [("S2", None)], [(rho_nem-rho_iso), 0, 0, 0, 0])
    # Define Pressure
    system.create_term("pressure", [("rho", (np.exp, None))], [p0, 0, 0, 0, 0])
    # Define rho minus rho_end
    system.create_term("rho_end_rho", [("rho_end", None)], [1, 0, 0, 0, 0])
    system.create_term("rho_end_rho", [("rho", None)], [-1, 0, 0, 0, 0])
    # Define Gamma_0
    system.create_term("Gamma", [("rho_end_rho", (np.tanh, None))], [Gamma0, 0, 0, 0, 0])
    # Define Hxx
    system.create_term("Hxx", [("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("S2", None), ("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxx", [("Ident", None)], [Pii*pxx, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None)], [-chi/2, 0, 2, 0, 0])
    system.create_term("Hxx", [("rho", None)], [chi/2, 0, 0, 2, 0])
    # Define Hxy
    system.create_term("Hxy", [("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("S2", None), ("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxy", [("rho", None)], [chi, 0, 1, 1, 0])

    # Create terms for rho timestepping
    system.create_term("rho", [("rho", None)], [-D, 1, 0, 0, 0])
    system.create_term("rho", [("rho", None)], [1, 0, 0, 0, 0])
    system.create_term("rho", [("rho", (np.power, 2)), ("rho_end", (np.power, -1))], [-1, 0, 0, 0, 0])
    system.create_term("rho", [("pressure", None)], [-1/gamma, 1, 0, 0, 0])
        # active terms now
    system.create_term("rho", [("Qxx", None)], [-alpha/gamma, 0, 2, 0, 0])
    system.create_term("rho", [("Qxx", None)], [+alpha/gamma, 0, 0, 2, 0])
    system.create_term("rho", [("Qxy", None)], [-2*alpha/gamma, 0, 1, 1, 0])

    # Create terms for Qxx timestepping
    system.create_term("Qxx", [("Gamma", None), ("Hxx", None)], [1, 0, 0, 0, 0])

    # Create terms for Qxy timestepping
    system.create_term("Qxy", [("Gamma", None), ("Hxy", None)], [1, 0, 0, 0, 0])


    rho     = system.get_field('rho')
    Qxx     = system.get_field('Qxx')
    Qxy     = system.get_field('Qxy')
    Gamma   = system.get_field('Gamma')
    pressure= system.get_field('pressure')

    # Initial Conditions for rho
    #cx, cy = 0.5 * grid_size; r = 0.2 * grid_size[0]
    #distance = np.sqrt((xv-cx)**2 + (yv-cy)**2)
    #rhoinit = np.where(distance<r, rho_seed, 0)
    # set init condition and synchronize momentum with the init condition, important!!
    set_rho_islands(rho, 100, rho_seed, grid_size, dr)
    #rho.set_real(np.abs(np.random.normal(rho_seed, rho_seed/1.2, size=grid_size)))
    #rho.synchronize_momentum()

    # Initialise Qxx and Qxy
    Qxx.set_real(0.1*(np.random.rand(mx, my)-0.5)+0.01)
    Qxx.synchronize_momentum()
    Qxy.set_real(0.1*(np.random.rand(mx, my)-0.5)+0.01)
    Qxy.synchronize_momentum()

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()
    # Initial Conditions for rho_end
    system.get_field('rho_end').set_real(rho_iso*np.ones(shape=grid_size))
    system.get_field('rho_end').synchronize_momentum()
    # Initial Conditions for rho_end_rho
    system.get_field('rho_end_rho').set_real(rho_iso*np.ones(shape=grid_size) - rho.get_real())
    system.get_field('rho_end_rho').synchronize_momentum()
    # Initialise Pressure
    pressure.set_real(p0*np.exp(rho.get_real()))
    pressure.synchronize_momentum()

    with open(savedir+'parameters.json', 'w') as f:
        json.dump(parameters, f)

    for t in tqdm(range(n_steps)):

        system.update_system(dt)

        if t % dn_dump == 0:
            np.savetxt(savedir+'rho.csv.'+ str(t), rho.get_real(), delimiter=',')
            np.savetxt(savedir+'Qxx.csv.'+ str(t), Qxx.get_real(), delimiter=',')
            np.savetxt(savedir+'Qxy.csv.'+ str(t), Qxy.get_real(), delimiter=',')
            np.savetxt(savedir+'Gamma.csv.'+ str(t), Gamma.get_real(), delimiter=',')
            np.savetxt(savedir+'pressure.csv.'+ str(t), pressure.get_real(), delimiter=',')

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

def set_rho_islands(rhofield, ncluster, rhoseed, grid_size, dr):

    ncluster = 50
    centers = grid_size[0]*np.random.rand(ncluster,2); radii = 0.1*np.random.rand(ncluster)*grid_size[0]
    mean = rhoseed; std = rhoseed/2

    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, dr[0])
    y   = np.arange(0+tol, grid_size[1]-tol, dr[1])
    r   = np.meshgrid(x,y)

    rhoinit = np.zeros(grid_size)
    for i in np.arange(ncluster):
        distance = np.sqrt((r[0]-centers[i,0])**2+(r[1]-centers[i,1])**2)
        rhoseeds = np.abs(np.random.normal(mean, std, size=np.shape(distance)))
        rhoinit += np.where(distance <= radii[i], rhoseeds*(radii[i]-distance)/radii[i], 1e-6)
    
    rhofield.set_real(rhoinit)
    rhofield.synchronize_momentum()


if __name__=="__main__":
    main()
