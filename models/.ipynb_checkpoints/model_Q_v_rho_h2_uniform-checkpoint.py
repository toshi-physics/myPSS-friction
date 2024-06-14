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
    D         = parameters["D"]        # Density diffusion coefficient in density dynamics
    lambd     = parameters["lambda"]   # flow alignment parameter
    p0        = parameters["p0"]       # pressure when cells are close packed, should be very high
    Pii       = parameters["Pi"]       # strength of alignment
    rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
    rho_seed  = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
    rho_iso   = parameters["rhoisoend"] /rho_in   # jamming density
    rho_nem   = parameters["rhonemend"] /rho_in   # jamming density max for nematic substrate
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])
    pxx       = 1.0

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
    system.create_field('rho_end', k_list, k_grids, dynamic=False)
    system.create_field('rho_end_rho', k_list, k_grids, dynamic=False)

    system.create_field('pressure', k_list, k_grids, dynamic=False)
    system.create_field('iqxp', k_list, k_grids, dynamic=False)
    system.create_field('iqyp', k_list, k_grids, dynamic=False)

    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('S2', k_list, k_grids, dynamic=False)

    system.create_field('Gamma', k_list, k_grids, dynamic=False)

    system.create_field('vx', k_list, k_grids, dynamic=False)
    system.create_field('vy', k_list, k_grids, dynamic=False)

    system.create_field('kappa_a_xy', k_list, k_grids, dynamic=False)
    system.create_field('kappa_s_xx', k_list, k_grids, dynamic=False)
    system.create_field('kappa_s_xy', k_list, k_grids, dynamic=False)

    system.create_field('iqxQxx', k_list, k_grids, dynamic=False)
    system.create_field('iqyQxx', k_list, k_grids, dynamic=False)
    system.create_field('iqxQxy', k_list, k_grids, dynamic=False)
    system.create_field('iqyQxy', k_list, k_grids, dynamic=False)

    system.create_field('charge', k_list, k_grids, dynamic=False)
    system.create_field('curldivQ', k_list, k_grids, dynamic=False)
    
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
    system.create_term("iqxp", [("pressure", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyp", [("pressure", None)], [1, 0, 0, 1, 0])
    # Define rho minus rho_end
    system.create_term("rho_end_rho", [("rho_end", None)], [1, 0, 0, 0, 0])
    system.create_term("rho_end_rho", [("rho", None)], [-1, 0, 0, 0, 0])
    # Define Gamma
    system.create_term("Gamma", [("rho_end_rho", (np.tanh, None))], [Gamma0, 0, 0, 0, 0])
    # Define Hxx
    system.create_term("Hxx", [("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("S2", None), ("Qxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None), ("Qxx", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxx", [("rho", None)], [Pii*pxx, 0, 0, 0, 0])
    system.create_term("Hxx", [("rho", None)], [chi/2, 0, 2, 0, 0])
    system.create_term("Hxx", [("rho", None)], [-chi/2, 0, 0, 2, 0])
    # Define Hxy
    system.create_term("Hxy", [("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("S2", None), ("Qxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Hxy", [("rho", None), ("Qxy", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxy", [("rho", None)], [chi, 0, 1, 1, 0])
    # Define iqxQxx and so on
    system.create_term("iqxQxx", [("Qxx", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyQxx", [("Qxx", None)], [1, 0, 0, 1, 0])
    system.create_term("iqxQxy", [("Qxy", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyQxy", [("Qxy", None)], [1, 0, 0, 1, 0])
    # Define vx
    system.create_term("vx", [('iqxQxx', None), ("rho", (np.power, -1))], [alpha/gammaxx, 0, 0, 0, 0])
    system.create_term("vx", [('iqyQxy', None), ("rho", (np.power, -1))], [alpha/gammaxx, 0, 0, 0, 0])
    system.create_term("vx", [('iqxp', None), ("rho", (np.power, -1))], [-1/gammaxx, 0, 0, 0, 0])
    # Define vy
    system.create_term("vy", [('iqxQxy', None), ("rho", (np.power, -1))], [alpha/gammayy, 0, 0, 0, 0])
    system.create_term("vy", [('iqyQxx', None), ("rho", (np.power, -1))], [-alpha/gammayy, 0, 0, 0, 0])
    system.create_term("vy", [('iqyp', None), ("rho", (np.power, -1))], [-1/gammayy, 0, 0, 0, 0])
    # Define kappa_a_xy
    system.create_term("kappa_a_xy", [("vx", None)], [0.5, 0, 0, 1, 0]) # iqy vx / 2
    system.create_term("kappa_a_xy", [("vy", None)], [-0.5, 0, 1, 0, 0]) # -iqx vy / 2
    # Define kappa_s_xx
    system.create_term("kappa_s_xx", [("vx", None)], [0.5, 0, 1, 0, 0])
    system.create_term("kappa_s_xx", [("vy", None)], [-0.5, 0, 0, 1, 0])
    # Define kappa_s_xy
    system.create_term("kappa_s_xy", [("vx", None)], [0.5, 0, 0, 1, 0])
    system.create_term("kappa_s_xy", [("vy", None)], [0.5, 0, 1, 0, 0])
    # Define nematic charge density
    system.create_term('charge', [('iqxQxx', None), ('iqyQxy', None)], [1, 0, 0, 0, 0])
    system.create_term('charge', [('iqyQxx', None), ('iqxQxy', None)], [-1, 0, 0, 0, 0])
    # Define curl of div of Q
    system.create_term('curldivQ', [('Qxy', None)], [1, 0, 2, 0, 0])
    system.create_term('curldivQ', [('Qxy', None)], [-1, 0, 0, 2, 0])
    system.create_term('curldivQ', [('Qxx', None)], [-2, 0, 1, 1, 0])

    # Create terms for rho timestepping
    system.create_term("rho", [("rho", None)], [-D, 1, 0, 0, 0])
    system.create_term("rho", [("rho", None), ("rho_end", (np.power, -1)), ("rho_end_rho", None), ("rho_end_rho", (np.heaviside, 0))], [1, 0, 0, 0, 0])
    system.create_term("rho", [("pressure", None)], [1/gammaxx, 0, 2, 0, 0])
    system.create_term("rho", [("pressure", None)], [1/gammayy, 0, 0, 2, 0])
        # active terms now
    system.create_term("rho", [("Qxx", None)], [-alpha/gammaxx, 0, 2, 0, 0])
    system.create_term("rho", [("Qxx", None)], [+alpha/gammayy, 0, 0, 2, 0])
    system.create_term("rho", [("Qxy", None)], [-alpha*((1/gammaxx) + (1/gammayy)), 0, 1, 1, 0])
    system.create_term("rho", [("Qxx", None)], [Gamma0*Pii, 1, 0, 0, 0])

    # Create terms for Qxx timestepping
    system.create_term("Qxx", [("Gamma", None), ("Gamma", (np.heaviside, 0)), ("Hxx", None)], [1, 0, 0, 0, 0])
    system.create_term("Qxx", [("vx", None), ("iqxQxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Qxx", [("vy", None), ("iqyQxx", None)], [-1, 0, 0, 0, 0])
    system.create_term("Qxx", [("Qxy", None), ("kappa_a_xy", None)], [2, 0, 0, 0, 0])
    system.create_term("Qxx", [("kappa_s_xx", None)], [lambd, 0, 0, 0, 0])

    # Create terms for Qxy timestepping
    system.create_term("Qxy", [("Gamma", None), ("Gamma", (np.heaviside, 0)), ("Hxy", None)], [1, 0, 0, 0, 0])
    system.create_term("Qxy", [("vx", None), ("iqxQxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Qxy", [("vy", None), ("iqyQxy", None)], [-1, 0, 0, 0, 0])
    system.create_term("Qxy", [("Qxx", None), ("kappa_a_xy", None)], [-2, 0, 0, 0, 0])
    system.create_term("Qxy", [("kappa_s_xy", None)], [lambd, 0, 0, 0, 0])

    rho     = system.get_field('rho')
    Qxx     = system.get_field('Qxx')
    Qxy     = system.get_field('Qxy')
    vx      = system.get_field('vx')
    vy      = system.get_field('vy')
    #Gamma   = system.get_field('Gamma')
    pressure= system.get_field('pressure')
    charge  = system.get_field('charge')
    curldivQ= system.get_field('curldivQ')

    # set init condition and synchronize momentum with the init condition, important!!
    #set_rho_islands(rho, int(100*rho_seed/0.16), rho_seed, grid_size)
    rhoseed = np.random.rand(mx, my) + np.ones([mx, my])
    rhoseed = rhoseed*rho_seed/np.average(rhoseed)
    rho.set_real(rhoseed)
    rho.synchronize_momentum()

    # Initialise Qxx and Qxy
    itheta = 2 * np.pi * np.random.rand(mx, my)
    iS = 0.1*np.random.rand(mx, my)
    Qxx.set_real(iS*(np.cos(itheta)))
    Qxx.synchronize_momentum()
    Qxy.set_real(iS*(np.sin(itheta)))
    Qxy.synchronize_momentum()
    #set_aster(Qxx, Qxy, grid_size, dr)

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

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):

        system.update_system(dt)

        if t % dn_dump == 0:
            np.savetxt(savedir+'/data/'+'rho.csv.'+ str(t//dn_dump), rho.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxx.csv.'+ str(t//dn_dump), Qxx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxy.csv.'+ str(t//dn_dump), Qxy.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'vx.csv.'+ str(t//dn_dump), vx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'vy.csv.'+ str(t//dn_dump), vy.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'curldivQ.csv.'+ str(t//dn_dump), curldivQ.get_real(), delimiter=',')

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
