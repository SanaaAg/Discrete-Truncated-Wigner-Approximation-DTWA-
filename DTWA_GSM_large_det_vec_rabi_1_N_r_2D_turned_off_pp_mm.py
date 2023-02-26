#import libraries


import scipy
from scipy.integrate import solve_ivp
import numpy as np
import cmath as cm
import h5py
from numpy.linalg import multi_dot
from scipy.linalg import logm
from scipy.special import factorial
from scipy.special import *
from scipy import sparse
from scipy.sparse import csr_matrix
from numpy.linalg import eig
from scipy.linalg import eig as sceig
import math
import time
from sympy.physics.quantum.cg import CG
from sympy import S
import collections
import numpy.polynomial.polynomial as poly
#from numpy.random import default_rng
import sys
argv=sys.argv

if len(argv) < 2:
    print ("Input error")
else:
    try:
        run_id = int(argv[3])
        Natoms = int(argv[2])
        n_traj = int(argv[4])
        initial_real_val = (int(argv[1])-1)*n_traj     
    except:
        print ("Input error")

# some definitions (do not change!)

fe = 1/2
fg = 1/2

# some definitions (do not change!)

e0 = np.array([0, 0, 1])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
eplus = -(ex + 1j*ey)/np.sqrt(2)
eminus = (ex - 1j*ey)/np.sqrt(2)
single_decay = 1.0 # single atom decay rate

#direc = '/Users/saag4275/Documents/Rey/projects/multilevel-entangled/calculations/python/data_files/' # directory for saving data on local macbook

direc = '/data/rey/saag4275/data_files/'   # directory for saving data on the cluster

# initial condition sampling

IC_y = np.zeros((n_traj, Natoms), complex)
IC_z = np.zeros((n_traj, Natoms), complex)

arr_index_to_s = np.zeros((4,2), complex)
arr_index_to_s[0] = np.array([-1,-1])
arr_index_to_s[1] = np.array([-1,1]) 
arr_index_to_s[2] = np.array([1,-1]) 
arr_index_to_s[3] = np.array([1,1]) 


ind = 0
for real_val in range(initial_real_val, n_traj+initial_real_val):

    rng = np.random.default_rng(real_val)
        
    #np.random.seed(real_val)

    # sx = +1 is fixed so we sample for sy and sz

    #IC_y[ind,:] = rng.randint(low=0, high=2, size=Natoms) # 0 is sy = -1 and 1 is sy = +1
    #IC_z[ind,:] = rng.randint(low=0, high=2, size=Natoms) # 0 is sz = -1 and 1 is sz = +1
    
    random_index_list = rng.integers(low=0, high=4, size=Natoms) #np.random.randint(low=0, high=4, size=Natoms)
    IC_y[ind,:] = arr_index_to_s[random_index_list,0]  #np.random.randint(low=0, high=2, size=Natoms) # 0 is sy = -1 and 1 is sy = +1
    IC_z[ind,:] = arr_index_to_s[random_index_list,1]  #np.random.randint(low=0, high=2, size=Natoms) # 0 is sz = -1 and 1 is sz = +1
    
    ind += 1

# parameter setting box (may change)

ratio = np.round(0.1*run_id,2)
rabi = 1.0*single_decay
t_final_input = 1.0*(run_id**2)
det_val_input = 1.0

print("Detuning = "+str(det_val_input), flush=True)
print("Rabi = "+str(rabi), flush=True)
print("t_final = "+str(t_final_input), flush=True)
print("atomic spacing = "+str(ratio), flush=True)

eL = np.array([1, 0, 0]) # polarisation of laser, can be expressed in terms of the vectors defined above
e0_desired = eL

IC_chosen = 'equal_gs'
print("IC = "+IC_chosen, flush=True)

#interactions turned off
turn_off_list = ['incoherent','coherent']
turn_off = ['coherent_pp'] #[turn_off_list[0], turn_off_list[1]] # leave turn_off = [], if nothing is to be turned off


turn_off_txt = ''
if turn_off != []:
    turn_off_txt += '_no_int_'
    for item in turn_off:
        turn_off_txt += '_'+ item

add_txt_in_params = turn_off_txt

if IC_chosen!='':
    add_txt_in_params = '_'+IC_chosen + '_IC'
else:
    add_txt_in_params = ''

add_txt_in_params += '_real_id_' + str(initial_real_val) + '_to_' + str(initial_real_val+n_traj)

add_txt_in_params += turn_off_txt

num_pts_dr = int(2*1e2)

t_initial_dr = 0.0
t_final_dr = t_final_input 
t_range_dr = [t_initial_dr, t_final_dr]
t_vals_dr = np.concatenate((np.arange(0,10,1)/100.0,np.logspace(np.log10(.1),np.log10(t_final_dr),num_pts_dr-10)))
t_vals_dr[-1] = t_final_dr

# more definitions and functions (do not change!)

wavelength = 1 # wavelength of incident laser
k0 = 2*np.pi/wavelength
kvec = k0*np.array([0, 1, 0]) # k vector of incident laser
r_axis = np.array([1, 0, 0]) # orientation of the distance between atoms, 3 vector
r_axis = r_axis/np.linalg.norm(r_axis)
ratio = ratio*wavelength

#################################################################################################

# more definitions and functions (do not change!)

def rotation_matrix_a_to_b(va, vb): #only works upto 1e15-ish precision
    ua = va/np.linalg.norm(va)
    ub = vb/np.linalg.norm(vb)
    if np.dot(ua, ub) == 1:
        return np.identity(3)
    elif np.dot(ua, ub) == -1: #changing z->-z changes y->-y, thus preserving x->x, which is the array direction (doesn't really matter though!)
        return -np.identity(3)
    uv = np.cross(ua,ub)
    c = np.dot(ua,ub)
    v_mat = np.zeros((3,3))
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    v_mat[:,0] = np.cross(uv, ux)
    v_mat[:,1] = np.cross(uv, uy)
    v_mat[:,2] = np.cross(uv, uz)
    matrix = np.identity(3) + v_mat + (v_mat@v_mat)*1.0/(1.0+c)
    return matrix

 
if np.abs(np.conj(e0)@e0_desired) < 1.0:
    rmat = rotation_matrix_a_to_b(e0,e0_desired)
    eplus = rmat@eplus
    eminus = rmat@eminus
    ex = rmat@ex
    ey = rmat@ey
    e0 = e0_desired

print('kL = '+str(kvec/np.linalg.norm(kvec)))
print('e0 = '+str(e0))
print('ex = '+str(ex))
print('ey = '+str(ey))

Natoms_along_x_axis = int(np.sqrt(Natoms))
rvecall = np.zeros((Natoms, 3)) # position of each atom
x_id = 1
z_id = 0
for ind in range(1, Natoms): # positions of atoms if there are more than 1
    if ind%Natoms_along_x_axis == 0:
        x_id = 0       
        z_id += 1
    rvecall[ind,0] = ratio*x_id
    rvecall[ind,2] = ratio*z_id
    x_id += 1
    
print('Atomic positions: ', flush=True)
print(rvecall)
    
    
HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**Natoms) # size of total Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb

# levels
deg_e = int(2*fe + 1)
deg_g = int(2*fg + 1)

if (deg_e == 1 and deg_g == 1):
    qmax = 0
else:
    qmax = 1

    
###########################################################################

# dictionaries

# Clebsch Gordan coeff
cnq = {}
arrcnq = np.zeros((deg_g, 2*qmax+1), complex)
if (deg_e == 1 and deg_g ==1):
    cnq[0, 0] = 1
    arrcnq[0, 0] =  1
else:
    for i in range(0, deg_g):
        mg = i-fg
        for q in range(-qmax, qmax+1):
            if np.abs(mg + q) <= fe:
                cnq[mg, q] =  float(CG(S(fg), S(mg), S(qmax), S(q), S(fe), S(mg+q)).doit())
                arrcnq[i, q+qmax] = cnq[mg, q]
cnq = collections.defaultdict(lambda : 0, cnq) 

# Dipole moment
dsph = {}
if (deg_e == 1 and deg_g ==1):
    dsph[0, 0] = np.conjugate(evec[0])
else:
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            dsph[me, mg] = (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

dsph = collections.defaultdict(lambda : np.array([0,0,0]), dsph) 

# normalise vector
def hat_op(v):
    return (v/np.linalg.norm(v))


# plot properties

levels = int(deg_e + deg_g)

rdir_fig = '_r_'+str(np.round(ratio,2)).replace('.',',')+'_along_xz_2D'

eLdir_fig = '_eL_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if eL[i]!=0:
        if temp_add == 0:
            eLdir_fig += dirs[i]
        else:
            eLdir_fig += '_and_'+ dirs[i]
        temp_add += 1

kLdir_fig = '_k_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if kvec[i]!=0:
        if temp_add == 0:
            kLdir_fig += dirs[i]
        else:           
            kLdir_fig += '_and_'+ dirs[i]
        temp_add += 1


rabi_add = '_rabi_'+str(rabi).replace('.',',')

det_fig = '_det_'+str(det_val_input).replace('.',',')  + '_tfin_'+str(np.round(t_final_dr,2)).replace('.',',')


h5_title = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kLdir_fig+eLdir_fig+rabi_add+det_fig+add_txt_in_params+'.h5'


# Green's function
def funcG(r):
    tempcoef = 3*single_decay/4.0
    temp1 = (np.identity(3) - np.outer(hat_op(r), hat_op(r)))*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r)) 
    temp2 = (np.identity(3) - 3*np.outer(hat_op(r), hat_op(r)))*((1j*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**2) - np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**3)
    return (tempcoef*(temp1 + temp2))

def funcGij(i, j):
    return (funcG(rvecall[i] - rvecall[j]))

fac_inc = 1.0
fac_coh = 1.0
if turn_off!=[]:
    for item in turn_off:
        if item == 'incoherent':
            fac_inc = 0
        if item == 'coherent':
            fac_coh = 0


taD = time.time()


fac_coh_pp = 1.0
if turn_off!=[]:
    for item in turn_off:
        if item == 'coherent_pp':
            fac_coh_pp = 0

#arrIij = np.zeros((2, Natoms, Natoms), complex)
arrRij = np.zeros((3, Natoms, Natoms), complex)

for i in range(0, Natoms):
    for j in range(0, Natoms):
        if i==j:
            continue
        #arrIij[0,i,j] = fac_inc*np.conjugate(evec[1])@np.imag(funcGij(i, j))@evec[1]
        #arrIij[1,i,j] = fac_inc*np.conjugate(evec[1])@np.imag(funcGij(i, j))@evec[-1]
        arrRij[0,i,j] = fac_coh*np.conjugate(evec[1])@np.real(funcGij(i, j))@evec[1]
        arrRij[1,i,j] = fac_coh_pp*fac_coh*np.conjugate(evec[1])@np.real(funcGij(i, j))@evec[-1]
        arrRij[2,i,j] = fac_coh_pp*fac_coh*np.conjugate(evec[-1])@np.real(funcGij(i, j))@evec[1]

        

tbD = time.time()
print("time to assign Rij, Iij arrays: "+str(tbD-taD), flush=True)

if fac_coh != 1.0 and fac_inc == 1.0:
    print("Coherent interactions turned off!", flush=True)
elif fac_coh == 1.0 and fac_inc != 1.0:
    print("Incoherent interactions turned off!", flush=True)
elif fac_coh != 1.0 and fac_inc != 1.0:
    print("Coherent AND incoherent interactions turned off!", flush=True)
elif fac_coh_pp != 1.0:
    print("Coherent pp and mm interactions turned off!", flush=True)


#############################################################################################################

# don't change


# single atom operators' index + correlations' index -- in the list of all ops for all atoms

dict_ops = {}
index = 0
for n in range(0, Natoms):

    dict_ops['z', n] = index
    index += 1
    
for n in range(0, Natoms):

    dict_ops['p', n] = index
    index += 1
        
dict_ops = collections.defaultdict(lambda : 'None', dict_ops)

####################################################################################################

# (1-identity) array to prevent counting of i = j terms (remove self-interaction terms)

delta_ij = 1 - np.identity(Natoms)

#EOM for one point functions

const_fac = (2/9.0)*((rabi/det_val_input)**2)

# arrRij[0] = Rij_{1,1}, arrRij[1] = Rij_{1,-1}, arrRij[2] = Rij_{-1,1} 

def f_sig_z_dot(sig_z, sig_p): #n = 0, 1, 2, ... = atom no., drive = 0, 1
    #tempsum = -2*sig_z*single_decay    
    tempsum =  - (-4*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[1],sig_p,sig_p)) + (-4*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[2],np.conj(sig_p),np.conj(sig_p)))
    tempsum +=  (4*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[0],sig_p,np.conj(sig_p))) + (-4*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[0],np.conj(sig_p),sig_p))
    return const_fac*tempsum

def f_sig_p_dot(sig_z, sig_p): #n = 0, 1, 2, ... = atom no.
    #tempsum = -sig_p*single_decay     
    tempsum = (2*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[2],sig_z,np.conj(sig_p)))
    tempsum += (2*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrRij[0],sig_z,sig_p))
    
    return const_fac*tempsum

##########################################################################################

#final EOM based on which terms are two point and which are one point correlators

def f_sig_dot_vec(t, sig_list):
    
    sig_z = sig_list[:Natoms] 
    sig_p = sig_list[Natoms:int(2*Natoms)] 

    sig_z_dot = (f_sig_z_dot(sig_z, sig_p)) #.flatten()
    sig_p_dot = (f_sig_p_dot(sig_z, sig_p)) #.flatten()

    sig_dot_mat = np.concatenate((sig_z_dot,sig_p_dot))
    return sig_dot_mat.flatten()

'''
def f_reached_equilibrium(t, y, drive, detuning):
    return np.linalg.norm(f_sig_dot_vec(t, y, drive, detuning)) - tol_params_equil

f_reached_equilibrium.terminal = True
f_reached_equilibrium.direction = -1
'''

###################################################################################

# initial condition

num_single_particle_ops = int(Natoms*2) # sig_z, sig_p

# no. of two pt ops

total_num = num_single_particle_ops

#initialising system in which all atoms are in the superposition of up and down states

initial_sig_vec = np.zeros((n_traj, total_num),complex)


for n1 in range(0, Natoms):
    initial_sig_vec[:, dict_ops['p',n1]] = 0.5*(np.ones(n_traj) + IC_y[:,n1]*1j) # sigma_p = 0.5*(sx + i sy) for each atom
    initial_sig_vec[:, dict_ops['z',n1]] = IC_z[:,n1]

    
initial_sig_vec = initial_sig_vec +0*1j

########################################################################

#driven evolution from a ground state to get to the steady state

ta1 = time.time()
#sol_arr = np.zeros((n_traj, Natoms*2, len(t_vals_dr)), complex)
sz_dr = np.zeros((n_traj, Natoms, len(t_vals_dr)))
sx_dr = np.zeros((n_traj, Natoms, len(t_vals_dr)))
sy_dr = np.zeros((n_traj, Natoms, len(t_vals_dr)))
for id_traj in range(0, n_traj):
    sol = solve_ivp(f_sig_dot_vec, t_range_dr, initial_sig_vec[id_traj,:], method='RK45', t_eval=t_vals_dr, dense_output=False, events=None, atol = 1e-6, rtol = 1e-5)
    #sol_arr[id_traj,:,:] = sol.y[:,:]

    sz_dr[id_traj,:,:] = sol.y[:Natoms,:].real
    sx_dr[id_traj,:,:] = 2*sol.y[Natoms:int(2*Natoms),:].real
    sy_dr[id_traj,:,:] = 2*sol.y[Natoms:int(2*Natoms),:].imag
tb1 = time.time()
runtime1 = tb1-ta1

print('runtime DTWA trajectories equilibrate = '+str(runtime1),flush=True)

total_Sz_dr = np.einsum('njt->t', sz_dr)/n_traj
total_Sx_dr = np.einsum('njt->t', sx_dr)/n_traj
total_Sy_dr = np.einsum('njt->t', sy_dr)/n_traj

delta_ij = 1 - np.identity(Natoms)

total_Sz_Sz_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sz_dr, sz_dr))/n_traj + Natoms
total_Sz_Sx_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sz_dr, sx_dr))/n_traj + 1j*total_Sy_dr
total_Sz_Sy_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sz_dr, sy_dr))/n_traj - 1j*total_Sx_dr

total_Sx_Sx_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sx_dr, sx_dr))/n_traj + Natoms
total_Sx_Sz_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sx_dr, sz_dr))/n_traj - 1j*total_Sy_dr
total_Sx_Sy_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sx_dr, sy_dr))/n_traj + 1j*total_Sz_dr

total_Sy_Sy_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sy_dr, sy_dr))/n_traj + Natoms
total_Sy_Sz_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sy_dr, sz_dr))/n_traj + 1j*total_Sx_dr
total_Sy_Sx_dr = np.einsum('nt->t', np.einsum('jk,njt,nkt->nt', delta_ij, sy_dr, sx_dr))/n_traj - 1j*total_Sz_dr

#'''
spin_inequal_param_list_30a = np.zeros(len(t_vals_dr)) # if any of these are < 1, then that inequality is violated
spin_inequal_param_list_30b = np.zeros(len(t_vals_dr))
spin_inequal_param_list_30c = np.zeros(len(t_vals_dr))
spin_inequal_param_list_30d = np.zeros(len(t_vals_dr))

for i in range(0, len(t_vals_dr)): 
    # calculate spin squeezing inequalities

    C_matrix = np.array([[total_Sx_Sx_dr[i], 0.5*(total_Sx_Sy_dr[i]+total_Sy_Sx_dr[i]), 0.5*(total_Sx_Sz_dr[i]+total_Sz_Sx_dr[i])],[0.5*(total_Sx_Sy_dr[i]+total_Sy_Sx_dr[i]), total_Sy_Sy_dr[i], 0.5*(total_Sy_Sz_dr[i]+total_Sz_Sy_dr[i])],[0.5*(total_Sx_Sz_dr[i]+total_Sz_Sx_dr[i]), 0.5*(total_Sz_Sy_dr[i]+total_Sy_Sz_dr[i]), total_Sz_Sz_dr[i]]])
    
    gamma_matrix = C_matrix - np.array([[total_Sx_dr[i]**2, total_Sx_dr[i]*total_Sy_dr[i], total_Sx_dr[i]*total_Sz_dr[i]],[total_Sx_dr[i]*total_Sy_dr[i], total_Sy_dr[i]**2, total_Sz_dr[i]*total_Sy_dr[i]],[total_Sx_dr[i]*total_Sz_dr[i], total_Sz_dr[i]*total_Sy_dr[i], total_Sz_dr[i]**2]])
    
    C_matrix = 0.25*C_matrix
    
    gamma_matrix = 0.25*gamma_matrix
    
    X_matrix = (Natoms-1)*gamma_matrix + C_matrix
    
    eigvals, _ = scipy.linalg.eig(X_matrix)
    
    spin_inequal_param_list_30a[i] = ((Natoms+2)*Natoms/4.0)/np.trace(C_matrix)
    spin_inequal_param_list_30b[i] = np.trace(gamma_matrix)/(Natoms/2.0)

    spin_inequal_param_list_30c[i] = np.min(eigvals)/(np.trace(C_matrix) - Natoms/2.0)

    spin_inequal_param_list_30d[i] = ((Natoms-1)*np.trace(gamma_matrix) - (Natoms-2)*Natoms/4.0)/np.max(eigvals)

#'''

# save data (only connected part of corrs)

hf = h5py.File(direc+'Data_DTWA_GSM_large_det_lim_unitary_dynamics_'+h5_title, 'w')

hf.create_dataset('t_vals_dr', data=t_vals_dr, compression="gzip", compression_opts=9)

#hf.create_dataset('sz_dr', data=sz_dr, compression="gzip", compression_opts=9)
#hf.create_dataset('sx_dr', data=sx_dr, compression="gzip", compression_opts=9)
#hf.create_dataset('sy_dr', data=sy_dr, compression="gzip", compression_opts=9)


hf.create_dataset('total_Sz', data=total_Sz_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx', data=total_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy', data=total_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sz_Sz_dr', data=total_Sz_Sz_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sz_Sx_dr', data=total_Sz_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sz_Sy_dr', data=total_Sz_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx_Sx_dr', data=total_Sx_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx_Sz_dr', data=total_Sx_Sz_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx_Sy_dr', data=total_Sx_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy_Sy_dr', data=total_Sy_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy_Sx_dr', data=total_Sy_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy_Sz_dr', data=total_Sy_Sz_dr, compression="gzip", compression_opts=9)
#'''
hf.create_dataset('spin_inequal_param_list_30a', data=spin_inequal_param_list_30a, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30b', data=spin_inequal_param_list_30b, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30c', data=spin_inequal_param_list_30c, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30d', data=spin_inequal_param_list_30d, compression="gzip", compression_opts=9)
#'''


hf.close()

print('Data saved to: ', flush=True)
print(direc+'Data_DTWA_GSM_large_det_lim_unitary_dynamics_'+h5_title, flush=True)

tc1 = time.time()

print('runtime data analysis and saving = '+str(tc1-tb1),flush=True)


print("All runs done. May all your codes run this well! No decay run. :)", flush=True)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
