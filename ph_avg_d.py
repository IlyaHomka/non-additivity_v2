import numpy as np
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
# Integration
import scipy.integrate as integrate
import sympy as sp
from joblib import Parallel, delayed
from multiprocessing import Pool
from qutip import *
import multiprocessing

n_cores = 20
# Quantum Dots' Parametrs
eps = 1.0 # on-site mean local energy
delta = np.linspace(0.0,1.5,501)
eps_L = eps + 0.5 * delta
eps_R = eps - 0.5 * delta
g = 0.5 # coupling strength
Delta  = np.sqrt(delta**2 + g**2)
Gamma = 2e-2 #* Delta # the overall frequency scale of the dissipative dynamics
Omega = 100.0 # intra-reservoir tunnelling energy/hopping amplitude
# Basis Energies
E1 = eps + 0.5*Delta
E2 = eps - 0.5*Delta
# Reservoirs' parameters
T = np.linspace(2e-2, 2e-0, 501)
muL = 0.7
muR = 0.0
N = 2
tau =  np.linspace(1e-3, 1e3, 1001) # Time scale for CP test

# Dimensionless Fermi-Dirac fucntion expressed with reduced variable x=beta*(E-mu)
f = lambda x: 1./(1. + np.exp(x))
# Spectral density
def J(omega, Gamma, Omega):
     return Gamma/(2.0*np.pi)*np.sqrt(1.0-omega**2/Omega**2)*np.heaviside(Omega-abs(omega),0)
#  Probability distribution of excitation energy
def phi(omega, Gamma, Omega, E):
    E_dash = E
    Gamma_dash = 2.0*np.pi*J(E_dash, Gamma, Omega)
    return 1.0/((omega-E_dash)**2+(Gamma_dash/2.0)**2)*J(omega, Gamma, Omega)
# Trasnmission function
def Tau(omega, Gamma, Omega, E1, E2, g):
    return np.pi*g**2/2.0*phi(omega, Gamma, Omega, E1)*phi(omega, Gamma, Omega, E2)
# Gamma j
Gamma_function = lambda Gamma, Omega, E: (2.0*np.pi*J(E, Gamma, Omega))
# Current
def dI(omega, Gamma, Omega, E1, E2, g, T, muL, muR):
     return Tau(omega, Gamma, Omega, E1, E2, g)*(f((omega-muL)/T)-f((omega-muR)/T))
#Noise
def dS(omega, Gamma, Omega, E1, E2, g, T, muL, muR):
     return Tau(omega, Gamma, Omega, E1, E2, g)*(f((omega-muL)/T)+f((omega-muR)/T)-2.0*f((omega-muL)/T)*f((omega-muR)/T)-Tau(omega, Gamma, Omega, E1, E2, g)*(f((omega-muL)/T)-f((omega-muR)/T))**2)
#Functions for Current, Flcutuations
def Vectorize_Mat (M):
    return Qobj(np.concatenate([M[:, i:i+1] for i in range(M.shape[1])], axis=0))
# Correct QuTiP vec() (column-major):
#Inputs: LList: list of jump operators; NuList: list of jump coefficients (e.g. +1 and -1 for particle current jumps)
def JJ(LList, NuList):
    result = Qobj(np.zeros(tensor([LList[0].conj(), LList[0]]).shape))
    for i in range(len(NuList)):
        result +=  NuList[i]*Qobj(tensor([LList[i].conj(), LList[i]]),dims=tensor([LList[0].conj(), LList[0]]).shape)
    return result
#Inputs: sst: steady state as qutip object; JJj: Jump operators from above
def Current(sst,JJj):
    N_sst=sst.shape[0]
    id = Vectorize_Mat(qeye(N_sst))
    return(np.real(id.dag()*JJj*Vectorize_Mat(sst)))
##Inputs: Lbb: Lindbladian in qutip format; steadys: steady state in qutip format
# JJ2: the jump operator with all elements in NuList equal to 1; JJ: the jump operator
def Noise(Lbb, steadys, JJ2, JJ):
    N_sst=steadys.shape[0]
    vectsteady=Vectorize_Mat(steadys)
    I2L = Vectorize_Mat(qeye(N_sst)).dag()
    # Calculate the expression using NumPy operations
    alpha=JJ*vectsteady
    A=np.vstack((Lbb.full(),I2L.full()))
    B=np.vstack(((alpha - vectsteady*I2L*alpha).full(),0))
    z= Qobj(np.linalg.lstsq(A,B,rcond=None)[0])
    K=I2L*JJ2*vectsteady
    result = K-2*I2L*JJ*z
    return np.real(result)
# Creation and annihilation operators for N quantum dots: Define operators for each quantum dot
def fermion_annihilation(N, i):
    """
    Fermionic annihilation operator at site i using Jordan-Wigner transformation.

    Parameters:
    - N: total number of sites
    - i: site index (0-based)

    Returns:
    - Qobj: fermionic annihilation operator c_i
    """
    if not (0 <= i < N):
        raise ValueError("Index i must be between 0 and N-1")

    op_list = [sigmaz() for _ in range(i)]          # σ_z^⊗i
    op_list.append(sigmam())                        # σ_-
    op_list.extend([qeye(2) for _ in range(N - i - 1)])  # I^⊗(N-i-1)

    return tensor(op_list)

c0 = fermion_annihilation(N, 0)
c1 = fermion_annihilation(N, 1)
# Orthogonal rotation matrix
R = 1.0/np.sqrt(2 * Delta) * np.array([[np.sqrt(Delta+delta), -np.sqrt(Delta-delta)],[np.sqrt(Delta-delta), np.sqrt(Delta+delta)]])
# Gamma_L,1
GammaL1 = Gamma_function(Gamma, Omega, E1) * R[0,0]**2
# Gamma_L,2
GammaL2 = Gamma_function(Gamma, Omega, E2) * R[0,1]**2
# Gamma_R,1
GammaR1 = Gamma_function(Gamma, Omega, E1) * R[1,0]**2
# Gamma_R,1
GammaR2 = Gamma_function(Gamma, Omega, E2) * R[1,1]**2

# Decay and gain rates
def dg_m(omega, Gamma, Omega, E, mu, T):
  return phi(omega, Gamma, Omega, E) * (1.0 - f((omega-mu)/T))
def dg_p(omega, Gamma, Omega, E, mu, T):
  return phi(omega, Gamma, Omega, E) * f((omega-mu)/T)

# Lamda LR PM matrix: Re and Im parts
def dLambda_dash_p_re(omega, Gamma, Omega, E1, E2, muL, muR, T):
  return (Gamma_function(Gamma, Omega, E2)*phi(omega, Gamma, Omega, E2)-Gamma_function(Gamma, Omega, E1)*phi(omega, Gamma, Omega, E1))*(f((omega-muL)/T)+f((omega-muR)/T))
def dLambda_dash_p_im(omega, Gamma, Omega, E1, E2, muL, muR, T):
  return ((E1-omega)*phi(omega, Gamma, Omega, E1)-(E2-omega)*phi(omega, Gamma, Omega, E2))*(f((omega-muL)/T)-f((omega-muR)/T))
def dLambda_dash_m_re(omega, Gamma, Omega, E1, E2, muL, muR, T):
  return (Gamma_function(Gamma, Omega, E2)*phi(omega, Gamma, Omega, E2)-Gamma_function(Gamma, Omega, E1)*phi(omega, Gamma, Omega, E1))*(2.0 - f((omega-muL)/T)-f((omega-muR)/T))

def compute_row(i, T, Gamma, Omega, g, muL, muR, E1, E2):
    i = int(i)
    row = np.zeros(len(T))
    e1 = float(E1[i])
    e2 = float(E2[i])
    for j, Tj in enumerate(T):
        row[j] = integrate.quad(dI, -Omega, Omega, 
                               args=(Gamma, Omega, e1, e2, g, Tj, muL, muR))[0]
    return row
def compute_row_fluct(i, T, Gamma, Omega, g, muL, muR, E1, E2):
    i = int(i)
    row = np.zeros(len(T))
    e1 = float(E1[i])
    e2 = float(E2[i])
    for j, Tj in enumerate(T):
        row[j] = integrate.quad(dS, -Omega, Omega, 
                               args=(Gamma, Omega, e1, e2, g, Tj, muL, muR))[0]
    return row

def compute_rates_row(i, T, GammaL1, GammaL2, GammaR1, GammaR2, Gamma, Omega, E1, E2, muL, muR):
    """Same function as before - works perfectly with joblib"""
    i = int(i)
    nT = len(T)
    rates = np.zeros((8, nT))
    
    e1 = float(E1[i])
    e2 = float(E2[i])
    
    def safe_quad(func, *args, **kwargs):
        try:
            return integrate.quad(func, -Omega, Omega, *args, **kwargs)[0]
        except:
            return 0.0
    
    for j, Tj in enumerate(T):
        # Minus rates
        rates[0, j] = GammaL1[i] * safe_quad(dg_m, args=(Gamma, Omega, e1, muL, Tj), points=[e1])
        rates[1, j] = GammaR1[i] * safe_quad(dg_m, args=(Gamma, Omega, e1, muR, Tj), points=[e1])
        rates[2, j] = GammaL2[i] * safe_quad(dg_m, args=(Gamma, Omega, e2, muL, Tj), points=[e2])
        rates[3, j] = GammaR2[i] * safe_quad(dg_m, args=(Gamma, Omega, e2, muR, Tj), points=[e2])
        
        # Plus rates  
        rates[4, j] = GammaL1[i] * safe_quad(dg_p, args=(Gamma, Omega, e1, muL, Tj), points=[e1])
        rates[5, j] = GammaR1[i] * safe_quad(dg_p, args=(Gamma, Omega, e1, muR, Tj), points=[e1])
        rates[6, j] = GammaL2[i] * safe_quad(dg_p, args=(Gamma, Omega, e2, muL, Tj), points=[e2])
        rates[7, j] = GammaR2[i] * safe_quad(dg_p, args=(Gamma, Omega, e2, muR, Tj), points=[e2])
    
    return rates

def compute_lambda_row(i, T, Delta, E1, E2, Gamma, Omega, muL, muR, g):
    """
    Compute one row of Lambda matrices for fixed Delta index i.
    Returns [Lambda_dash_p_re, Lambda_dash_p_im, Lambda_dash_m_re, Lambda_dash_m_im]
    """
    i = int(i)
    nT = len(T)
    results = np.zeros((4, nT))
    
    delta_i = float(Delta[i])
    e1 = float(E1[i])
    e2 = float(E2[i])

    def safe_quad(func, *args, **kwargs):
        try:
            return integrate.quad(func, -Omega, Omega, *args, **kwargs)[0]
        except:
            return 0.0
    
    for j, Tj in enumerate(T):
        # Lambda_dash_p_re
        results[0, j] = g/(4*delta_i) * safe_quad(dLambda_dash_p_re, 
                                                 args=(Gamma, Omega, e1, e2, muL, muR, Tj))
        
        # Lambda_dash_p_im  
        results[1, j] = g/(2*delta_i) * safe_quad(dLambda_dash_p_im, 
                                                 args=(Gamma, Omega, e1, e2, muL, muR, Tj))
        
        # Lambda_dash_m_re
        results[2, j] = g/(4*delta_i) * safe_quad(dLambda_dash_m_re, 
                                                 args=(Gamma, Omega, e1, e2, muL, muR, Tj))
        
        # Lambda_dash_m_im = -Lambda_dash_p_im (vectorized)
        results[3, j] = -results[1, j]
    
    return results

# Lamda LR PM matrix: Re and Im parts
def d_xi(omega, Gamma, Omega, E1, E2, muL, muR, T):
  return (Gamma_function(Gamma, Omega, E2)*phi(omega, Gamma, Omega, E2)+Gamma_function(Gamma, Omega, E1)*phi(omega, Gamma, Omega, E1))*(f((omega-muL)/T)-f((omega-muR)/T))
def d_eta(omega, Gamma, Omega, E1, E2, muL, muR, T):
  return ((E1-omega)*phi(omega, Gamma, Omega, E1)-(E2-omega)*phi(omega, Gamma, Omega, E2))*(f((omega-muL)/T)-f((omega-muR)/T))
def compute_xi_eta_row(i, T, Delta, E1, E2, Gamma, Omega, muL, muR, g):
    """
    Compute one row of xi and eta matrices for fixed Delta index i.
    Returns [xi_row, eta_row]
    """
    i = int(i)
    nT = len(T)
    xi_row = np.zeros(nT)
    eta_row = np.zeros(nT)
    
    delta_i = float(Delta[i])
    e1 = float(E1[i])
    e2 = float(E2[i])

    def safe_quad(func, *args):
        return integrate.quad(func, -Omega, Omega, args=args, points=[e1, e2, muL, muR],
                          epsabs=1e-13, epsrel=1e-11, limit=400)[0]
    
    for j, Tj in enumerate(T):
        # xi
        xi_row[j] = g/(4*delta_i) * safe_quad(d_xi, 
                                            Gamma, Omega, e1, e2, muL, muR, Tj)
        
        # eta
        eta_row[j] = g/(4*delta_i) * safe_quad(d_eta, 
                                             Gamma, Omega, e1, e2, muL, muR, Tj)
    
    return xi_row, eta_row

# Parallel over Delta only (still 4-8x speedup)
I_avg_par = np.array(Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_row)(i, T, Gamma, Omega, g, muL, muR, E1, E2)
    for i in range(len(Delta))));

I_I_par = np.array(Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_row_fluct)(i, T, Gamma, Omega, g, muL, muR, E1, E2)
    for i in range(len(Delta))));

results = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_rates_row)(i, T, GammaL1, GammaL2, GammaR1, GammaR2, Gamma, Omega, E1, E2, muL, muR) 
    for i in range(len(delta)))
# Stack and extract
all_rates = np.stack(results, axis=0)

# Extract individual arrays
gammaL1_m = all_rates[:, 0, :]
gammaR1_m = all_rates[:, 1, :]
gammaL2_m = all_rates[:, 2, :]
gammaR2_m = all_rates[:, 3, :]
gammaL1_p = all_rates[:, 4, :]
gammaR1_p = all_rates[:, 5, :]
gammaL2_p = all_rates[:, 6, :]
gammaR2_p = all_rates[:, 7, :]

# Dash rates
gammaL_m_dash = (gammaL1_m + gammaL2_m)
gammaR_m_dash = (gammaR1_m + gammaR2_m)
gammaL_p_dash = (gammaL1_p + gammaL2_p)
gammaR_p_dash = (gammaR1_p + gammaR2_p)

Lresults = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_lambda_row)(i, T, Delta, E1, E2, Gamma, Omega, muL, muR, g)
    for i in range(len(Delta))
)
# Stack and extract
all_lambda = np.stack(Lresults, axis=0)

Lambda_dash_p_re = all_lambda[:, 0, :]
Lambda_dash_p_im = all_lambda[:, 1, :]
Lambda_dash_m_re = all_lambda[:, 2, :]
Lambda_dash_m_im = all_lambda[:, 3, :]
# Lamda LR PM matrix:
Lambda_dash_p_LR = Lambda_dash_p_re + 1j * Lambda_dash_p_im
Lambda_dash_m_LR = Lambda_dash_m_re + 1j * Lambda_dash_m_im
# Lamda RL PM matrix:
Lambda_dash_p_RL = Lambda_dash_p_LR.conjugate()
Lambda_dash_m_RL = Lambda_dash_m_LR.conjugate()

c0 = fermion_annihilation(N, 0)
c1 = fermion_annihilation(N, 1)
# Orthogonal rotation matrix
R = 1.0/np.sqrt(2 * Delta) * np.array([[np.sqrt(Delta+delta), -np.sqrt(Delta-delta)],[np.sqrt(Delta-delta), np.sqrt(Delta+delta)]])
# Gamma_L,1
GammaL1 = Gamma_function(Gamma, Omega, E1) * R[0,0]**2
# Gamma_L,2
GammaL2 = Gamma_function(Gamma, Omega, E2) * R[0,1]**2
# Gamma_R,1
GammaR1 = Gamma_function(Gamma, Omega, E1) * R[1,0]**2
# Gamma_R,1
GammaR2 = Gamma_function(Gamma, Omega, E2) * R[1,1]**2

def compute_iscp_row(i, T, eps_L, eps_R, Delta, Lambda_dash_p_LR, Lambda_dash_m_LR, 
                    Lambda_dash_p_RL, Lambda_dash_m_RL, gammaL_m_dash, gammaL_p_dash, 
                    gammaR_m_dash, gammaR_p_dash, g, c0, c1, I_avg, I_I, tau):
    """Compute one Delta[i] row - all T[j] in serial (cache H_sys)"""
    row_iscp = np.zeros(len(T), dtype=bool)
    
    # Cache H_sys for this Delta[i] - computed ONCE per row
    H_sys = eps_L[i] * c0.dag()*c0 + eps_R[i] * c1.dag()*c1 - 0.5 * g * (c0.dag() * c1 + c1.dag() * c0)
    
    # Precompute dissipators (same for all T)
    ad_b = c0 * c1.dag()
    D_notLinblad1 = spre(c1.dag()) * spost(c0) - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)
    cd_d = c1.dag() * c0
    D_notLinblad2 = spre(c0) * spost(c1.dag()) - 0.5 * spre(cd_d) - 0.5 * spost(cd_d)
    ed_f = c1 * c0.dag()
    D_notLinblad3 = spre(c0.dag()) * spost(c1) - 0.5 * spre(ed_f) - 0.5 * spost(ed_f)
    gd_f = c0.dag() * c1
    D_notLinblad4 = spre(c1) * spost(c0.dag()) - 0.5 * spre(gd_f) - 0.5 * spost(gd_f)
    
    for j, Tj in enumerate(T):
        # Non-Lindblad (matrix mult - fast)
        D_notLinblad = (Lambda_dash_p_LR[i,j] * D_notLinblad1 + 
                       Lambda_dash_m_LR[i,j] * D_notLinblad2 + 
                       Lambda_dash_p_RL[i,j] * D_notLinblad3 + 
                       Lambda_dash_m_RL[i,j] * D_notLinblad4)
        
        # Lindblad operators
        DL_m = np.sqrt(gammaL_m_dash[i,j]) * c0
        DL_p = np.sqrt(gammaL_p_dash[i,j]) * c0.dag()
        DR_m = np.sqrt(gammaR_m_dash[i,j]) * c1
        DR_p = np.sqrt(gammaR_p_dash[i,j]) * c1.dag()
        
        L_total = liouvillian(H_sys, [DL_m, DL_p, DR_m, DR_p]) + D_notLinblad
        
        # Steady state (MAIN BOTTLENECK)
        try:
            rho_ss_l = steadystate(L_total, method='eigen')
            vector_rho_ss = operator_to_vector(rho_ss_l)

            a = (c0.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0.dag()*rho_ss_l*c0))*c0).tr()
            b = (c0.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0*rho_ss_l*c0.dag()))*c0).tr()
            c = (c0*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0.dag()*rho_ss_l*c0))*c0.dag()).tr()
            d = (c0*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0*rho_ss_l*c0.dag()))*c0.dag()).tr()
            n_ss = (c0.dag() * c0*rho_ss_l).tr()

            x, y = sp.symbols('x y') #, real=False

            eq1 = sp.Eq(x - (x+y)*n_ss, I_avg[i,j])  # linear
            eq2 = sp.Eq(x - (x-y)*n_ss -a*x**2+(b+c)*x*y-d*y**2, I_I[i,j])  # quadratic

            solutions = sp.solve([eq1, eq2], (x, y))
            gamma_plus = np.array([sol[0].evalf(chop=1e-15) for sol in solutions], dtype=complex)  # [1.0, 3.0]
            gamma_minus = np.array([sol[1].evalf(chop=1e-15) for sol in solutions], dtype=complex)  
            L1 = np.sqrt((gamma_plus[0]))* c0.dag()
            L2 = np.sqrt((gamma_minus[0]))* c0
            L3 = np.sqrt((gamma_plus[1]))* c0.dag()
            L4 = np.sqrt((gamma_minus[1]))* c0
  
            a_0 = (c1.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1.dag()*rho_ss_l*c1))*c1).tr()
            b_0 = (c1.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1*rho_ss_l*c1.dag()))*c1).tr()
            c_0 = (c1*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1.dag()*rho_ss_l*c1))*c1.dag()).tr()
            d_0 = (c1*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1*rho_ss_l*c1.dag()))*c1.dag()).tr()
            n_rr = (c1.dag()*c1*rho_ss_l).tr()

            z, t = sp.symbols('z t') #, real=False

            eq = sp.Eq(z - (z+t)*n_rr, -I_avg[i,j])  # linear
            eq0 = sp.Eq(z - (z-t)*n_rr -a_0*z**2+(b_0+c_0)*z*t-d_0*t**2, I_I[i,j])  # quadratic

            results = sp.solve([eq, eq0], (z, t))
            z_vals = np.array([sol[0].evalf(chop=1e-15) for sol in results], dtype=complex)  # [1.0, 3.0]
            t_vals = np.array([sol[1].evalf(chop=1e-15) for sol in results], dtype=complex)  
            M1 = np.sqrt((z_vals[0]))* c1.dag()
            M2 = np.sqrt((t_vals[0]))* c1
            M3 = np.sqrt((z_vals[1]))* c1.dag()
            M4 = np.sqrt((t_vals[1]))* c1
            J0 = (tau*(L_total - sum(sprepost(L, L.dag()) for L in [L1, L2, M1, M2]))).expm() 
            eigvals_l = (to_choi(J0)).eigenenergies()
        # return np.all(eigvals_l >= atol)
            row_iscp[j] = np.all(eigvals_l >= -1e-10)#J0.iscp
        except:
            row_iscp[j] = False
    
    return row_iscp

def compute_iscp_row_g(i, T, R, E1, E2, Lambda_p_12, Lambda_p_21, 
                       gammaL1_m, gammaL1_p, gammaL2_m, gammaL2_p,
                       gammaR1_m, gammaR1_p, gammaR2_m, gammaR2_p,
                       I_avg_par, I_I_par, tau, c0, c1):  # Add c0, c1
    """Compute one Delta[i] row - all T[j] in serial (cache H_sys)"""
    row_iscp = np.zeros(len(T), dtype=bool)
    
    d1 = R[0,0,i] * c0 + R[0,1,i] * c1
    d2 = R[1,0,i] * c0 + R[1,1,i] * c1
    H_d = E1[i] * d1.dag() * d1 + E2[i] * d2.dag() * d2
    
    # Precompute dissipators (same for all T)
    D_int_12_p = (sprepost(d2.dag(), d1) - 0.5 * spre(d1 * d2.dag()) - 0.5 * spost(d1 * d2.dag()))
    D_int_12_m = (sprepost(d1, d2.dag()) - 0.5 * spre(d2.dag() * d1) - 0.5 * spost(d2.dag() * d1))
    D_int_21_p = (sprepost(d1.dag(), d2) - 0.5 * spre(d2 * d1.dag()) - 0.5 * spost(d2 * d1.dag()))
    D_int_21_m = (sprepost(d2, d1.dag()) - 0.5 * spre(d1.dag() * d2) - 0.5 * spost(d1.dag() * d2))
    
    x, y = sp.symbols('x y')
    z, t_sym = sp.symbols('z t')  # Avoid conflict with tau
    
    for j, Tj in enumerate(T):
        # **FIXED**: Use correct Lambda arrays
        D_int = (Lambda_p_12[i,j] * D_int_12_p - 
                 Lambda_p_12[i,j] * D_int_12_m +    
                 Lambda_p_21[i,j] * D_int_21_p -    
                 Lambda_p_21[i,j] * D_int_21_m)
        
        # Lindblad operators
        DL_1 = np.sqrt(gammaL1_m[i,j]) * d1 
        DL_2 = np.sqrt(gammaL2_m[i,j]) * d2 
        DL_1d = np.sqrt(gammaL1_p[i,j]) * d1.dag()
        DL_2d = np.sqrt(gammaL2_p[i,j]) * d2.dag()
        DR_1 = np.sqrt(gammaR1_m[i,j]) * d1
        DR_2 = np.sqrt(gammaR2_m[i,j]) * d2
        DR_1d = np.sqrt(gammaR1_p[i,j]) * d1.dag()
        DR_2d = np.sqrt(gammaR2_p[i,j]) * d2.dag()
        
        L_total_g = liouvillian(H_d, [DL_1, DL_2, DL_1d, DL_2d, DR_1, DR_2, DR_1d, DR_2d]) + D_int
        
        try:
            rho_ss_g = steadystate(L_total_g, method='eigen')
            
            # Site 1 coefficients
            aa = (d1.dag()*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d1.dag()*rho_ss_g*d1))*d1).tr()
            bb = (d1.dag()*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d1*rho_ss_g*d1.dag()))*d1).tr()
            cc = (d1*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d1.dag()*rho_ss_g*d1))*d1.dag()).tr()
            dd = (d1*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d1*rho_ss_g*d1.dag()))*d1.dag()).tr()
            n_11 = (d1.dag()*d1*rho_ss_g).tr()

            eq1 = sp.Eq(x - (x+y)*n_11, I_avg_par[i,j])
            eq2 = sp.Eq(x - (x-y)*n_11 -aa*x**2+(bb+cc)*x*y-dd*y**2, I_I_par[i,j])
            mysol = sp.solve([eq1, eq2], (x, y))
            gamma_plus_d = np.array([sol[0].evalf(chop=1e-15) for sol in mysol], dtype=complex)
            gamma_minus_d = np.array([sol[1].evalf(chop=1e-15) for sol in mysol], dtype=complex)
            
            L1 = np.sqrt(gamma_plus_d[0])* d1.dag()
            L2 = np.sqrt(gamma_minus_d[0])* d1
            
            # Site 2 coefficients  
            a_00 = (d2.dag()*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d2.dag()*rho_ss_g*d2))*d2).tr()
            b_00 = (d2.dag()*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d2*rho_ss_g*d2.dag()))*d2).tr()
            c_00 = (d2*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d2.dag()*rho_ss_g*d2))*d2.dag()).tr()
            d_00 = (d2*vector_to_operator(pseudo_inverse(L_total_g, method="solve")*operator_to_vector(d2*rho_ss_g*d2.dag()))*d2.dag()).tr()
            n_22 = (d2.dag()*d2*rho_ss_g).tr()

            eq_d = sp.Eq(z - (z+t_sym)*n_22, -I_avg_par[i,j])
            eq0_d = sp.Eq(z - (z-t_sym)*n_22 -a_00*z**2+(b_00+c_00)*z*t_sym-d_00*t_sym**2, I_I_par[i,j])
            res_d = sp.solve([eq_d, eq0_d], (z, t_sym))
            z_vals_d = np.array([sol[0].evalf(chop=1e-15) for sol in res_d], dtype=complex)
            t_vals_d = np.array([sol[1].evalf(chop=1e-15) for sol in res_d], dtype=complex)
            
            M1 = np.sqrt(z_vals_d[0])* d2.dag()
            M2 = np.sqrt(t_vals_d[0])* d2
            
            # ISCP test - FIXED tau variable
            Jg0 = (tau*(L_total_g - sum(sprepost(L, L.dag()) for L in [L1, L2, M1, M2]))).expm() 
            row_iscp[j] = Jg0.iscp
        except:
            row_iscp[j] = False
    
    return row_iscp

def compute_iscp_row(i, T, eps_L, eps_R, Delta, Lambda_dash_p_LR, Lambda_dash_m_LR, 
                    Lambda_dash_p_RL, Lambda_dash_m_RL, gammaL_m_dash, gammaL_p_dash, 
                    gammaR_m_dash, gammaR_p_dash, g, c0, c1, I_avg, I_I, tau):
    """Compute one Delta[i] row - all T[j] in serial (cache H_sys)"""
    nT = len(T)
    row_iscp = np.zeros(nT, dtype=bool)

    gamma_plus_0 = np.zeros(nT, dtype=complex)
    gamma_minus_0 = np.zeros(nT, dtype=complex)
    z_vals_0 = np.zeros(nT, dtype=complex)
    t_vals_0 = np.zeros(nT, dtype=complex)
    
    # Cache H_sys for this Delta[i] - computed ONCE per row
    H_sys = eps_L[i] * c0.dag()*c0 + eps_R[i] * c1.dag()*c1 - 0.5 * g * (c0.dag() * c1 + c1.dag() * c0)
    
    # Precompute dissipators (same for all T)
    ad_b = c0 * c1.dag()
    D_notLinblad1 = spre(c1.dag()) * spost(c0) - 0.5 * spre(ad_b) - 0.5 * spost(ad_b)
    cd_d = c1.dag() * c0
    D_notLinblad2 = spre(c0) * spost(c1.dag()) - 0.5 * spre(cd_d) - 0.5 * spost(cd_d)
    ed_f = c1 * c0.dag()
    D_notLinblad3 = spre(c0.dag()) * spost(c1) - 0.5 * spre(ed_f) - 0.5 * spost(ed_f)
    gd_f = c0.dag() * c1
    D_notLinblad4 = spre(c1) * spost(c0.dag()) - 0.5 * spre(gd_f) - 0.5 * spost(gd_f)
    
    for j, Tj in enumerate(T):
        # Non-Lindblad (matrix mult - fast)
        D_notLinblad = (Lambda_dash_p_LR[i,j] * D_notLinblad1 + 
                       Lambda_dash_m_LR[i,j] * D_notLinblad2 + 
                       Lambda_dash_p_RL[i,j] * D_notLinblad3 + 
                       Lambda_dash_m_RL[i,j] * D_notLinblad4)
        
        # Lindblad operators
        DL_m = np.sqrt(gammaL_m_dash[i,j]) * c0
        DL_p = np.sqrt(gammaL_p_dash[i,j]) * c0.dag()
        DR_m = np.sqrt(gammaR_m_dash[i,j]) * c1
        DR_p = np.sqrt(gammaR_p_dash[i,j]) * c1.dag()
        
        L_total = liouvillian(H_sys, [DL_m, DL_p, DR_m, DR_p]) + D_notLinblad
        
        # Steady state (MAIN BOTTLENECK)
        try:
            rho_ss_l = steadystate(L_total, method='eigen')
            vector_rho_ss = operator_to_vector(rho_ss_l)

            a = (c0.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0.dag()*rho_ss_l*c0))*c0).tr()
            b = (c0.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0*rho_ss_l*c0.dag()))*c0).tr()
            c = (c0*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0.dag()*rho_ss_l*c0))*c0.dag()).tr()
            d = (c0*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c0*rho_ss_l*c0.dag()))*c0.dag()).tr()
            n_ss = (c0.dag() *c0*rho_ss_l).tr()

            x, y = sp.symbols('x y') #, real=False

            eq1 = sp.Eq(x - (x+y)*n_ss, I_avg[i,j])  # linear
            eq2 = sp.Eq(x - (x-y)*n_ss -a*x**2+(b+c)*x*y-d*y**2, I_I[i,j])  # quadratic

            solutions = sp.solve([eq1, eq2], (x, y))
            gamma_plus = np.array([sol[0].evalf(chop=1e-15) for sol in solutions], dtype=complex)  # [1.0, 3.0]
            gamma_minus = np.array([sol[1].evalf(chop=1e-15) for sol in solutions], dtype=complex)  
            L1 = np.sqrt((gamma_plus[0]))* c0.dag()
            L2 = np.sqrt((gamma_minus[0]))* c0
            #L3 = np.sqrt((gamma_plus[1]))* c0.dag()
            #L4 = np.sqrt((gamma_minus[1]))* c0
  
            a_0 = (c1.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1.dag()*rho_ss_l*c1))*c1).tr()
            b_0 = (c1.dag()*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1*rho_ss_l*c1.dag()))*c1).tr()
            c_0 = (c1*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1.dag()*rho_ss_l*c1))*c1.dag()).tr()
            d_0 = (c1*vector_to_operator(pseudo_inverse(L_total, method="solve")*operator_to_vector(c1*rho_ss_l*c1.dag()))*c1.dag()).tr()
            n_rr = (c1.dag()*c1*rho_ss_l).tr()

            z, t = sp.symbols('z t') #, real=False

            eq = sp.Eq(z - (z+t)*n_rr, -I_avg[i,j])  # linear
            eq0 = sp.Eq(z - (z-t)*n_rr -a_0*z**2+(b_0+c_0)*z*t-d_0*t**2, I_I[i,j])  # quadratic

            results = sp.solve([eq, eq0], (z, t))
            z_vals = np.array([sol[0].evalf(chop=1e-15) for sol in results], dtype=complex) 
            t_vals = np.array([sol[1].evalf(chop=1e-15) for sol in results], dtype=complex)  
            M1 = np.sqrt((z_vals[0]))* c1.dag()
            M2 = np.sqrt((t_vals[0]))* c1

            gamma_plus_0[j] = gamma_plus[0]
            gamma_minus_0[j] = gamma_minus[0]
            z_vals_0[j] = z_vals[0]
            t_vals_0[j] = t_vals[0]
            
            L_generator = L_total - sum(sprepost(L, L.dag()) for L in [L1, L2, M1, M2])
            tau_cumulative = True
            for k, t_val in enumerate(tau): 
                if not tau_cumulative:  # Early exit tau loop
                    break
                J0 = (t_val*L_generator).expm()
                tau_cumulative = tau_cumulative  and J0.iscp
            row_iscp[j] = tau_cumulative  
        except:
            row_iscp[j] = False
            gamma_plus_0[j] = np.nan + 0j
            gamma_minus_0[j] = np.nan + 0j
            z_vals_0[j] = np.nan + 0j
            t_vals_0[j] = np.nan + 0j
    
    return row_iscp, gamma_plus_0, gamma_minus_0, z_vals_0, t_vals_0

# RUN
iscp_test_l0 = np.zeros((len(Delta), len(T)), dtype=bool)
gamma_plus_2d = np.zeros((len(Delta), len(T)), dtype=complex)
gamma_minus_2d = np.zeros((len(Delta), len(T)), dtype=complex)
z_vals_2d = np.zeros((len(Delta), len(T)), dtype=complex)
t_vals_2d = np.zeros((len(Delta), len(T)), dtype=complex)


Fresults = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_iscp_row)(i, T, eps_L, eps_R, Delta, Lambda_dash_p_LR, Lambda_dash_m_LR, 
                            Lambda_dash_p_RL, Lambda_dash_m_RL, gammaL_m_dash, gammaL_p_dash, 
                            gammaR_m_dash, gammaR_p_dash, g, c0, c1, I_avg_par, I_I_par, tau)
    for i in range(len(Delta))
)
for i, (iscp_row, gp0, gm0, z0, t0) in enumerate(Fresults):
    iscp_test_l0[i, :] = iscp_row
    gamma_plus_2d[i, :] = gp0
    gamma_minus_2d[i, :] = gm0
    z_vals_2d[i, :] = z0
    t_vals_2d[i, :] = t0

iscp_test_g0 = np.zeros((len(Delta), len(T)), dtype=bool)
gamma_plus_d = np.zeros((len(Delta), len(T)), dtype=complex)
gamma_minus_d = np.zeros((len(Delta), len(T)), dtype=complex)
z_vals_d = np.zeros((len(Delta), len(T)), dtype=complex)
t_vals_d = np.zeros((len(Delta), len(T)), dtype=complex)
#iscp_test_g0 = np.zeros((len(Delta), len(T)), dtype=bool)
Gresults = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
    delayed(compute_iscp_row_g)(i, T, R, E1, E2, Lambda_p_12, Lambda_p_21, 
                                gammaL1_m, gammaL1_p, gammaL2_m, gammaL2_p,
                                gammaR1_m, gammaR1_p, gammaR2_m, gammaR2_p,
                                I_avg_par, I_I_par, tau, c0, c1)
    for i in range(len(Delta))
)
for i, (iscp_row, gp0, gm0, z0, t0) in enumerate(Gresults):
    iscp_test_g0[i, :] = iscp_row
    gamma_plus_d[i, :] = gp0
    gamma_minus_d[i, :] = gm0
    z_vals_d[i, :] = z0
    t_vals_d[i, :] = t0

np.savez('rates_l.npz', gamma_plus_2d=gamma_plus_2d,
         gamma_minus_2d=gamma_minus_2d, z_vals_2d=z_vals_2d, t_vals_2d=t_vals_2d)
np.savez('koskos.npz', Lambda_dash_p_LR=Lambda_dash_p_LR, Lambda_dash_m_LR=Lambda_dash_m_LR) #,
np.savez('ph_avg_p.npz', iscp_test_l0=iscp_test_l0)
