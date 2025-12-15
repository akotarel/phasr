import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.special import spence 
from phasr import constants,masses

alpha = constants.alpha_el
le=386.159 # in fm
e2=1.44 # in MeV*fm
me=masses.me
hc=constants.hc

def Li(x):
        if np.isscalar(x) and x<1 and x>0:
            return spence(1 - x)
        else:
            raise ValueError("Li function only implemented for inputs in (0,1)")
Li=np.vectorize(Li)

def exp_continuation(x_values,y_values):
    if x_values.size != 2 or y_values.size != 2:
        raise ValueError("x_values and y_values must have size 2.")
    N0=y_values[0]
    Lambda=-(np.log(y_values[1]/y_values[0]))/(x_values[1] - x_values[0])
    return N0, Lambda

class potential_corrections():
    """Module for calculating QED corrections to nuclear potentials."""
    def __init__(self, nucleus,included_corrections=[],r_values=None,threshold=1e-4):
        self.nucleus=nucleus
        self.Z=nucleus.Z
        self.included_corrections=included_corrections
        
        self.threshold=threshold
        self.r_critical=np.inf
        self.N0=0;self.Lambda=0

        if r_values is not None:
            self.r_values=r_values
        else:
            self.r_values=np.concatenate([np.linspace(0.01,50,40), np.logspace(np.log10(60),np.log10(5000),20)])

         # Get path relative to this file
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        K0_PATH = os.path.join(MODULE_DIR, 'function_splines', 'K0_func.txt')
        L0_PATH = os.path.join(MODULE_DIR, 'function_splines', 'L0_func.txt')
        
        K0_data= np.loadtxt(K0_PATH)
        L0_data= np.loadtxt(L0_PATH)
        self.K0 = CubicSpline(K0_data[:,0], K0_data[:,1],bc_type='natural')
        self.L0 = CubicSpline(L0_data[:,0], L0_data[:,1],bc_type='natural')

        self.corrections={}
        self.calculate_corrections()
        self.continue_corrections()

    def potential_V2_integral(self,r_prime,r):
        return r_prime*self.nucleus.charge_density(r_prime)*(self.K0(2/le*np.abs(r-r_prime))-self.K0(2/le*(r+r_prime)))
    
    def V2_Uehling(self,r):
        cutoff=10*le
        subintervals=np.logspace(-1,np.log10(cutoff),500)
        return -2./3.*self.Z*alpha*le*e2/r*quad(lambda r_prime: self.potential_V2_integral(r_prime,r), 0, cutoff, points=subintervals, limit=2*subintervals.size)[0]

    def potential_V4_integral(self,r_prime,r):
        return r_prime*self.nucleus.charge_density(r_prime)*(self.L0(2/le*np.abs(r-r_prime))-self.L0(2/le*(r+r_prime)))
    
    def V4_Uehling(self,r):
        cutoff=10*le
        subintervals=np.logspace(-1,np.log10(cutoff),500)
        return -self.Z*alpha**2*le*e2/(np.pi*r)*quad(lambda r_prime: self.potential_V4_integral(r_prime,r), 0, cutoff, points=subintervals, limit=2*subintervals.size)[0]

    def F1(self,q):
        v = np.sqrt(1 + 4*(me/q)**2)
        return alpha/(2*np.pi)*((v**2+1)/(4*v)*np.log((v+1)/(v-1))*np.log((v**2-1)/(4*v**2)) \
                         + (2*v**2+1.)/(2*v)*np.log((v+1)/(v-1)) - 2 + (v**2+1)/(2*v)* (Li((v+1)/(2*v)) - Li((v-1)/(2*v))))
    
    def vs_integrand(self, q, r):
        return np.sin(q*r/hc)/(q*r/hc)*(alpha*self.nucleus.form_factor(q)*self.F1(q))
    
    def V_vs(self, r):
        integral = quad(lambda q: self.vs_integrand(q,r), 0, np.inf, limit=50000)[0]
        return -2/np.pi*self.nucleus.Z*integral

    def calculate_corrections(self):
        # Potential in fm^-1 units
        print("Calculating QED potential corrections...")
        if 'Uehling_2' in self.included_corrections:
            V2_vals = np.array([self.V2_Uehling(r)/hc for r in self.r_values])
            self.corrections['Uehling_2']=CubicSpline(self.r_values, V2_vals, bc_type='natural')
        if 'Uehling_4' in self.included_corrections:
            V4_vals = np.array([self.V4_Uehling(r)/hc for r in self.r_values])
            self.corrections['Uehling_4']=CubicSpline(self.r_values, V4_vals, bc_type='natural')
        if 'vs' in self.included_corrections:
            Vvs_vals = np.array([self.V_vs(r)/hc for r in self.r_values])
            self.corrections['vs']=CubicSpline(self.r_values, Vvs_vals, bc_type='natural')
        return
    
    def continue_corrections(self):
        print(f"Continuing corrections to high energies below threshold {self.threshold}...")
        Coulomb_potential=self.nucleus.electric_potential(self.r_values)
        corrections=self.corrected_potential(self.r_values)-Coulomb_potential
        ratio=np.abs(corrections/Coulomb_potential)

        index=None
        r_match=np.array([])
        ratio_match=np.array([])
        for i in np.arange(self.r_values.size):
            if ratio[i]<self.threshold and self.r_values[i]>500:
                index=i
                r_match=np.append(r_match, self.r_values[i])
                ratio_match=np.append(ratio_match, ratio[i])
                break

        if r_match.size==0:
            print("Warning: The given corrections remain always above the threshold. Continuation is not possible")
        else:
            for i in np.arange(index+1,self.r_values.size):
                if self.r_values[i]>1.2*r_match[0]: # keep some distance between matching points to achieve better stability
                    r_match=np.append(r_match, self.r_values[i])
                    ratio_match=np.append(ratio_match, ratio[i])
                    break
            if r_match.size<2:
                print("Warning: Not suitable points found for the potential continuation.")
            else:
                self.N0,self.Lambda=exp_continuation(r_match, ratio_match)
        
        self.r_critical=r_match[0] if r_match.size>0 else np.inf


    def corrected_potential(self, r):
        if np.isscalar(r):
            V_corr=self.nucleus.electric_potential(r)
            if r<self.r_critical:
                for key in self.corrections:
                    V_corr+=self.corrections[key](r)
            else:
                V_corr= V_corr*(1. + self.N0*np.exp(-self.Lambda*(r-self.r_critical)))
            return V_corr
        else:
            V_corr=self.nucleus.electric_potential(r)
            for i in range(r.size):
                if r[i]<self.r_critical:
                    for key in self.corrections:
                        V_corr[i]+=self.corrections[key](r[i])
                else:
                    V_corr[i]= V_corr[i]*(1. + self.N0*np.exp(-self.Lambda*(r[i]-self.r_critical)))
            return V_corr

    
