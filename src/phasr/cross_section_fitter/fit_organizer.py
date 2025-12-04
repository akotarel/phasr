from ..config import local_paths

from .. import constants

import numpy as np
pi = np.pi

import os
import copy

from .fit_performer import fitter
from .fit_initializer import initializer
from .data_prepper import load_dataset
#from .uncertainties import add_systematic_uncertanties
from .. import nucleus
from phasr.nuclei.parameterizations.fourier_bessel import get_ai_from_charge_density

from multiprocessing import Pool, cpu_count

from ..utility.mpsentinel import MPSentinel
MPSentinel.As_master()

def parallel_fitting_manual(datasets:dict,Z:int,A:int,RN_tuples=[],redo_N=False,redo_aggressive=False, redo_R=False,N_processes=cpu_count()-2,**args):
    
    results={}
    
    if MPSentinel.Is_master():    
        
        pairings = []
        
        for i in range(len(RN_tuples)):
            R,N=RN_tuples[i]
            R = np.float64(R)
            N = np.int64(N)
            pairings.append((copy.deepcopy(datasets),Z,A,R,N,args))
        
        N_tasks = len(pairings)
        N_processes = np.min([N_processes,N_tasks])
        
        print('Queuing',N_tasks,'tasks, which will be performed over',N_processes,'processes.')
        
        with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
            results = pool.starmap(fit_runner,pairings)
    
        print('Finished all tasks.')

        results_dict = { 'R'+str(pairings[i][3]) + '_N'+str(pairings[i][4]) : results[i] for i in range(len(results))}

        if redo_N:

            print('Check if any fits need to be redone. (redo_N)')

            #redo fits with bad convergence
            redo_pairings = []
            eps_N=1e-3

            for j in range(len(pairings)):
                
                pairing = pairings[j]
                key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4])
                best_key_RN = key_RN
                chisq_RN = results_dict[key_RN]['chisq']
                best_chisq_RN = chisq_RN
                for N_off in np.arange(pairing[4]-3 if redo_aggressive else 1,0,-1):
                    key_RNm1 = 'R'+str(pairing[3]) + '_N'+str(pairing[4]-N_off) 
                    if key_RNm1 in results_dict:
                        chisq_RNm1 = results_dict[key_RNm1]['chisq']
                        if eps_N<(best_chisq_RN-chisq_RNm1)/chisq_RNm1:
                            best_key_RN = key_RNm1
                            best_chisq_RN = chisq_RNm1
                            best_N_off = N_off
                
                if best_key_RN != key_RN:
                    print('For '+ key_RN +' chi^2 with '+str(best_N_off)+' less parameter is more than 1 permil better:',chisq_RN,'vs',best_chisq_RN)
                    
                    pairing[5]['ai_ini'] = results_dict[best_key_RN]['ai']
                    for data_name in pairing[0]:
                        if pairing[0][data_name].get('fit_luminosities','n')=='y':
                            pairing[0][data_name]['luminosities'] = results_dict[best_key_RN][data_name]['luminosities']

                    redo_pairings.append(copy.deepcopy(pairing))
                    
            N_tasks = len(redo_pairings)
            
            if N_tasks>0:
                N_processes = np.min([N_processes,N_tasks])
                print('Queuing',N_tasks,'tasks that need to be redone, which will be performed over',N_processes,'processes.')
                
                with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
                    redo_results = pool.starmap(fit_runner,redo_pairings)
                
                redo_results_dict = { 'R'+str(redo_pairings[i][3]) + '_N'+str(redo_pairings[i][4]) : redo_results[i] for i in range(len(redo_results))}

                for pairing in redo_pairings:
                    key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4]) 
                    chisq_RN_old = results_dict[key_RN]['chisq']
                    chisq_RN_new = redo_results_dict[key_RN]['chisq']
                    if chisq_RN_new < chisq_RN_old:
                        results_dict[key_RN] = redo_results_dict[key_RN]
            else:
                print('No fits need to be redone.')
                
        if redo_R:
            print('Check if any fits need to be redone (redo_R).') 
            #redo fits with bad convergence
            redo_pairings = []

            N_list=np.array([pairing[4] for pairing in pairings],dtype=int)
            N_list=np.unique(N_list)
            
            for N in N_list:
                R_list=np.array([pairing[3] for pairing in pairings if pairing[4]==N],dtype=float)
                R_list=np.sort(R_list)
                
                if len(R_list)>=3:
                    i=1
                    while i<len(R_list)-1:
                        R_sample=np.array([R_list[i-1],R_list[i],R_list[i+1]],dtype=float)
                        chisq_sample=np.array([results_dict['R'+str(R) + '_N'+str(N)]['chisq'] for R in R_sample],dtype=float)
                        
                        curvature=2/(R_sample[2]-R_sample[0]) * ( (chisq_sample[2]-chisq_sample[1])/(R_sample[2]-R_sample[1]) - (chisq_sample[1]-chisq_sample[0])/(R_sample[1]-R_sample[0]) )

                        vector_1=np.array([R_sample[1]-R_sample[0],chisq_sample[1]-chisq_sample[0]])
                        vector_2=np.array([R_sample[2]-R_sample[1],chisq_sample[2]-chisq_sample[1]])
                        angle=np.arccos( np.dot(vector_1,vector_2) / (np.linalg.norm(vector_1)*np.linalg.norm(vector_2)) )
                        # criterion that the middle point (R1,chisq_R1[1]) has not converged
                        if curvature<0 and angle>10.*np.pi/180:
                            pairing= (copy.deepcopy(datasets),Z,A,R_sample[1],N,args)

                            guess_key_RN = 'R'+str(R_sample[0]) + '_N'+str(N)
                            guess_nucleus = nucleus('nucleus_guess',Z,A,ai=results_dict[guess_key_RN]['ai'],R=R_sample[0])
                            pairing[5]['ai_ini'] = get_ai_from_charge_density(guess_nucleus.charge_density, R_sample[1], N) # get ai's adjusted to new R
                            for data_name in pairing[0]:
                                if pairing[0][data_name].get('fit_luminosities','n')=='y':
                                    pairing[0][data_name]['luminosities'] = results_dict[guess_key_RN][data_name]['luminosities']

                            redo_pairings.append(copy.deepcopy(pairing))
                            R_list=np.delete(R_list,i) # this point should not be taken into account
                            i-=1
                        i+=1
                

            N_tasks = len(redo_pairings)
            if N_tasks>0:
                N_processes = np.min([N_processes,N_tasks])
                print('Queuing',N_tasks,'tasks that need to be redone, which will be performed over',N_processes,'processes.')
                
                with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
                    redo_results = pool.starmap(fit_runner,redo_pairings)
                
                redo_results_dict = { 'R'+str(redo_pairings[i][3]) + '_N'+str(redo_pairings[i][4]) : redo_results[i] for i in range(len(redo_results))}

                for pairing in redo_pairings:
                    key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4]) 
                    chisq_RN_old = results_dict[key_RN]['chisq']
                    chisq_RN_new = redo_results_dict[key_RN]['chisq']
                    if chisq_RN_new < chisq_RN_old:
                        results_dict[key_RN] = redo_results_dict[key_RN]
                
        
    return results_dict

def parallel_fitting_automatic(datasets:dict,Z:int,A:int,Rs=np.arange(5.00,12.00,0.25),N_base_offset=0,N_base_span=2,redo_N=False,redo_R=False,redo_aggressive=False,N_processes=cpu_count()-2,**args):
    
    results={}
    
    if MPSentinel.Is_master():    
        q_max=0
        for data_name in datasets:
            dataset, _, _ = load_dataset(data_name,Z,A,verbose=False) 
            energy = dataset[:,0]
            theta = dataset[:,1]
            q_mom_approx = 2*energy/constants.hc*np.sin(theta/2)
            q_mom_approx = np.append(q_mom_approx,q_max)
            q_max = np.max(q_mom_approx)
            Ns = np.ceil((Rs*q_max)/pi).astype(int)+N_base_offset
        
        pairings = []
        
        for i in range(len(Rs)):
            R=np.float64(Rs[i])
            N=np.int64(Ns[i])
            for N_offset in np.arange(N-N_base_span,N+N_base_span+1,1,dtype=int):
                if N_offset>2:
                    pairings.append((copy.deepcopy(datasets),Z,A,R,N_offset,args))
        
        N_tasks = len(pairings)
        N_processes = np.min([N_processes,N_tasks])
        print('Queuing',N_tasks,'tasks, which will be performed over',N_processes,'processes.')
        
        with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
            results = pool.starmap(fit_runner,pairings)
        
        print('Finished all tasks.')

        results_dict = { 'R'+str(pairings[i][3]) + '_N'+str(pairings[i][4]) : results[i] for i in range(len(results))}

        if redo_N:

            print('Check if any fits need to be redone (redo_N).')
            #redo fits with bad convergence
            redo_pairings = []
            eps_N=1e-3
            for j in range(len(pairings)):
                pairing = pairings[j]
                key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4])
                best_key_RN = key_RN
                chisq_RN = results_dict[key_RN]['chisq']
                best_chisq_RN = chisq_RN
                for N_off in np.arange(pairing[4]-3 if redo_aggressive else 1,0,-1):
                    key_RNm1 = 'R'+str(pairing[3]) + '_N'+str(pairing[4]-N_off) 
                    if key_RNm1 in results_dict:
                        chisq_RNm1 = results_dict[key_RNm1]['chisq']
                        if eps_N<(best_chisq_RN-chisq_RNm1)/chisq_RNm1:
                            best_key_RN = key_RNm1
                            best_chisq_RN = chisq_RNm1
                            best_N_off = N_off
                
                if best_key_RN != key_RN:
                    print('For '+ key_RN +' chi^2 with '+str(best_N_off)+' less parameter is more than 1 permil better:',chisq_RN,'vs',best_chisq_RN)
                    
                    pairing[5]['ai_ini'] = results_dict[best_key_RN]['ai']
                    for data_name in pairing[0]:
                        if pairing[0][data_name].get('fit_luminosities','n')=='y':
                            pairing[0][data_name]['luminosities'] = results_dict[best_key_RN]['luminosities'][data_name]

                    redo_pairings.append(copy.deepcopy(pairing))

            N_tasks = len(redo_pairings)
            if N_tasks>0:
                N_processes = np.min([N_processes,N_tasks])
                print('Queuing',N_tasks,'tasks that need to be redone, which will be performed over',N_processes,'processes.')
                
                with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
                    redo_results = pool.starmap(fit_runner,redo_pairings)
                
                redo_results_dict = { 'R'+str(redo_pairings[i][3]) + '_N'+str(redo_pairings[i][4]) : redo_results[i] for i in range(len(redo_results))}

                for pairing in redo_pairings:
                    key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4]) 
                    chisq_RN_old = results_dict[key_RN]['chisq']
                    chisq_RN_new = redo_results_dict[key_RN]['chisq']
                    if chisq_RN_new < chisq_RN_old:
                        results_dict[key_RN] = redo_results_dict[key_RN]

        if redo_R:
            print('Check if any fits need to be redone (redo_R).') 
            #redo fits with bad convergence
            redo_pairings = []

            N_list=np.array([pairing[4] for pairing in pairings],dtype=int)
            N_list=np.unique(N_list)
            
            for N in N_list:
                R_list=np.array([pairing[3] for pairing in pairings if pairing[4]==N],dtype=float)
                R_list=np.sort(R_list)
                
                if len(R_list)>=3:
                    i=1
                    while i<len(R_list)-1:
                        R_sample=np.array([R_list[i-1],R_list[i],R_list[i+1]],dtype=float)
                        chisq_sample=np.array([results_dict['R'+str(R) + '_N'+str(N)]['chisq'] for R in R_sample],dtype=float)
                        
                        curvature=2/(R_sample[2]-R_sample[0]) * ( (chisq_sample[2]-chisq_sample[1])/(R_sample[2]-R_sample[1]) - (chisq_sample[1]-chisq_sample[0])/(R_sample[1]-R_sample[0]) )

                        vector_1=np.array([R_sample[1]-R_sample[0],chisq_sample[1]-chisq_sample[0]])
                        vector_2=np.array([R_sample[2]-R_sample[1],chisq_sample[2]-chisq_sample[1]])
                        angle=np.arccos( np.dot(vector_1,vector_2) / (np.linalg.norm(vector_1)*np.linalg.norm(vector_2)) )
                        # criterion that the middle point (R1,chisq_R1[1]) has not converged
                        if curvature<0 and angle>10.*np.pi/180:
                            pairing= (copy.deepcopy(datasets),Z,A,R_sample[1],N,args)

                            guess_key_RN = 'R'+str(R_sample[0]) + '_N'+str(N)
                            guess_nucleus = nucleus('nucleus_guess',Z,A,ai=results_dict[guess_key_RN]['ai'],R=R_sample[0])
                            pairing[5]['ai_ini'] = get_ai_from_charge_density(guess_nucleus.charge_density, N,R_sample[1]) # get ai's adjusted to new R
                            for data_name in pairing[0]:
                                if pairing[0][data_name].get('fit_luminosities','n')=='y':
                                    pairing[0][data_name]['luminosities'] = results_dict[guess_key_RN]['luminosities'][data_name]

                            redo_pairings.append(copy.deepcopy(pairing))
                            R_list=np.delete(R_list,i) # this point should not be taken into account
                            i-=1
                        i+=1

            N_tasks = len(redo_pairings)
            for pairing in redo_pairings:
                key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4])
                print("R"+str(pairing[3])+ '_N'+str(pairing[4])+'chisq_old'+str(results_dict[key_RN]['chisq']))
            if N_tasks>0:
                N_processes = np.min([N_processes,N_tasks])
                print('Queuing',N_tasks,'tasks that need to be redone, which will be performed over',N_processes,'processes.')
                
                with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
                    redo_results = pool.starmap(fit_runner,redo_pairings)
                
                redo_results_dict = { 'R'+str(redo_pairings[i][3]) + '_N'+str(redo_pairings[i][4]) : redo_results[i] for i in range(len(redo_results))}

                for pairing in redo_pairings:
                    key_RN = 'R'+str(pairing[3]) + '_N'+str(pairing[4]) 
                    chisq_RN_old = results_dict[key_RN]['chisq']
                    chisq_RN_new = redo_results_dict[key_RN]['chisq']
                    if chisq_RN_new < chisq_RN_old:
                        results_dict[key_RN] = redo_results_dict[key_RN]

    return results_dict

def fit_runner(datasets:dict,Z,A,R,N,args):
    print("Start fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    
    args = copy.deepcopy(args) # prevents that args are poped from the source
    
    if 'initialize_from' in args:
        initialize_from = args['initialize_from']
        args.pop('initialize_from')
    else:
        initialize_from = 'reference'
    
    if 'ai_ini' in args:
        ai_ini = args['ai_ini']
        args.pop('ai_ini')
        #print(ai_ini)
    else:
        ai_ini = None

    initialization = initializer(Z,A,R,N,datasets,ai=ai_ini,initialize_from=initialize_from)
    result = fitter(datasets,initialization,**args)
    print("Finished fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    return result

def select_RN_based_on_property(results_dict,property,limit,sign=+1):
    
    RN_tuples=[]
    for key in results_dict:
        if sign*results_dict[key][property] > sign*limit:
            RN_tuples.append((results_dict[key]['R'],results_dict[key]['N']))
    
    return RN_tuples

def split_based_on_asymptotic_and_p_val(results_dict,qs=[400,580,680,1000],dq=1.,m=None,p_val_lim=0):
    
    if m is None:
        q0=np.arange(qs[0],qs[1],dq)
        q1=np.arange(qs[1],qs[2],dq)
        q2=np.arange(qs[2],qs[3],dq)
        As=[]
        ms=[]
        for key in results_dict:
            #
            result = results_dict[key]
            nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            #
            F0=np.abs(nuc_result.form_factor(q0))
            F1=np.abs(nuc_result.form_factor(q1))
            F0_max=np.max(F0)
            q0_max=q0[np.argmax(F0)]
            F1_max=np.max(F1)
            q1_max=q1[np.argmax(F1)]
            #
            m_key=-np.log(F1_max/F0_max)/np.log(q1_max/q0_max)
            #
            A=F0_max*q0_max**m_key
            As.append(A)
            ms.append(m_key)
        m_min=np.min(ms)
        A=As[np.argmin(ms)]
    else:
        m_min=0
        
    if m_min<4:# or (m is not None):
        if m is None:
            m_min= 4
            q0=np.arange(qs[1],qs[2],dq)
            q2=np.arange(qs[2],qs[3],dq)
        else:
            m_min= m
            q0=np.arange(qs[0],qs[1],dq)
            q2=np.arange(qs[1],qs[2],dq)
            
        As=[]
        for key in results_dict:
            #
            result = results_dict[key]
            nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            #
            F0=np.abs(nuc_result.form_factor(q0))
            F0_max=np.max(F0)
            q0_max=q0[np.argmax(F0)]
            #
            A=F0_max*q0_max**m_min
            As.append(A)
        A=np.max(As)
    
    print('Asymptotic Parameter: m='+str(m_min))
    
    F2_lim = A/q2**m_min
    
    def limit_asymptotic(q):
        return A/q**m_min
    
    results_dict_survive, results_dict_veto = {}, {}
    
    for key in results_dict:
        
        result = results_dict[key]
        nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            
        F2=np.abs(nuc_result.form_factor(q2))
        
        if np.all(F2_lim>F2) and result['p_val']>=p_val_lim:
            results_dict_survive[key]=result
        else:
            results_dict_veto[key]=result
        
    return results_dict_survive, results_dict_veto, limit_asymptotic
