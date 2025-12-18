from .. import constants

import numpy as np
pi = np.pi

import copy

import numdifftools as ndt
from scipy.linalg import inv
from scipy.optimize import minimize#, OptimizeResult
from scipy.integrate import quad
from scipy.stats import chi2

from .statistical_measures import minimization_measures
from .fit_initializer import initializer
from .parameters import parameter_set
from .data_prepper import load_dataset, load_barrett_moment
from .pickler import pickle_load_all_results_dicts_R_N, pickle_load_result_dict, pickle_dump_result_dict

from ..dirac_solvers import crosssection_lepton_nucleus_scattering

from .. import nucleus

def fitter(datasets:dict,initialization:initializer,barrett_moment_keys=[],monotonous_decrease_precision=np.inf,xi_diff_convergence_limit=1e-4,numdifftools_step=1.e-4,verbose=True,renew=False,cross_section_args={},**minimizer_args):
    ''' **minimzer_args is passed to scipy minimize '''
    # usually: monotonous_decrease_precision=0.04
    # xi_diff_convergence_limit is a deprecated feature and has no effect   
    
    list_fit_luminosities=[datasets[data_name].get('fit_luminosities','n') for data_name in datasets]
    settings_dict = {'datasets':list(datasets.keys()),'fit_luminosities': list_fit_luminosities,'datasets_barrett_moment':barrett_moment_keys,'monotonous_decrease_precision':monotonous_decrease_precision,'xi_diff_convergence_limit':xi_diff_convergence_limit,'numdifftools_step':numdifftools_step,**cross_section_args,**minimizer_args}
    
    initial_parameters = parameter_set(initialization.R,initialization.Z,ai=initialization.ai,ai_abs_bound=initialization.ai_abs_bound,luminosities=initialization.luminosities)

    initializer_dict = {'Z':initialization.Z,'A':initialization.A,'R':initialization.R,'N':initialization.N,'xi_ini':initial_parameters.get_xi(),'ai_ini':initial_parameters.get_ai(),'ai_abs_bounds':initial_parameters.ai_abs_bound,
                        'luminosities_ini':initialization.luminosities}
    
    test_dict = {**settings_dict,**initializer_dict}
    visible_keys = ['Z','A','R','N','datasets']
    tracked_keys = list(test_dict.keys())
    
    if minimizer_args.get('load_best_fit', True):
        multiple_result_dicts = pickle_load_all_results_dicts_R_N(initialization.Z,initialization.A,initialization.R,initialization.N,settings_dict['datasets'])
        num_saved_fits=len(list(multiple_result_dicts.keys()))
        if num_saved_fits==0:
            print('No saved fit with these initial values found (R='+str(initialization.R)+',N='+str(initialization.N)+')')
            loaded_results_dict = None
        elif num_saved_fits>1:
            print('Multiple saved fits with these initial values found (R='+str(initialization.R)+',N='+str(initialization.N)+').',num_saved_fits)
            best_redchisq = np.inf
            best_key = None
            for key in multiple_result_dicts:
                redchisq=multiple_result_dicts[key].get('redchisq',np.inf)
                if redchisq < best_redchisq:
                    best_redchisq=redchisq
                    best_key=key
            print('Chosen fit with redchisq =',best_redchisq, '(R='+str(initialization.R)+',N='+str(initialization.N)+').')
            loaded_results_dict = multiple_result_dicts[best_key]
        else:
            print('Fit with these initial values found (R='+str(initialization.R)+',N='+str(initialization.N)+').')
            loaded_results_dict = list(multiple_result_dicts.values())[0]
    else:
        loaded_results_dict = pickle_load_result_dict(test_dict,tracked_keys,visible_keys,verbose=verbose)
                

    def find_luminosity_index(Data_name,Energy,luminosities):
        array_index=0
        for data_name in datasets:
            if datasets[data_name].get('fit_luminosities','n')=='y':
                for energy in datasets[data_name]['luminosities']:
                    if Data_name==data_name and Energy==energy:
                        return array_index
                    elif array_index<len(luminosities)-1:
                        array_index+=1
                    else:
                        raise ValueError("Fit-parameter not found in luminosities vector")
        
        raise ValueError("Fit-parameter not found in luminosities vector")
                    
    def update_datasets_with_luminosities(luminosities):
        array_index=0
        for data_name in datasets:
            if datasets[data_name].get('fit_luminosities','n')=='y':
                for energy in datasets[data_name]['luminosities']:
                    datasets[data_name]['luminosities'][energy]=luminosities[array_index]
                    array_index+=1
    
    def adjust_arguements(measure_key):
        if measure_key in datasets.keys() and datasets[measure_key].get('fit_luminosities','n')=='y': 
            return (current_nucleus,datasets[measure_key]['luminosities'])
        else:
            return (current_nucleus,)
        
    if (loaded_results_dict is None) or renew:
    
        measures = construct_measures(initialization.Z,initialization.A,datasets,barrett_moment_keys,monotonous_decrease_precision,cross_section_args)
        
        global loss_eval
        loss_eval = 0 
        # define loss function
        def loss_function(fit_params):
            global loss_eval
            loss_eval+=1
            parameters = parameter_set(initialization.R,initialization.Z,xi=fit_params[:initialization.N-1],luminosities=fit_params[initialization.N-1:],ai_abs_bound=initialization.ai_abs_bound)
            current_nucleus.update_ai(parameters.get_ai())
            
            update_datasets_with_luminosities(parameters.luminosities)

            loss=0
            for measures_key in measures:
                loss += measures[measures_key].loss(*adjust_arguements(measures_key))
            if loss_eval%10==0:
                print("Loss (R="+str(current_nucleus.R)+",N="+str(current_nucleus.N_a)+",eval:"+str(loss_eval)+") =",loss)
            return loss

        off_diagonal_covariance=False
        for key in measures:
            off_diagonal_covariance+=measures[key].off_diagonal_covariance

        params_initial = initial_parameters.get_params()
        xi_bounds = len(initial_parameters.get_xi())*[(numdifftools_step,1-numdifftools_step)]
        luminosities_bounds=len(initialization.luminosities)*[(0,np.inf)]

        # Restrict luminosities
        for data_name in datasets:
            if datasets[data_name].get('fit_luminosities','n')=='y' and 'luminosities_bounds' in datasets[data_name]:
                for energy in datasets[data_name]['luminosities_bounds']:
                    luminosities_bounds[find_luminosity_index(data_name,energy,luminosities_bounds)]=datasets[data_name]['luminosities_bounds'][energy]

        bounds_params=xi_bounds+luminosities_bounds
        
        current_nucleus = copy.deepcopy(initialization.nucleus)
        
        converged=False
        rand = 1
        while not converged and rand > 0:
            for key in measures:
                measures[key].set_cov(*adjust_arguements(key))    
            print('Starting current fit step (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+') with loss =',loss_function(params_initial))
            result = minimize(loss_function,params_initial,bounds=bounds_params,**minimizer_args)
            print('Finished current fit step (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+') with loss =',result.fun)
            
            if not off_diagonal_covariance:
                converged=True
            else:
                xi_diff = result.x[:initialization.N-1] - params_initial[:initialization.N-1]
                lum_diff= (result.x[initialization.N-1:] - params_initial[initialization.N-1:])/(params_initial[initialization.N-1:])

                if np.all(np.abs(xi_diff) < xi_diff_convergence_limit) and np.all(np.abs(lum_diff) < 5*xi_diff_convergence_limit):
                    converged=True
                else:
                    if lum_diff.size>0:
                        print('Not converged (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+'): x_f-x_i =', xi_diff, '(lum_f-lum_i)/lum_i:', lum_diff)
                    else:
                        print('Not converged (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+'): x_f-x_i =', xi_diff)
                    
            params_initial = result.x
            parameters = parameter_set(initialization.R,initialization.Z,xi=result.x[:initialization.N-1],luminosities=result.x[initialization.N-1:],ai_abs_bound=initialization.ai_abs_bound)
            current_nucleus.update_ai(parameters.get_ai())
            update_datasets_with_luminosities(parameters.luminosities)
            rand = 1.0 + np.random.rand()
        print('Finished fit (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+'), Calculating Hessian')
        
        Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
        hessian = Hessian_function(result.x)
        try:
            hessian_inv = inv(hessian)
        except np.linalg.LinAlgError:
            print("(R="+str(current_nucleus.R)+",N="+str(current_nucleus.N_a)+")")
            print('Hessian is singular')
            print('result is:',result.x)
            print('Hessian is:',hessian)
            results_dict ={'error':'Hessian is singular'}
            return results_dict
        
        covariance_params = 2*hessian_inv
        
        print('Finished, Constructing results dictionary (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+')')
        
        out_parameters = parameter_set(initialization.R,initialization.Z,xi=result.x[:initialization.N-1],ai_abs_bound=initialization.ai_abs_bound,luminosities=result.x[initialization.N-1:])
        out_parameters.update_cov_xi_then_cov_ai(covariance_params[:len(xi_bounds),:len(xi_bounds)])
        out_parameters.set_ai_tilde_from_xi()
        out_parameters.set_ai_from_ai_tilde()
        update_datasets_with_luminosities(out_parameters.luminosities)

        parameters_results={'xi':out_parameters.get_xi(),'ai':out_parameters.get_ai(),'dxi_stat':np.sqrt(out_parameters.cov_xi.diagonal()),'dai_stat':np.sqrt(out_parameters.cov_ai.diagonal()),'cov_xi_stat':out_parameters.cov_xi,'cov_ai_stat':out_parameters.cov_ai,
                            'luminosities':{data_name:datasets[data_name]['luminosities'] for data_name in datasets if datasets[data_name].get('fit_luminosities','n')=='y'}, 
                            'cov_luminosities':covariance_params[len(xi_bounds):,len(xi_bounds):] if len(luminosities_bounds)>0 else np.array([])}
        
        # calc statistical measures
        chisq, resid, sample_size, dof, redchisq, p_val = {}, {}, {}, {}, {}, {}
        chisq['total'], sample_size['total'] = 0, 0
        for measures_key in measures:
            resid[measures_key] = measures[measures_key].residual(*adjust_arguements(measures_key))
            chisq[measures_key] = measures[measures_key].loss(*adjust_arguements(measures_key))
            sample_size[measures_key] = len(resid[measures_key])
            if sample_size[measures_key] > out_parameters.N_x:
                dof[measures_key] = sample_size[measures_key] - out_parameters.N_x
                redchisq[measures_key] = chisq[measures_key]/dof[measures_key]
                p_val[measures_key] = chi2.sf(chisq[measures_key],dof[measures_key])
            chisq['total'] += chisq[measures_key]
            sample_size['total'] += sample_size[measures_key]
        dof['total'] = sample_size['total'] - out_parameters.N_x
        redchisq['total'] =  chisq['total']/dof['total']
        p_val['total'] = chi2.sf(chisq['total'],dof['total'])    
        statistics_dict = {'chisq':chisq,'redchisq':redchisq,'p_val':p_val,'dof':dof,'sample_size':sample_size,'resid':resid}
        
        statistics_results={'chisq':chisq['total'],'redchisq':redchisq['total'],'p_val':p_val['total'],'dof':dof['total'],'sample_size':sample_size['total'],'nfev':loss_eval,'statistics_dict':statistics_dict}
        
        values_results={}
        for measures_key in measures:
            values_results['x_'+measures_key]=measures[measures_key].x_data
            values_results['y_'+measures_key]=measures[measures_key].test_function_eval(*adjust_arguements(measures_key))
        
        # calc radius and barrett moment uncertainties
        r_ch = current_nucleus.charge_radius
        dr_ch = np.sqrt(np.einsum('i,ij,j->',current_nucleus.charge_radius_jacobian,out_parameters.cov_ai,current_nucleus.charge_radius_jacobian))
        
        radius_dict={'r_ch':r_ch,'dr_ch_stat':dr_ch}
        
        barrett_dict={}
        for barrett_moment_key in barrett_moment_keys:
            k, alpha = measures['barrett_moment_'+barrett_moment_key].x_data
            barrett = current_nucleus.barrett_moment(k,alpha)
            barrett_jacob = current_nucleus.barrett_moment_jacobian(k,alpha)           
            dbarrett = np.sqrt(np.einsum('i,ij,j->',barrett_jacob,out_parameters.cov_ai,barrett_jacob))
            barrett_dict={**barrett_dict,'k_'+barrett_moment_key:k,'alpha_'+barrett_moment_key:alpha,'barrett_'+barrett_moment_key:barrett,'dbarrett_'+barrett_moment_key:dbarrett}
            
        results_dict={**settings_dict,**initializer_dict,**statistics_results,**parameters_results,**values_results,**radius_dict,**barrett_dict}
        
        print('Dumping results (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+')')
        
        pickle_dump_result_dict(results_dict,tracked_keys,visible_keys,overwrite=renew)
    
    else:
        print('Fit with these initial values was already calculated before (R='+str(initialization.R)+',N='+str(initialization.N)+')')
        
        results_dict = loaded_results_dict
    
    return results_dict 

def construct_measures(Z,A,datasets:dict,barrett_moment_keys=[],monotonous_decrease_precision=np.inf,cross_section_args={}):
    
    barrett_moment_constraint = (len(barrett_moment_keys)>0) # not (barrett_moment_key is None) #
    monotonous_decrease_constraint = (monotonous_decrease_precision<np.inf)
    
    measures={}
    
    for data_name in datasets:
        fit_luminosities = datasets[data_name].get('fit_luminosities','n')
        dataset, corr_stat, corr_syst = load_dataset(data_name,Z,A,verbose=False)    
        dy_stat, dy_syst = dataset[:,3], dataset[:,4]
        datasets[data_name]['x_data'] = dataset[:,(0,1)]
        datasets[data_name]['y_data'] = dataset[:,2]
        datasets[data_name]['cov_stat_data'] = np.einsum('i,ij,j->ij',dy_stat,corr_stat,dy_stat)
        datasets[data_name]['cov_syst_data'] = np.einsum('i,ij,j->ij',dy_syst,corr_syst,dy_syst)

        def cross_section(energy_and_theta,nucleus):
            energies = energy_and_theta[:,0]
            thetas = energy_and_theta[:,1]
            cross_section = np.zeros(len(energies))
            for energy in np.unique(energies):
                mask = (energy==energies)
                cross_section[mask] = crosssection_lepton_nucleus_scattering(energy,thetas[mask],nucleus,**cross_section_args)*constants.hc**2
            return cross_section
        
        def reaction_rate(energy_and_theta,nucleus,luminosities:dict):
            energies = energy_and_theta[:,0]
            thetas = energy_and_theta[:,1]
            reaction_rate = np.zeros(len(energies))
            for energy in np.unique(energies):
                mask = (energy==energies)
                reaction_rate[mask] = luminosities[energy]*crosssection_lepton_nucleus_scattering(energy,thetas[mask],nucleus,**cross_section_args)*constants.hc**2
            return reaction_rate
        
        if fit_luminosities=='y':
            measures[data_name] = minimization_measures(reaction_rate,datasets[data_name]['x_data'],datasets[data_name]['y_data'],datasets[data_name]['cov_stat_data'],datasets[data_name]['cov_syst_data'])
        else:
            measures[data_name] = minimization_measures(cross_section,datasets[data_name]['x_data'],datasets[data_name]['y_data'],datasets[data_name]['cov_stat_data'],datasets[data_name]['cov_syst_data'])

    
    if barrett_moment_constraint:
        barrett_moments = {}
        for barrett_moment_key in barrett_moment_keys:
            barrett_moments['barrett_moment_'+barrett_moment_key]={}
            barrett_dict = load_barrett_moment(barrett_moment_key,Z,A,verbose=False)
            barrett_moments['barrett_moment_'+barrett_moment_key]['x_data'] = (barrett_dict["k"],barrett_dict["alpha"])
            barrett_moments['barrett_moment_'+barrett_moment_key]['y_data'] = barrett_dict["barrett"]
            barrett_moments['barrett_moment_'+barrett_moment_key]['cov_stat_data'] = barrett_dict["dbarrett"]**2
            barrett_moments['barrett_moment_'+barrett_moment_key]['cov_syst_data'] = 0
            def barrett_moment(k_alpha_tuple,nucleus):
                return np.atleast_1d(nucleus.barrett_moment(*k_alpha_tuple))
            measures['barrett_moment_'+barrett_moment_key]=minimization_measures(barrett_moment,**barrett_moments['barrett_moment_'+barrett_moment_key])
        
    if monotonous_decrease_constraint:
        def positive_slope_component_to_radius_squared(_,nucleus):
            integrand = lambda r: -4*pi*r**5/5*nucleus.dcharge_density_dr(r)/nucleus.Z
            integrand_positive_slope = lambda r: np.where(integrand(r)<0,integrand(r),0) 
            positive_slope_component = quad(integrand_positive_slope,0,nucleus.R,limit=1000)[0]
            return np.atleast_1d(positive_slope_component)
        measures['monotonous_decrease']=minimization_measures(positive_slope_component_to_radius_squared,x_data=np.nan,y_data=0,cov_stat_data=monotonous_decrease_precision**2,cov_syst_data=0)    
    
    return measures




def recalc_covariance(fit_result:dict,numdifftools_step=1.e-4,cross_section_args={}):

    datasets_keys, barrett_moment_keys, monotonous_decrease_precision = fit_result['datasets'] , fit_result['datasets_barrett_moment'], fit_result['monotonous_decrease_precision']

    Z, A, R, ai_abs_bounds = fit_result['Z'], fit_result['A'], fit_result['R'], fit_result['ai_abs_bounds']
    ai_bestfit = fit_result['ai']
    xi_bestfit = fit_result['xi']

    measures = construct_measures(Z,A,datasets_keys,barrett_moment_keys,monotonous_decrease_precision,cross_section_args)
    
    def loss_function(xi):
        parameters = parameter_set(R,Z,xi=xi,ai_abs_bound=ai_abs_bounds)
        current_nucleus.update_ai(parameters.get_ai())
        loss=0
        for dataset_key in measures:
            loss += measures[dataset_key].loss(current_nucleus)
        return loss

    current_nucleus = nucleus('FB_stat_calc',Z,A,ai=ai_bestfit,R=R)

    for key in measures:
        measures[key].set_cov(current_nucleus)

    Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
    hessian = Hessian_function(xi_bestfit)
    hessian_inv = inv(hessian)
    covariance_xi = 2*hessian_inv

    return covariance_xi

def overwrite_statistical_uncertainties(fit_result, covariance_xi, barrett_moment_keys=[]):
    
    Z, A, R, ai_abs_bounds = fit_result['Z'], fit_result['A'], fit_result['R'], fit_result['ai_abs_bounds']
    ai_fit = fit_result['ai']
    xi_fit = fit_result['xi']
    
    current_nucleus = nucleus('FB_stat_calc',Z,A,ai=ai_fit,R=R)
    
    out_parameters = parameter_set(R,Z,xi=xi_fit,ai_abs_bound=ai_abs_bounds)
    out_parameters.update_cov_xi_then_cov_ai(covariance_xi)
    out_parameters.set_ai_tilde_from_xi()
    out_parameters.set_ai_from_ai_tilde()
    
    parameters_results={'xi':out_parameters.get_xi(),'ai':out_parameters.get_ai(),'dxi_stat':np.sqrt(out_parameters.cov_xi.diagonal()),'dai_stat':np.sqrt(out_parameters.cov_ai.diagonal()),'cov_xi_stat':out_parameters.cov_xi,'cov_ai_stat':out_parameters.cov_ai}
    
    # calc radius and barrett moment uncertainties
    r_ch = current_nucleus.charge_radius
    dr_ch = np.sqrt(np.einsum('i,ij,j->',current_nucleus.charge_radius_jacobian,out_parameters.cov_ai,current_nucleus.charge_radius_jacobian))
    
    radius_dict={'r_ch':r_ch,'dr_ch_stat':dr_ch}
    
    barrett_dict={}
    for barrett_moment_key in barrett_moment_keys:
        k, alpha = fit_result['k_'+barrett_moment_key], fit_result['alpha_'+barrett_moment_key]
        barrett = current_nucleus.barrett_moment(k,alpha)
        barrett_jacob = current_nucleus.barrett_moment_jacobian(k,alpha)           
        dbarrett = np.sqrt(np.einsum('i,ij,j->',barrett_jacob,out_parameters.cov_ai,barrett_jacob))
        barrett_dict={**barrett_dict,'k_'+barrett_moment_key:k,'alpha_'+barrett_moment_key:alpha,'barrett_'+barrett_moment_key:barrett,'dbarrett_'+barrett_moment_key:dbarrett}
    
    fit_result={**fit_result,**parameters_results,**radius_dict,**barrett_dict}
    
    return fit_result    
