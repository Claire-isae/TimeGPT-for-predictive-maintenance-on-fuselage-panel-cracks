"""
-------------------------------------------------------------------------------
                        GENERATING CRACK PROPAGATION DATA
-------------------------------------------------------------------------------
Author: Claire M. based on Anass Akrim's work
"""


"""
----------------------------------- IMPORTS -----------------------------------
"""
from pandas import json_normalize
from numba import jit
import pickle
import numpy as np
import scipy.stats
import crack
import pandas as pd
import copy


"""
--------------------------------- FUNCTIONS -----------------------------------
"""

def build_dataset(data, cols, folder_data, file_path, nb_gauges = 3, thinning = 500) :
    """
    For the moment only 3 gauges
    
    Parameters
    ----------
    data : pickle file
        unstructured dataset
    cols : string array
        i.e. columns considered for the structure dataframe 
    folder_data : string
        path to folder in which the informations are stored
    file_path : string
        path of the file we work on, where it is created/used
    nb_gauges : int
        number on considered gauges
    thinning : int
        Provide samples for every ``thinning`` cycle

    Returns
    -------
    NONE
    """

    # CREATING A NEW FILE
    with open(file_path, 'wb'): 
        pass
    
    # DATAFRAME FOR EACH STRUCTURE
    for i in range(data.shape[0]) :      
        print("Structuring loop ", i+1)
        
        # CREATING A TEMPORARY DATAFRAME
        len_max = data.loc[i,'Nb_measures'] # nb of measures for structure i
        nb_cycles = data.loc[i].nb_cycles # nb of cycles for structure i
        temp = pd.DataFrame(np.zeros((len_max,len(cols))), columns =cols, dtype='float').astype({"cycle": int, "ID": int, "RUL": int}) # initialize a sub-dataframe for structure i
        
        # FILLING THE DATAFRAME
        temp.loc[0:len_max,'ID'] = i + 1 # ID of the structure
        temp.loc[0:len_max,'cycle'] = thinning*np.linspace(0, len_max-1, len_max, dtype='int') # liste of the cycles every delta_k
        
        for k in range(nb_gauges) : # raw measures for each gauge, i.e. the input columns
                    temp.loc[0:len_max,'gauge' + str(k+1)] = data.loc[i].strains[0:len_max,k]        
                    
        temp.loc[0:len_max,'RUL'] = np.array(range(nb_cycles,nb_cycles%thinning - thinning, -thinning)) # the output column (RUL)

        # SAVE THE DATAFRAME OF STRUCTURE I IN THE FILE
        with open(file_path, 'ab') as file: # add the dataFrame in the pickle file for temporary storage (list of dataFrames) 
            pickle.dump(temp, file)  

    # CREATING THE PROPER FILE
    df = [] # creat a list where the list of dataFrames will be stored
    with open(file_path, 'rb') as file: # open the file
        try:
            while True:
                value = pickle.load(file)    # recover the data from the pickle file
                df.append(value)     # add it to the list
        except EOFError:
            pass
    
    df_concatenated = pd.concat(df, ignore_index=True)  # concatenate the items of the list (index will be revisited and only one heading)
    df_concatenated.to_pickle(file_path)    # turn it into a DataFrame (format used in the rest of the project)
    
    print('Done. \n')



@jit
def gen_param_sample(a_0_mean, a_0_std, C_mean, C_std, m_mean, m_std):
    """
    Generate joint sample of crack parameters :math:`a_0`, :math:`C`,
    :math:`m`.

    Parameters
    ----------
    a_0_mean : float
        Mean of initial half crack length
    a_0_std : float
        Standard deviation of initial half crack length
    C_mean : float
        Mean of crack parameter :math:`C`
    C_std : float
        Std dev of crack parameter :math:`C`    
    m_mean : float
        Mean of crack parameter :math:`m`
    m_std : float
        Standard deviation onf crack parameter :math:`m`

    Returns
    -------
    a_0, C, m
    """
    
    # TRUNCATED GAUSSIAN LAW FOR THE INITIAL HALF CRACK LENGTH
    a_0_scaled_lower = 0 # Scale lower limit (0) appropriately for scipy; upper limit is infinite at any scale
    a_0 = scipy.stats.truncnorm.rvs(a_0_scaled_lower, np.inf, a_0_mean, a_0_std) # Perform the actual sample

    log_C_mean = np.log(C_mean) - (C_std**2)/2 #ln(E[C]) - Var[C]/2
    log_C_std = np.log(1+ (C_std**2)/(C_mean**2))
    
    C = 0
    rho = -0.996
    m = -1

    cov = np.array(((log_C_std**2, rho * log_C_std * m_std), (rho * log_C_std * m_std, m_std**2)))
    
    while m < 0:
        params = scipy.stats.multivariate_normal.rvs((log_C_mean, m_mean), cov)
        
        C = np.exp(params[0])
        m = params[1]

    return a_0, C, m



def gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu, noise_std):
    """
    Generate strain value for gauge position and half crack length

    ``crack.strain`` provides the full strain state, but strain gauges can
    only deliver a strain measurement in one direction. This routine calculates
    the strain value "seen" by a unidirectional strain gauge from the full
    strain state.

    Parameters
    ----------
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    a : float
         Half crack length
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    noise_std : float
        percentage of the average values of the gauges taken into account for the standard deviation of the white noise

    
    Returns
    -------
    The strain value for the given gauge positions the gauge "sees".
    """
    # RETRIEVE THE STRAINS
    epsilon_11, epsilon_22, epsilon_12  = crack.strain(np.abs(x_gauge), y_gauge, a, delta_sigma, E, nu, noise_std)

    # CALCULATE THE UNIDIRECTIONAL STRAIN
    epsilon_gauges = epsilon_11 * np.cos(theta_gauge)**2 + epsilon_22 * np.sin(theta_gauge)**2 + 2* epsilon_12 * np.cos(theta_gauge) * np.sin(theta_gauge)

    return epsilon_gauges



@jit
def gen_dataset(type_data, x_gauge, y_gauge, theta_gauge, delta_sigma, E, nu, a_0_mean, a_0_std, a_crit, C_mean, C_std, m_mean, m_std, n_samples, thinning, noise_std):
    """
    Generate a dataset for crack propagation measurement data.

    Parameters
    ----------
    type_data : string
        'train', 'val' or 'test' dataset
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    a_0_mean : float
        Mean of initial half crack length
    a_0_std : float
        Standard deviation of initial half crack length
    C_mean : float
        Mean of crack parameter :math:`C`
    C_std : float
        Std dev of crack parameter :math:`C` 
    m_mean : float
        Mean of crack parameter :math:`m`
    m_std : float
        Standard deviation of crack parameter :math:`m`
    n_samples : int
        Number of samples
    thinning : int
        Provide samples for every ``thinning`` cycle

    Returns
    -------
    Dictionary of crack propagation sequence datasets, each containing
    C
      :math:`C` parameter of Paris' law, used for the generation of the
      sequence
    m
      :math:`m` parameter of Paris' law , used for the generation of
      the sequence
    a_0
      Half crack length at the start of the crack propagation
    x_gauges
      :math:`x` positions of the strain gauges
    y_gauges
      :math:`y` positions of the strain gauges
    crack_lengths
      Half crack lengths for every ``thinning`` cycles
    initial_strains
      Strains at initial half crack length
    """
    
    # INITIALISE 2 LISTES: PURE AND NOISY
    sequence_datasets_pure = []
    sequence_datasets = []

    # CREATE DATA FOR EACH STRUCTURE I
    for i in range(n_samples):
        
        a_0, C, m = gen_param_sample(a_0_mean, a_0_std, C_mean, C_std, m_mean, m_std) # Structure i random parameters
        
        k, crack_lengths_numpy, strains_numpy, strains_pure_numpy = gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning, noise_std) # Structure i crack sequence
        
        # SAVING THE PURE DATA
        dataset_pure = {'C': C, 'm': m, 'a_0': a_0, 'x_gauges': (x_gauge), 'y_gauges': (y_gauge), 'crack_lengths': list(crack_lengths_numpy), 'nb_cycles' : k, 'Nb_measures' : k//thinning + 1, 'strains' : strains_pure_numpy}
        sequence_datasets_pure.append(dataset_pure)

        # SAVING THE NOISY DATA    
        dataset = {'C': C, 'm': m, 'a_0': a_0, 'x_gauges': (x_gauge), 'y_gauges': (y_gauge), 'crack_lengths': list(crack_lengths_numpy), 'nb_cycles' : k, 'Nb_measures' : k//thinning + 1, 'strains' : strains_numpy}      
        sequence_datasets.append(dataset)

        print('Created dataset no. {0: d} with {1: d} cycles'.format(i + 1, (strains_pure_numpy.shape[0] - 1)*thinning)) 
       
    return pd.DataFrame.from_dict(json_normalize(sequence_datasets), orient='columns'), pd.DataFrame.from_dict(json_normalize(sequence_datasets_pure), orient='columns')



@jit
def gen_crack_sequence(a_0, a_crit, C, m, E, nu, delta_sigma, x_gauge, y_gauge, theta_gauge, thinning, noise_std):
    """
    Generate one crack propagation sequence from parameters

    Parameters
    ----------
    a_0 : float
         Initial half crack length
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    E : float
         Young's modulus of the material
    nu : float
         Poisson's ratio of the material
    delta_sigma : float
         Difference between maximum and minimum load in the cycle
    x_gauge : array_like
         :math:`x` position of the strain gauge
    y_gauge : array_like
         :math:`y` position of the strain gauge
    theta_gauge : array_like
         Angle of the strain gauge
    thinning : int
         Record strain every ``thinning`` cycles
    noise_std : float
        percentage of the average values of the gauges taken into account for the standard deviation of the white noise

    Returns
    -------
    k : int
        number of cycles
    np.array(crack_lengths) : array
        list of the half crack length
    np.array(strains_noisy) : array
        liste of the values of the gauges NOISY
    np.array(strains) : array
        list of the pure values of the gauges no noise)
    """
    
    # CALCULATE INITIAL STRAINS & STRESS & DISPLACEMENTS
    epsilon_gauges = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a_0, delta_sigma, E, nu, noise_std) 

    # INITIALISE SEQUENCE GENERATION LOOP
    crack_lengths = [a_0] # list of half crack length
    strains = [epsilon_gauges] # list of strains
    k = 0 # initial cycle
    a = a_0 # initial half crack length
    
    # INITIALISE THE VARIABLES FOR THE WHITE NOISE
    somme_moyBruit_1 = 0
    somme_moyBruit_2 = 0
    somme_moyBruit_3 = 0

    # PROPAGATE UNTIL THE CRACK IS a_crit (mm) LONG, AT WHICH POINT IT IS CONSIDERED THAT THE SAMPLE WILL BREAK
    while a < a_crit:
        
        # NEW CYCLE
        k = k + thinning     
        
        # CALCULATE THE ITERATION'S VALUES
        a = crack.length_paris_law(k, delta_sigma, a_0, C, m) # calculate the half crack length
        crack_lengths.append(a)
        
        epsilon_gauges = gen_strain_value_gauge(x_gauge, y_gauge, theta_gauge, a, delta_sigma, E, nu, noise_std) # calculate the strains
        strains.append((epsilon_gauges))
        
        # PREPARING THE WHITE NOISE: SUM UP OF THE GAUGES' VALUES
        somme_moyBruit_1 += epsilon_gauges[0]
        somme_moyBruit_2 += epsilon_gauges[1]
        somme_moyBruit_3 += epsilon_gauges[2]
    
    # PREPARING THE WHITE NOISE:
    nb_cycles = (k//thinning) # calculate the number of cycles
    moyBruit_1 = somme_moyBruit_1 / nb_cycles # calculate the mean value for the gauge 1
    moyBruit_2 = somme_moyBruit_2 / nb_cycles
    moyBruit_3 = somme_moyBruit_3 / nb_cycles
    
    strains_noisy = copy.deepcopy(strains) # save the noise value in a new array

    # ADD THE WHITE NOISE: /!\ THE WHITE NOISE OF THE GAUGES ARE INDEPENDANT 
    for epsilon_gauges in strains_noisy:
        epsilon_gauges[0] = epsilon_gauges[0] + np.random.normal(0, noise_std*moyBruit_1, 1) # adding a gaussian white noise N(0, mean*0.01)
        epsilon_gauges[1] = epsilon_gauges[1] + np.random.normal(0, noise_std*moyBruit_2, 1)
        epsilon_gauges[2] = epsilon_gauges[2] + np.random.normal(0, noise_std*moyBruit_3, 1)

    return k, np.array(crack_lengths), np.array(strains_noisy), np.array(strains)