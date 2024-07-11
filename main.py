"""
-------------------------------------------------------------------------------
                        MAIN PAGE TO CREAT THE DATASETS
-------------------------------------------------------------------------------
Author: Claire M. based on Anass Akrim's work
"""


"""
----------------------------------- IMPORTS -----------------------------------
"""

import argparse
import pickle
import numpy as np
import warnings 
import os
import utils
warnings.filterwarnings("ignore")


"""
----------------------------- SETTING PARAMETERS ------------------------------
------------------- IF NEED BE: CHANGE THEM IN THIS SECTION -------------------
"""

def training_args():
    parser=argparse.ArgumentParser(description='Generate_Set')
    
    # FOLDER PATH
    parser.add_argument('--folder_data', default='data', type=str, help="Set the folder path")
 
    # ELASTIC PARAMETERS
    parser.add_argument('--E', default=71.7e9, type=float, help="Young's modulus of the structure")
    parser.add_argument('--nu', default=0.33, type=float, help="Poisson's ratio of the structure")
    
    # STRAIN FIELD PARAMETERS
    parser.add_argument('--K_IC', default=19.7, type=float, help="Fracture toughness of the structure")
    parser.add_argument('--sigma_inf', default=78.6e6, type=float, help="Maximum stress intensity")
    
    # STRAIN GAUGES
    parser.add_argument('--nb_gauges', default=3, type=int, help="Number of gauges")
    parser.add_argument('--x_gauge', default=(0.003, 0.014, 0.025), nargs='+', type=float, help="x position of the gauges placed")
    parser.add_argument('--y_gauge', default=(0.014, 0.014, 0.014), nargs='+', type=float, help="y position of the gauges placed")
    parser.add_argument('--theta_gauges', default=45, type=float, help="Angle of the gauges placed")
  
    # CRACK INITIALISATION PARAMETERS
    parser.add_argument('--a_0_mean', default=0.001, type=float, help="Initial half crack length mean in [m]")
    parser.add_argument('--C_mean', default=1e-10, type=float, help="C mean")
    parser.add_argument('--m_mean', default=3.5, type=float, help="m mean")
    parser.add_argument('--m_std', default=0.125, type=float, help="m std") 
    parser.add_argument('--noise_std', default=1e-2, type=float, help="Gaussian white noise std")
    
    # SAMPLING SIZES
    parser.add_argument('--n_train', default=10, type=int, help="Number of training structures")

    # OTHER PARAMETERS
    parser.add_argument('--delta_k', default=500, type=int, help="Data collection interval")
    parser.add_argument('--lb_star', default=0.33, type=float, help="lower boundary (used for generating t* for the test set)") 
    parser.add_argument('--ub_star', default=0.95, type=float, help="upper boundary (used for generating t* for the test set)")

    args=parser.parse_args()
    return args



"""
------------------------- GLOBALISING THE PARAMETERS --------------------------
"""

args = training_args()

# SET ELASTIC PARAMETERS
E = args.E
nu = args.nu

# SET STRAIN FIELD PARAMETERS
K_IC = args.K_IC  
sigma_inf = args.sigma_inf 
sigma_0 = 0 # Minimum load in MPa
delta_sigma = sigma_inf-sigma_0 # Difference between maximum and minimum load in MPa
a_crit = (K_IC/(delta_sigma*1e-6*np.sqrt(np.pi)))**2  # (half) crack size at failure time T

# SET STRAIN GAUGES 
nb_gauges = args.nb_gauges
x_gauge = args.x_gauge
y_gauge = args.y_gauge
theta_gauge = args.theta_gauges

# SET CRACK INITIALISATION PARAMETERS
a_0_mean = args.a_0_mean
a_0_std = a_0_mean*0.125  # Standard deviation of a_0 corresponding to a CoV of a0_mean, here CoV = 0.125
C_mean = args.C_mean # Value of Paris law constant C representing its log mean (exp(mean(log(C))))
C_std = C_mean*(8e3-1)/(2+2*8e3) # Ratio 95%
m_mean = args.m_mean # Paris law exponent
m_std = args.m_std # Standard deviation of Paris law exponent (1/1.96, correspondig to having 95% confidence interval at +- 1
noise_std = args.noise_std

# SET SAMPLING SIZES
n_train = args.n_train # Number of structures for training

# SET OTHER PARAMETERS
thinning = args.delta_k # Take values only every `thinning` cycles
lb_tstar = args.lb_star # Lower boundary
ub_tstar = args.ub_star # Upper boundary



"""
 -------------------------- IF __NAME__ = '__MAIN__' --------------------------
"""

if __name__ == "__main__":
    
    # CREATE FOLDER
    print('Create folder...')
    folder_data = args.folder_data
    os.mkdir(folder_data)
    print('Done. \n')
    
    # SAVE THE ARGUMENTS IN .TXT
    print('Save the args...')
    f = open(folder_data+"/args.txt", "w+")
    f.write(str(args))
    f.close()
    print('Done. \n')
    
    # GENERATE THE DATASET
    print('Generate datasets...')
    
    # TRAINING SET
    if n_train >= 1 : 
        print('Training set:')
        training_set, training_set_pure = utils.gen_dataset('train', x_gauge, y_gauge, theta_gauge, delta_sigma, E, nu, a_0_mean, a_0_std, a_crit, C_mean, C_std, m_mean, m_std, n_train, thinning, noise_std)
        training_set_pure.to_pickle(folder_data + '/training_set_pure', protocol = pickle.HIGHEST_PROTOCOL) # Saving the pure training set
        training_set.to_pickle(folder_data + '/training_set', protocol = pickle.HIGHEST_PROTOCOL) # Saving the noisy training set
    
    print("Done. \n")

    # STRUCTURING INTO DATAFRAME
    print("Structuring the datasets...")
    cols = ['ID', 'cycle'] +  ['gauge' + str(i+1) for i in range(nb_gauges)] +  ['RUL'] # Set columns for the datasets

    # TRAINING SET
    if n_train >= 1 :
        print('Training set pure:')
        utils.build_dataset(training_set_pure, cols, folder_data, folder_data + '/data_train_pure', nb_gauges = nb_gauges, thinning = thinning)
        
        print('Training set noisy:')
        utils.build_dataset(training_set, cols, folder_data, folder_data + '/data_train', nb_gauges = nb_gauges, thinning = thinning)
   
    print(f'Datasets created and saved in the folder "{folder_data}".')

