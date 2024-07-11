"""
-------------------------------------------------------------------------------
                        GRAPH: OVERLAY OF THE CURVES
-------------------------------------------------------------------------------
Author: Claire M. based on Anass Akrim's work
"""


"""
------------------------------- IMPORTS ---------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)



"""
------------------------------ PARAMETERS -------------------------------------
"""

nb_gauges = 3 # if the number of gauges change, it will be necessary to modify the code to plot more/less curves
id_to_plot = 1
delta_k = 500
h_size = 8


path_dico_temp = 'data/dico_temp.txt'
path_data_pure = 'data/data_train_pure'
path_data_noisy = 'data/data_train'
path_data_forecasted = 'data/data_forecasted'

exogenous_variable = True



"""
------------------------------- FUNCTIONS -------------------------------------
"""

def graphs(pure = True, noisy = True, forecasted = True, id_ploted = id_to_plot, h = h_size, delta_k_param=delta_k, file_dico = path_dico_temp, file_pure=path_data_pure, file_noisy=path_data_noisy, file_forecast=path_data_forecasted):
    """
    Generate the graph with the pure, noisy and/or forecasted data of one structure
    
    Parameters
    ----------
    pure : boolean
        True to show the pure data on the graph
    noisy : boolean
        True to show the noisy data on the graph
    forecasted : boolean
        True to show the forecasted data on the graph
    id_ploted : int
        ID of the structure to be ploted
    h : int
        number of delta_k in the forecast (0 if labeled/before restructuration)
    delta_k : int
        Size of the steps between recoveries of data
    file_dico : String
        file path to the dictionnary with the RUL_max of each structure from the dataset
    file_pure : String
        file path to the pure data
    file_noisy : String
        file path to the noisy data
    file_forecasted : String
        file path to the forecasted data

    Returns
    -------
    Plot of the graph with the 3 curves of each gauge from pure/noisy/forecasted data
    """
    
    plt.figure(figsize=(20, 12)) # Creation of the figure and setting the size
    
    # FOR THE PURE DATA:
    if pure:
        data_train_pure = pd.read_pickle(file_pure)    # Retrieving the dataFrame from the pure data file
        print(data_train_pure.loc[data_train_pure.ID==id_ploted])   # Print the dataFrame for the ID chosen with id_ploted
        # the following lines plot the 3 curves of the 3 gauges of the chosen ID
        plt.plot(data_train_pure.loc[data_train_pure.ID==id_ploted].cycle.values,data_train_pure.loc[data_train_pure.ID==id_ploted,'gauge1'], linewidth = 5, color = 'darkgreen', alpha = 0.25, label = 'Gauge placed on $(x_1,y_1) = (03,14) ~ mm$')
        plt.plot(data_train_pure.loc[data_train_pure.ID==id_ploted].cycle.values,data_train_pure.loc[data_train_pure.ID==id_ploted,'gauge2'], linewidth = 5, color = 'darkblue', alpha = 0.25, label = 'Gauge placed on $(x_2,y_2) = (14,14)~ mm$')
        plt.plot(data_train_pure.loc[data_train_pure.ID==id_ploted].cycle.values,data_train_pure.loc[data_train_pure.ID==id_ploted,'gauge3'], linewidth = 5, color = 'darkred', alpha = 0.25, label = 'Gauge placed on $(x_3,y_3) = (25,14)~ mm$')

    # FOR THE NOISY DATA:    
    if noisy:
        data_train = pd.read_pickle(file_noisy) 
        print(data_train.loc[data_train.ID==id_ploted])
        plt.plot(data_train.loc[data_train.ID==id_ploted].cycle.values,data_train.loc[data_train.ID==id_ploted, 'gauge1'], linewidth = 1, color = 'darkgreen', alpha = 0.5, label = 'Noisy gauge placed on $(x_1,y_1) = (03,14) ~ mm$')
        plt.plot(data_train.loc[data_train.ID==id_ploted].cycle.values,data_train.loc[data_train.ID==id_ploted,'gauge2'], linewidth = 1, color = 'darkblue', alpha = 0.5, label = 'Noisy gauge placed on $(x_2,y_2) = (14,14)~ mm$')
        plt.plot(data_train.loc[data_train.ID==id_ploted].cycle.values,data_train.loc[data_train.ID==id_ploted,'gauge3'], linewidth = 1, color = 'darkred', alpha = 0.5, label = 'Noisy gauge placed on $(x_3,y_3) = (25,14)~ mm$')
    
    # FOR THE FORECASTED DATA (AFTER RESTRUCTURATION TO THE ORIGINAL FORMAT):
    if forecasted:
        data_forecasted = pd.read_pickle(file_forecast)
        print(data_forecasted.loc[data_forecasted.ID==id_ploted])
        plt.plot(data_forecasted.loc[data_forecasted.ID==id_ploted].cycle.values,data_forecasted.loc[data_forecasted.ID==id_ploted,'gauge1'], linewidth = 2, color = 'green', label = 'Forecasted gauge placed on $(x_1,y_1) = (03,14) ~ mm$')
        plt.plot(data_forecasted.loc[data_forecasted.ID==id_ploted].cycle.values,data_forecasted.loc[data_forecasted.ID==id_ploted,'gauge2'], linewidth = 2, color = 'blue', label = 'Forecasted gauge placed on $(x_2,y_2) = (14,14)~ mm$')
        plt.plot(data_forecasted.loc[data_forecasted.ID==id_ploted].cycle.values,data_forecasted.loc[data_forecasted.ID==id_ploted,'gauge3'], linewidth = 2, color = 'red',label = 'Forecasted gauge placed on $(x_3,y_3) = (25,14)~ mm$')
   
    # PLOT A VERTICAL LINE TO REPRESENT THE INSTANT WHERE THE DATA IS UNLABELED (PERCENT OF REMOVAL)
    if h != 0:
        RUL_max = data_forecasted.loc[h-1, 'cycle']
        x_percent = RUL_max-(h-1)*delta_k 
        plt.axvline(x=x_percent, ls='--', color='grey', lw = 3, label='Break time - '+str(h*delta_k_param)+' cycles') # plot the vertical line
    
    # REST OF THE GRAPH
    plt.grid() # Adding a grid to the graph
    plt.legend(fontsize="20") # Adding a legend, setting the fontsize
    plt.xlabel('Cycles (delta_k =' + str(delta_k_param) + ')', fontsize="20") # Labelling the x-axis, specifying the step sizes
    plt.ylabel('Strains', fontsize="20") # Labelling the y-axis
    
    plt.show()


    
"""
 ------------------------ IF __NAME__ = '__MAIN__' ----------------------------
"""

if __name__ == "__main__":
    if exogenous_variable:
        path_data_forecasted+=str(id_to_plot)
        graphs(file_forecast = path_data_forecasted)
    else:
        graphs()

