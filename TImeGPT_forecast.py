"""
-------------------------------------------------------------------------------
                        TIMEGPT BASIC FORECASTING
-------------------------------------------------------------------------------
Author: Claire M.
"""


"""
------------------------------- IMPORTS ---------------------------------------
"""

from nixtla import NixtlaClient
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timezone
import math
import os
import ast

# INSTANTIATE NIXTLACLIENT ACCOUNT (Get your API Key at dashboard.nixtla.io)
nixtla_client = NixtlaClient(api_key = 'COPY_YOUR_API_KEY_HERE')


"""
------------------------------ PARAMETERS -------------------------------------
"""


fine_tuning_steps = 0
h_size = 8
delta_k = 500

folder_path = 'data'
file_data_path = folder_path + '/data_train'
file_data_restructured_for_TimeGPT = folder_path +'/restructured_for_TimeGPT'
file_dico = folder_path + '/dico_temp.txt'
file_data_TimeGPT_forecasting = folder_path +'/TimeGPT_forecasting'
file_data_forecasted_restructured = folder_path + '/data_forecasted'
file_MAPE = folder_path + '/resultats.txt'

"""
 -------------------------------- FUNCTIONS -----------------------------------
"""

def k_to_seconds(k):
    """
    Transform delta_k into a string of seconds
    
    Parameters
    ----------
    delta_k: int
        the number of cycles between two retrieve of data
        
    ----------
    
    Returns
    ----------
    string version with 's' at the end for the forecasting frequency
    """
    
    return str(k)+'s'



"""
 ------------------------------- FORECASTING ----------------------------------
"""

def forecast_timeGPT(df=None, h=h_size, freq=k_to_seconds(delta_k), id_col='unique_id', time_col='ds', target_col='y', X_df=None, finetune_steps=fine_tuning_steps, add_history=False, file_path = file_data_TimeGPT_forecasting, file_path_for_timeGPT=file_data_restructured_for_TimeGPT):
    """
    Generate a dataset for crack propagation measurement data.
    
    Parameters
    ----------
    df: DataFrame
        past data
    h: int
        number of future time steps to be forecasted (horizon)
    freq: string
         frequency of the time series in Pandas format
    id_col : String
         column that identifies each serie
    time_col: string
        column that identifies the datestamp
    target_col: string
        variable to forecast
    X_df: DataFrame
        dataframe with future exogenous
    finetune_steps: int
        number of iterations of training with the data
    add_history: Bool
        if True: retrieves the historical forecast (not only the h time steps of the future forecast)
    file_path: String
        default file path where the forecasted data of TimeGPT will be stored
    file_path_for_timeGPT: String
        default file path to the restructured data to fit TimeGPT's requirements
        
    ----------
    
    Returns
    ----------
    timegpt_fcst_df: DataFrame
        DataFrame of the forecast
    """
    
    if df is None: # default setting made with this code
        df = pd.read_pickle(file_path_for_timeGPT)
        
    # FORECASTING
    timegpt_fcst_df = nixtla_client.forecast(df=df, X_df=X_df, h=h, freq=freq, time_col=time_col, finetune_steps=finetune_steps, target_col=target_col, add_history=add_history)
    timegpt_fcst_df.to_pickle(file_path)
    
    return timegpt_fcst_df



"""
 ----------------------- RESTRUCTURING THE DATAFRAME --------------------------
"""

def restructure_unlabeled_timegpt_df(df=None, h=h_size, file_path_to_data = file_data_path, path_for_restructured=file_data_restructured_for_TimeGPT, path_dico=file_dico, delta_k_loc=delta_k):
    """
    Restructure the DataFrame from the generation of datasets from Claire M. to suit TimeGPT's requirements
    
    Parameters
    ----------
    df : DataFrame
       DataFrame to be restructured
    h : int
        horizon of the forecast (number of steps to forecast) ie: number of steps to remove
    main_folder_path : String
       folder path to store the restructured dataframe      
    file_path_to_data : String
        path to the file with the data
    delta_k_loc : int
        delta_k from the creation of the dataset
   ----------
    
    Returns
    ----------
    restructured_df : DataFrame
        restructured DataFrame, ready to be used in TimeGPT
    dictionnaire_temp:  dictionnary
        unique_id (float) : date of the failure (dateStamp)
    """
    
    if df is None: # default setting made with this code
        df = pd.read_pickle(file_path_to_data)

    cols = ['unique_id', 'ds', 'y'] # initiating the columns for TimeGPT
    dictionnaire_temp={} # creating a dictionnary with the number of cycles for each structure (RUL_max)
    df_temp = df.copy() # creating a deep copy to avoid any modification on the original one
    groups = df_temp.groupby(df_temp.ID) # creating sub-dataframe, grouped by the unique_id
    
    with open(path_for_restructured, 'wb'): # create/overwrite the file with the restructured dataFrame
        pass
 
    restructured_df = pd.DataFrame(None) # intitialize the dataframe

    with open(path_for_restructured, 'ab') as file: # append at the end of the file
        pickle.dump(restructured_df, file)
    
    # RESTRUCTURE THE STRUCTURES OF THE DATASET TO MATCH TIMEGPT'S REQUIREMENTS:
    for i in range(df.loc[df.index.max(), 'ID']): # for structure i
        print("Structuring loop ", i+1)
        
        group = groups.get_group(i+1) # calling the sub-dataframe labeled 'i+1' (ie: the structure with the ID i+1)
        group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        len_max = group.index.max()+1
        len_max_group = len_max*3

        restructured_df = pd.DataFrame(np.zeros((len_max_group,len(cols))), columns =cols, dtype='float').astype({"ds": int}) # initialize a sub-dataframe for structure i with TimeGPT requirements
        
        RUL_max = group.loc[0, 'RUL']
        dictionnaire_temp[i+1] = RUL_max 

        # FILLING THE SUB-DATAFRAME:
        for index, row in group.iterrows():
            # REMOVE THE LAST CYCLES FROM THE DATA / UNLABELLING THE DATASET:
            if row['cycle'] >= RUL_max-h*delta_k_loc :  
                restructured_df=restructured_df.drop(restructured_df.tail((len_max-index)*3).index)
                restructured_df = restructured_df.reset_index(drop=True)
                break
                
            time_cycle = pd.to_datetime(row['cycle'], unit='s') # convert the cycles into dates: YYYY-MM-DD HH:MM:SS at UTC 1970-01-01 00:00:00   
            index = 3*index # *3 for the 3 lines of the 3 gauges
            
            # ADDING THE ELEMENTS IN THE DATAFRAME:
            restructured_df.loc[index,'unique_id']=i+1.1 # for gauge 1, the unique_id is renamed ID.1
            restructured_df.loc[index, 'ds']=time_cycle # fill the time
            restructured_df.loc[index, 'y']=row['gauge1'] # fill the gauge 1 value
            
            restructured_df.loc[index+1, 'unique_id']=i+1.2
            restructured_df.loc[index+1, 'ds']=time_cycle
            restructured_df.loc[index+1, 'y']=row['gauge2']
            
            restructured_df.loc[index+2, 'unique_id']=i+1.3
            restructured_df.loc[index+2, 'ds']=time_cycle
            restructured_df.loc[index+2, 'y']=row['gauge3']
            
        groups_gauges = restructured_df.groupby(restructured_df.unique_id) # creating sub-sub-dataframe, grouped by the unique_id

        for unique_id in [i+1.1, i+1.2, i+1.3] :          
            group_gauge = groups_gauges.get_group(unique_id) # calling the sub-sub-dataframe labeled 'id_gauge'
            group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe
            
            with open(path_for_restructured, 'ab') as file: # append at the end of the file
                pickle.dump(group_gauge, file)
    
    # CREATING THE PROPER FILE TO REMOVE THE HEADING:
    db = [] # creat a list where the information will be stored
    with open(path_for_restructured, 'rb') as file:
        try:
            while True:
                value = pickle.load(file)    # recover the data from the pickle file
                db.append(value)     # add it to the list
        except EOFError:
            pass
        
    os.remove(path_for_restructured) # delete the file because the presentation is not adequate
    
    restructured_df = pd.concat(db, ignore_index=True)  # concatenate the items of the list (index will be revisited, no [XX rows x 6 columns], and no heading)
    restructured_df.to_pickle(path_for_restructured)    # turn it into a DataFrame (format used in the rest of the project)
    
    # CREATING THE FILE FOR THE DICTIONNARY
    f = open(path_dico, 'w')
    f.write(str(dictionnaire_temp))
    f.close()
    
    return restructured_df, dictionnaire_temp



def restructure_desirable_original_df(df= None, file_path_timeGPT=file_data_TimeGPT_forecasting, file_path=file_data_forecasted_restructured):
    """
    Restructure the DataFrame to its original format
    
    Parameters
    ----------
   df : DataFrame
       DataFrame to be restructured
   file_path_timeGPT : String
       file path to the forecasted pickle file of TimeGPT
   file_path : String
       path to store the restructured file pickle
   ----------
    
    Returns
    ----------
    format_final_df: DataFrame
        restructured DataFrame, with [index, 'ID', 'cycle', 'gauge1', 'gauge1', 'gauge3', 'RUL'], RUL=0 (not evaluated by TimeGPT)
    """
    
    if df is None: # default setting made with this code
        temp_df = pd.read_pickle(file_path_timeGPT)
    else:
        temp_df = df # creating a copy of df, to avoid modifying df outside of the function

    cols_temp = ['ID', 'cycle', 'gauge1', 'gauge2', 'gauge3', 'RUL'] # creating 3 columns 'cycle' to check if the time/cycles are identical
    groups = temp_df.groupby(temp_df.unique_id) # creating sub-dataframe, grouped by the unique_id
    len_max = temp_df.index.max()+1
              
    with open(file_path, 'wb'): # create/overwrite the file with the restructured dataFrame
        pass
 
    db = pd.DataFrame(None) #intitialize the dataframe

    with open(file_path, 'ab') as file: # append at the end of the file
        pickle.dump(db, file)
    
    # CREATING A DICTIONNARY WITH ALL THE UNIQUE ID'S (1.1, 1.2, 1.3, 2.1, ETC)
    dictionnaire_id={} 
    for i in range(math.floor(temp_df.loc[len_max-1, 'unique_id'])):
        dictionnaire_id[i+1]=[i+1.1, i+1.2, i+1.3] 
        
    # RESTRUCTURE THE DATAFRAME 
    for unique_id in dictionnaire_id.keys(): # for every ID (1, 2, 3, etc)
        print('restructuring loop ', unique_id)
        group = groups.get_group(unique_id+0.1) # calling the sub-dataframe labeled 'id_gauge'
        group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        len_max_group = group.index.max()+1
        temp_format_df = pd.DataFrame(np.zeros((len_max_group,len(cols_temp))), columns =cols_temp, dtype='float').astype({"cycle": int, "ID": int, "RUL": int})
        
        for id_gauge in dictionnaire_id[unique_id]: # for every unique_ID of the ID (.1, .2, and .3)
            group = groups.get_group(id_gauge) # calling the sub-dataframe labeled 'id_gauge'
            group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
            len_max_group = group.index.max()+1   
            
            # FILLING THE TEMPRARY DATAFRAME WITH THE ELEMENTS OF THE 3 GAUGES
            for j in range(len_max_group):  
                if j > temp_format_df.index.max():
                    break
                elif id_gauge%1 < 0.15: # if gauge1, %1 is usually reel but not rationnal, therefore the use of < 0.15, 0.25
                    temp_format_df.loc[j, 'ID'] = unique_id
                    temp_format_df.loc[j, 'cycle'] = group.loc[j, 'ds'] 
                    temp_format_df.loc[j, 'gauge1'] = group.loc[j, 'TimeGPT']
                elif id_gauge%1 < 0.25: #if gauge2
                    temp_format_df.loc[j, 'gauge2'] = group.loc[j, 'TimeGPT']
                elif id_gauge%1 > 0.25: # if gauge3
                    temp_format_df.loc[j, 'gauge3'] = group.loc[j, 'TimeGPT']
                    
    
        with open(file_path, 'ab') as file: # append at the end of the file
            pickle.dump(temp_format_df, file)
    
    print('Saving all in one file with proper format...')
    db = [] # creat a list where the information will be stored
    with open(file_path, 'rb') as file:
        try:
            while True:
                value = pickle.load(file)    # recover the data from the pickle file
                db.append(value)     # add it to the list
        except EOFError:
            pass
        
    os.remove(file_path) # delete the file because the presentation is not adequate
    format_final_df = pd.concat(db, ignore_index=True)  # concatenate the items of the list (index will be revisited, no [XX rows x 6 columns], and no heading)

    # TURNING THE DATES INTO INTEGER NUMBER OF CYCLES
    for index, row in format_final_df.iterrows():
        date_dt = datetime.strptime(row['cycle'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) # transform string date to datetime date
        format_final_df.loc[index, 'cycle'] = int(date_dt.timestamp()) # convert datetime date to integer (back to cycles)

    format_final_df.to_pickle(file_path)
    
    return format_final_df



def mape_calculation(df_restructured, df_forecast, path_dico=file_dico, path_results=file_MAPE):
    """
    Calculate the mean MAPE for each gauge
    
    Parameters
    ----------
   df_restructured : DataFrame
       DataFrame with the original data
   df_forecast : DataFrame
       DataFrame with the forecasted data
    path_dico : String
        path to the file where the dictionnary with all the RUL_max are stored
   ----------
    
    Returns
    ----------
    moy_gauge : list
        mean MAPE values for each gauge 
    """
    
    # RETRIEVE THE DICTIONNARY WITH THE RUL_MAX
    f = open(path_dico, 'r')
    dictionnaire_temp = ast.literal_eval(f.read())
    f.close()
    
    # CREATING A LIST WITH ALL THE IDS (1.1, 1.2, 1.3, 2.1, ETC) AND RETRIEVING THE MAX ID
    ids = []
    max = 1
    for key in dictionnaire_temp.keys():
        ids.append(key+0.1)
        ids.append(key+0.2)
        ids.append(key+0.3)
        if max < key:
            max = key
    
    
    groups = df_restructured.groupby(df_restructured.unique_id) # creating sub-dataframe, grouped by the unique_id
    test = pd.DataFrame(None) # intitialize the dataframe
    f = open(path_results, 'w') # creating/overwrite the file with the results of the MAPE
    
    # FILLING THE DATAFRAME
    for unique_id in ids :
        group_gauge = groups.get_group(unique_id) # calling the sub-dataframe labeled 'id_gauge'
        group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        len_group = dictionnaire_temp[math.floor(unique_id)]//delta_k-1
        index_keep = [index for index in range(len_group - h_size+1, len_group+1)] # list with all the index to keep in the sub-dataframe
        test = pd.concat([test, group_gauge.iloc[index_keep, :]], ignore_index=True) # fill the dataframe with the original information and index's of the forecast
    
    preds = df_forecast['TimeGPT'] # isolate TimeGPT's forecast
    test.loc[:,'TimeGPT'] = preds  # add the column of TimeGPT's forecast to the dataframe test
    f.write(str(test)+'\n \n') # add to the .txt file the dataframe with the original value and the forecasted value
    
    
    groups = test.groupby(test.unique_id) # creating sub-dataframe, grouped by the unique_id
    moy = []
    
    # CALCULATING THE MEAN MAPE
    for unique_id in ids : # for each gauge of each structure
        mes_resultats = []
        group_gauge = groups.get_group(unique_id) # calling the sub-dataframe labeled 'id_gauge'
        group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe

        # CALCULATE THE MAPE FOR EACH CYCLE
        for index, row in group_gauge.iterrows():
            mes_resultats.append(abs((row['y']-row['TimeGPT'])/row['y']))

        moy.append(sum(mes_resultats)/len(mes_resultats)*100) # returning the mean MAPE of the gauge of the structure
        
        # ADD THE VALUES TO THE .TXT FILE
        f.write('MAPE for each time step of id.gauge : ' + str(unique_id)+'\n') 
        f.write(str(mes_resultats)+'\n')
        
    f.write('\n')    
    moy_gauge = [0, 0, 0] # initialise a list, each element represent the mean of each gauge for all the structures
    
    for i in range(len(moy)):
        moy_gauge[i%3] += moy[i] # sum up all the mean MAPE of each structure 
    
    for i in range(len(moy_gauge)):
        moy_gauge[i] = moy_gauge[i]/max # calculating the mean value
        f.write('average MAPE gauge ' + str(i+1) + ' : ' + str(moy_gauge[i])+'\n') # adding the information to the .txt file
    
    f.close()
        
    return moy_gauge

"""
 ------------------------ IF __NAME__ = '__MAIN__' ----------------------------
"""

if __name__ == "__main__":
    
    print('Loading dataset...')
    df_panel = pd.read_pickle(file_data_path) # Loading our dataset
    
    print('\nRestructured labelled...')
    df_panel_labelled, dico_temp = restructure_unlabeled_timegpt_df(df=df_panel, h=0) # To calculate the MAPE at the end
    
    print('\nRestructuring and unlabelling dataset...')
    df_panel_unlabeled, dico_temp = restructure_unlabeled_timegpt_df(df=df_panel, h=h_size) # Restructuring for TimeGPT
    
    print('\nForecasting dataset...')
    df_panel_fcst = forecast_timeGPT(df=df_panel_unlabeled) # Forecasting with TimeGPT
    
    print('\nRestructuring dataset...')
    df_result = restructure_desirable_original_df() # Restructuring to the initial format
    
    print('\nCalculating the MAPE...')
    mape3 = mape_calculation(df_panel_labelled, df_panel_fcst) # Calculating the MAPE to quantify the error
    print('MAPE gauge 1 : ', round(mape3[0], 2), '%')
    print('MAPE gauge 2 : ', round(mape3[1], 2), '%')
    print('MAPE gauge 3 : ', round(mape3[2], 2), '%')
    


