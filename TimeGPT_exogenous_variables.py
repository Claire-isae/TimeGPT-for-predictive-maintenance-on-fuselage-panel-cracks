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
id_to_forecast=1
nb_exo = 10

folder_path = 'data'
file_data_path = folder_path + '/data_train'
file_restructured_ID = folder_path + '/ID1_restructured'
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
    TimeGPT's forecast for one-shot or finte-tuning
    
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



def exogenous_variables(df, future_ex_vars_df):
    """
    TimeGPT's forecast with exogenous variables
    
    Parameters
    ----------
    df : DataFrame
        past data
    future_ex_vars_df : DataFrame
        future data for the exogenous variables
        
    ----------
    
    Returns
    ----------
    timegpt_fcst_ex_vars_df: DataFrame
        DataFrame of the forecast
    len_exo : int
        number of steps predicted
    """
    
    len_exo = (future_ex_vars_df.index.max()+1)//3 # calculate the number of steps to predict
    timegpt_fcst_ex_vars_df = nixtla_client.forecast(df=df, X_df=future_ex_vars_df, freq=k_to_seconds(delta_k), h=len_exo, finetune_loss='default') # TimeGPT forecasting
    
    return timegpt_fcst_ex_vars_df, len_exo


"""
 ----------------------- RESTRUCTURING THE DATAFRAME --------------------------
"""

def restructure_ID_df(df, h=h_size, id_forecast=id_to_forecast, path_dico=file_dico, file_path=file_restructured_ID):
    """
    Restructure the DataFrame from the generation of datasets from Claire M. for the ID to be forecasted
    
    Parameters
    ----------
    df: DataFrame
        past data
    h: int
        number of future time steps to be forecasted (horizon)
    id_forecast : int
        number of the ID to be forecasted
    path_dico : String
        default file path where the dictionnary with the max RUL will be stored
    file_path : String
        default file path to the restructured data to fit TimeGPT's requirements
   ----------
    
    Returns
    ----------
    restructured_df: DataFrame
        restructured DataFrame, ready to be used in TimeGPT
    dictionnaire_temp: dictionnary
        unique_id (float): date of the failure (dateStamp)
    """

    cols = ['unique_id', 'ds', 'y']
    dictionnaire_temp={}
    df_temp = df.copy()
    
    groups = df_temp.groupby(df_temp.ID) # creating sub-dataframe, grouped by the unique_id
    group = groups.get_group(id_forecast) # choose the sub-dataframe with the chosen id
    group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
    len_max = group.index.max()+1
    len_max_group = len_max*3
    
    with open(file_path, 'wb'): 
        pass
 
    restructured_df = pd.DataFrame(None) #intitialize the dataframe

    with open(file_path, 'ab') as file: 
        pickle.dump(restructured_df, file) # append at the end of the file
    
    restructured_df = pd.DataFrame(np.zeros((len_max_group,len(cols))), columns =cols, dtype='float').astype({"ds": int})  #initialize a sub-dataframe for structure i
    
    dictionnaire_temp[id_forecast] = group.loc[0, 'RUL'] # RUL_max 

    
    for index, row in group.iterrows():
        time_cycle = row['cycle']
        index = 3*index 
        
        # ADDING THE ELEMENTS IN THE DATAFRAME:
        restructured_df.loc[index,'unique_id']=id_forecast + 0.1 # for gauge 1, the unique_id is renamed ID.1
        restructured_df.loc[index, 'ds']=time_cycle # fill the time
        restructured_df.loc[index, 'y']=row['gauge1'] # fill the gauge 1 value
        
        restructured_df.loc[index+1, 'unique_id']=id_forecast + 0.2
        restructured_df.loc[index+1, 'ds']=time_cycle
        restructured_df.loc[index+1, 'y']=row['gauge2']
        
        restructured_df.loc[index+2, 'unique_id']=id_forecast + 0.3
        restructured_df.loc[index+2, 'ds']=time_cycle
        restructured_df.loc[index+2, 'y']=row['gauge3']
            
        groups_gauges = restructured_df.groupby(restructured_df.unique_id) # creating sub-dataframe, grouped by the unique_id

    for unique_id in [id_forecast+0.1, id_forecast+0.2, id_forecast+0.3] : 
        group_gauge = groups_gauges.get_group(unique_id) # calling the sub-dataframe labeled 'id_gauge'
        group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        
        with open(file_path, 'ab') as file: 
            pickle.dump(group_gauge, file) # append at the end of the file
    
    # CREATING THE PROPER FILE TO REMOVE THE HEADING:
    db = [] # creat a list where the information will be stored
    with open(file_path, 'rb') as file:
        try:
            while True:
                value = pickle.load(file)    # recover the data from the pickle file
                db.append(value)     # add it to the list
        except EOFError:
            pass
        
    os.remove(file_path) # delete the file because the presentation is not adequate
    
    restructured_df = pd.DataFrame(np.zeros((len_max_group,len(cols))), columns =cols, dtype='float').astype({"ds": int}) 
    
    restructured_df = pd.concat(db, ignore_index=True)  # concatenate the items of the list (index will be revisited, no [XX rows x 6 columns], and no heading)
    restructured_df.to_pickle(file_path)    # turn it into a DataFrame (format used in the rest of the project)
    
    # CREATING THE FILE FOR THE DICTIONNARY 
    f = open(path_dico, 'w')
    f.write(str(dictionnaire_temp))
    f.close()
        
    return restructured_df, dictionnaire_temp




def restructure_exogeneous_df(dfID, df, id_forecast=id_to_forecast, h=h_size, file_path = file_data_restructured_for_TimeGPT, path_dico=file_dico, delta_k_loc=delta_k):
    """
    Restructure the DataFrame from the generation of datasets from Claire M. to suit TimeGPT's requirements
    
    Parameters
    ----------
    dfID : DataFrame
        dataframe of the ID to be forecasted
    df: DataFrame
        DataFrame to be restructured
    id_forecast : int
        number of the ID to be forecasted
    h: int
        number of future time steps to be forecasted (horizon)
    file_path : String
        default file path to the restructured data to fit TimeGPT's requirements
    path_dico : String
        default file path where the dictionnary with the max RUL will be stored
    delta_k : int
        frequence at which the data is retrieved
   ----------
    
    Returns
    ----------
    restructured_df : DataFrame
        restructured dataframe, input for TimeGPT
    exo_df : DataFrame
        input TimeGPT with the exogenous variables
    """
    
    list_exo_ID = [i+1 for i in range(df.loc[df.index.max(), 'ID'])]
    list_exo_ID.remove(id_forecast)
    
    cols = ['unique_id', 'ds', 'y'] +  ['ID' + str(i) for i in list_exo_ID]
    df_temp = df.copy()
    
    group_ID = dfID.copy() 
    groups = df_temp.groupby(df_temp.ID) # creating sub-dataframe, grouped by the unique_id
    len_max = group_ID.index.max()+1
    
    with open(file_path, 'wb'): # create/overwrite the file with the restructured dataFrame
        pass
 
    restructured_df = pd.DataFrame(None) #intitialize the dataframe

    with open(file_path, 'ab') as file: # append at the end of the file
        pickle.dump(restructured_df, file)
            
    for i in range(len_max) : # index de la ligne sur laquelle on travaille
        
        restructured_df = pd.DataFrame(np.zeros((0,len(cols))), columns =cols, dtype='float').astype({"ds": int}) # creating sub-dataframe
        
        restructured_df.loc[i,'unique_id']=group_ID.loc[i, 'unique_id']
        restructured_df.loc[i,'ds']=group_ID.loc[i, 'ds']
        restructured_df.loc[i,'y']=group_ID.loc[i, 'y']
        cycles = restructured_df.loc[i, 'ds']
        
        for j in list_exo_ID : # ID sur lequel on travail
            group = groups.get_group(j) # calling the sub-dataframe labeled 'j+1'
            group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
            rows = group.index[group['cycle'] == cycles].tolist()
            
            if rows == []:
                restructured_df.loc[i, 'ID'+str(j)]=0
                pass
            else:
                row_index = rows[0]
        
            if row_index > group.index.max():
                pass
            elif restructured_df.loc[i, 'unique_id']%1 < 0.15:
                restructured_df.loc[i, 'ID'+str(j)]=group.loc[row_index, 'gauge1']
                
            elif restructured_df.loc[i, 'unique_id']%1 < 0.25:
                restructured_df.loc[i, 'ID'+str(j)]=group.loc[row_index, 'gauge2']
                
            elif restructured_df.loc[i, 'unique_id']%1 > 0.25:
                restructured_df.loc[i, 'ID'+str(j)]=group.loc[row_index, 'gauge3']
        
        with open(file_path, 'ab') as file: 
            pickle.dump(restructured_df, file)
        

    # CREATING THE PROPER FILE TO REMOVE THE HEADING:    
    db = [] # creat a list where the information will be stored
    with open(file_path, 'rb') as file:
        try:
            while True:
                value = pickle.load(file)    # recover the data from the pickle file
                db.append(value)     # add it to the list
                
        except EOFError:
            pass
    
    os.remove(file_path) # delete the file because the presentation is not adequate
    
    restructured_df = pd.concat(db, ignore_index=True)  # concatenate the items of the list (index will be revisited, no [XX rows x 6 columns], and no heading)
    exo_df = restructured_df.copy()
    
    # CREATING THE FILE FOR THE DICTIONNARY
    f = open(path_dico, 'r')
    dictionnaire_temp = ast.literal_eval(f.read())
    f.close()
    
    groups = exo_df.groupby(exo_df.unique_id) # creating sub-dataframe, grouped by the unique_id
    exo_df = pd.DataFrame(np.zeros((0,len(cols))), columns =cols, dtype='float').astype({"ds": int}) 
    
    # RESTRUCTURE THE EXOGENOUS VARIABLES DATAFRAME
    for unique_id in [id_forecast+0.1, id_forecast+0.2, id_forecast+0.3]:
        group = groups.get_group(unique_id) # calling the sub-dataframe labeled 'j+1'
        group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        for index, row in group.iterrows():
            if row['ds'] > dictionnaire_temp[id_forecast]-h*delta_k_loc : # find the row where the forecast whould start
                group=group.drop(group.head(index).index) # fill the exo_df with all the remaining rows of he exogenous data
                group = group.reset_index(drop=True)
                break
        exo_df = pd.concat([exo_df, group], ignore_index=True)
    
    groups = restructured_df.groupby(restructured_df.unique_id) # creating sub-dataframe, grouped by the unique_id
    restructured_df = pd.DataFrame(np.zeros((0,len(cols))), columns =cols, dtype='float').astype({"ds": int}) 
    
    # RESTRUCURE THE MAIN DATAFRAME FOR THE FORECAST
    for unique_id in [id_forecast+0.1, id_forecast+0.2, id_forecast+0.3]:
        group = groups.get_group(unique_id) # calling the sub-dataframe labeled 'j+1'
        group = group.reset_index(drop=True) # reinitialise the index of the sub-dataframe
        len_max = group.index.max()+1
        for index, row in group.iterrows():
            if row['ds'] > dictionnaire_temp[id_forecast]-h*delta_k_loc : # find the row where the forecast whould start
                group=group.drop(group.tail(len_max-index).index) # remove the remaining rows
                group = group.reset_index(drop=True)
                break
        restructured_df = pd.concat([restructured_df, group], ignore_index=True)
        
    exo_df = exo_df.drop(columns = ['y'])
    restructured_df['ds'] = pd.to_datetime(restructured_df['ds'], unit='s') # convert the cycles into dates: YYYY-MM-DD HH:MM:SS at UTC 1970-01-01 00:00:00
    exo_df['ds'] = pd.to_datetime(exo_df['ds'], unit='s') # convert the cycles into dates: YYYY-MM-DD HH:MM:SS at UTC 1970-01-01 00:00:00
    restructured_df.to_pickle(file_path)    # turn it into a DataFrame (format used in the rest of the project)
    exo_df.to_pickle(file_path+'_exo')    # turn it into a DataFrame (format used in the rest of the project)
    
    return restructured_df, exo_df
        

def restructure_desirable_original_df(df= None, id_forecast=id_to_forecast, file_path=file_data_forecasted_restructured, file_path_timeGPT = file_data_TimeGPT_forecasting, delta_k_loc=delta_k):
    """
    Restructure the DataFrame to its original format
    
    Parameters
    ----------
    df: DataFrame
        DataFrame to be restructured
    id_forecast : int
        number of the ID to be forecasted
    file_path : String
        default file path to save the restructured data after forecasting
    file_path_TimeGPT: String
        path twhere the forecast was saved
    delta_k : int
        frequence at which the data is retrieved

   ----------
    
    Returns
    ----------
    format_final_df: DataFrame
        restructured DataFrame, with [index, 'ID', 'cycle', 'gauge1', 'gauge1', 'gauge3', 'RUL'], RUL=0 (not evaluated by TimeGPT)
    """
    
    file_path+=str(id_forecast)
    
    if df is None: # default setting made with this code
        temp_df = pd.read_pickle(file_path_timeGPT)
    else:
        temp_df = df # creating a copy of df, to avoid modifying df outside of the function
        
    groups = temp_df.groupby(temp_df.unique_id) # creating sub-dataframe, grouped by the unique_id
    cols_temp = ['ID', 'cycle', 'gauge1', 'gauge2', 'gauge3', 'RUL'] # creating 3 columns 'cycle' to check if the time/cycles are identical
      
    with open(file_path, 'wb'): # create/overwrite the file with the restructured dataFrame
        pass
 
    db = pd.DataFrame(None) #intitialize the dataframe

    with open(file_path, 'ab') as file: # append at the end of the file
        pickle.dump(db, file)
        
    # CREATING A DICTIONNARY WITH ALL THE UNIQUE ID'S (1.1, 1.2, 1.3, 2.1, ETC)
    dictionnaire_id={}       
    dictionnaire_id[id_forecast]=[id_forecast+0.1, id_forecast+0.2, id_forecast+0.3]

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
    
    print('Saving all in one file with proper format')
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




def mape_calculation(df_restructured, df_forecast, len_exo, id_forecast=id_to_forecast, path_dico=file_dico, path_results=file_MAPE):
    """
    MAPE calculation
    
    Parameters
    ----------
    df_restructured : DataFrame
        restructured dataframe from the database
    df_forecast : DataFrame
        forecassted dataframe after restructuration
    len_exo : int
        length of the forecast (horizon)
    id_forecast : int
        ID that was forecasted
    path_dico : String
        path to the dictionnary with the max RUL
    path_results : String
        path to store the results in a .txt file

   ----------
    
    Returns
    ----------
    moy : list
        list of length 3: for every gauge
    """

    groups = df_restructured.groupby(df_restructured.unique_id) # creating sub-dataframe, grouped by the unique_id
    test = pd.DataFrame(None) #intitialize the dataframe
    f = open(path_results+str(id_forecast), 'w') # creating/overwrite the file with the results of the MAPE
    
    for unique_id in [id_forecast+0.1, id_forecast+0.2, id_forecast+0.3] :
        group_gauge = groups.get_group(unique_id) # calling the sub-dataframe labeled 'id_gauge'
        group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe

        test = pd.concat([test, group_gauge.tail(len_exo)], ignore_index=True)
    
    preds = df_forecast['TimeGPT'].values
    test.loc[:,'TimeGPT'] = preds
    f.write(str(test)+'\n \n') # add to the .txt file the dataframe with the original value and the forecasted value
    
    groups = test.groupby(test.unique_id) # creating sub-dataframe, grouped by the unique_id
    moy = []
    for unique_id in [id_forecast+0.1, id_forecast+0.2, id_forecast+0.3]:
        mes_resultats = []
        group_gauge = groups.get_group(unique_id) # calling the sub-dataframe labeled 'id_gauge'
        group_gauge = group_gauge.reset_index(drop=True) # reinitialise the index of the sub-dataframe

        for index, row in group_gauge.iterrows():
            mes_resultats.append(abs((row['y']-row['TimeGPT'])/row['y']))
        
        moy.append(sum(mes_resultats)/len(mes_resultats)*100)
        # ADD THE VALUES TO THE .TXT FILE
        f.write('MAPE for each time step of id.gauge : ' + str(unique_id)+'\n') 
        f.write(str(mes_resultats)+'\n')
        f.write('The mean for this gauge : '+str(moy))
        
    f.write('\n')   
    
    f.close()
    
    return moy
    
"""
 ------------------------ IF __NAME__ = '__MAIN__' ----------------------------
"""

if __name__ == "__main__":
    
    print('loading dataset...')
    df_panel = pd.read_pickle('data/data_train') # Loading our dataset
    
    tot=[0, 0, 0]
    
    for i in range(nb_exo):
        print('restructuring and unlabelling dataset...')
        df_panel_restructured, dico_temp = restructure_ID_df(df=df_panel, id_forecast=i+1) # Restructuring for TimeGPT
        
        print('adding the exogeneous variables...')
        df_with_exo, df_following_exo = restructure_exogeneous_df(df_panel_restructured, df_panel, id_forecast=i+1)
        
        #print(df_with_exo, '\n', df_following_exo)
        
        print('forecasting with exogenous...')
        df_forecast, len_exo = exogenous_variables(df_with_exo, df_following_exo)
        
        df_final = restructure_desirable_original_df(df_forecast, id_forecast=i+1)
        
        print('MAPE for the 3 gauges (%):')
        mape3 = mape_calculation(df_panel_restructured, df_forecast, len_exo, id_forecast=i+1)
        print(mape3, '\n')
        
        tot[0]+=mape3[0]
        tot[1]+=mape3[1]
        tot[2]+=mape3[2]
 
    tot[0]=tot[0]/nb_exo
    tot[1]=tot[1]/nb_exo
    tot[2]=tot[2]/nb_exo
    
    print('Average MAPE for the 3 gauges (%):')
    print(tot)
    
