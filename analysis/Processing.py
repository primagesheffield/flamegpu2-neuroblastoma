#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:41:30 2022

@author: kywertheim
"""

"""
Load the necessary libraries.
"""
import pandas
import os
import json

"""
Load the 1200 configurations, copy some features to a dataset, and convert
the remaining features (about clonal composition) into a readable form.
"""
df_raw = pandas.read_csv("LHC_hetNB.csv")
df_processed=df_raw[['Index', 'O2', 'cellularity', 'theta_sc', 'degdiff']]
for i in range(24):
    cloneID = 'C'+str(i+1)
    df_processed.insert(5+i, cloneID, 0)
cwd=os.getcwd()
for i in range(1200):
    dummy=list(df_raw.iloc[i, 5:28])
    dummy.sort()
    dummy.append(1)
    clones = []
    for j in range(len(dummy)):
        if j == 0:
            clones.append(dummy[j])
        else:
            clones.append(dummy[j]-dummy[j-1])
    df_processed.iloc[i, 5:29]=clones

"""
Create new columns* in the dataset to store the outputs corresponding to the
1200 configurations.
R means regression.
D means differentiation.
O means others.
The column denoted by `Files' indicates the number of complete JSON files for
each configuration.
*The first new column, denoted by `MYCN_init', is sttrictly speaking for an
input.
"""
df_processed.insert(29, 'MYCN_init', 0)
df_processed.insert(30, 'MYCN_final', 0)
df_processed.insert(31, 'MYCN_final_init', 0)
df_processed.insert(32, 'Nnb_final', 0)
df_processed.insert(33, 'degdiff_final', 0)
df_processed.insert(34, 'Nnbl_final', 0)
df_processed.insert(35, 'Nscl_final', 0)
df_processed.insert(36, 'O2_final', 0)
df_processed.insert(37, 'R', 0)
df_processed.insert(38, 'D', 0)
df_processed.insert(39, 'O', 0)
df_processed.insert(40, 'Files', 0)

"""
For each configuration, extract data from each JSON file and populate the new
columns with values averaged over each set of JSON files.
"""
cwd=os.getcwd() #Present working directory.
for i in range(1200):
    print(i+1)
    os.chdir(cwd+'/'+str(i+1)) #Enter the directory corresponding to the current configuration.
    MYCN_init=[]
    MYCN_final=[]
    MYCN_final_init=[]
    Nnb_final=[]
    degdiff_final=[]    
    Nnbl_final=[]    
    Nscl_final=[]
    O2_final=[]
    R=0
    D=0
    O=0
    for file in os.listdir():
        f = open(file)
        try:
            data = json.load(f) #Try to open each JSON file.
        except (ValueError, NameError):
            f.close()
            MYCN_init_dummy = sum(df_processed.iloc[i, 11:17])
            MYCN_init.append(MYCN_init_dummy)
            Nnb_final.append(0) 
            Nnbl_final.append(0)
            R+=1 #An incomplete JSON file means the corresponding run ended in regression: no living neuroblasts.
            continue
        f.close()
        """
        Classify the run into one of the three categories.
        """
        data_steps = data['steps']
        data_end = data_steps[3024]
        data_end_values = data_end['environment']        
        if data_end_values['Nnbl_count'] == 0:
            """
            Parse the JSON file.
            """
            MYCN_init_dummy = sum(df_processed.iloc[i, 11:17])
            MYCN_init.append(MYCN_init_dummy)
            Nnb_final.append(0) 
            Nnbl_final.append(0)
            Nscl_final.append(data_end_values['Nscl_count'])
            O2_final.append(data_end_values['O2'])
            R+=1      
        elif data_end_values['NB_living_degdiff_average']>0.99:
            """
            Parse the JSON file.
            """
            MYCN_init_dummy = sum(df_processed.iloc[i, 11:17])
            MYCN_init.append(MYCN_init_dummy)
            clones = data_end_values['NB_living_count']
            MYCNclones = clones[6]+clones[7]+clones[8]+clones[9]+clones[10]+clones[11]
            Allclones = sum(clones)
            MYCN_final.append(MYCNclones/Allclones)
            MYCN_final_init.append(MYCNclones/Allclones/MYCN_init_dummy)
            Nnb_final.append(sum(clones))
            degdiff_final.append(data_end_values['NB_living_degdiff_average'])
            Nnbl_final.append(data_end_values['Nnbl_count'])
            Nscl_final.append(data_end_values['Nscl_count'])
            O2_final.append(data_end_values['O2'])
            D+=1            
        else:
            """
            Parse the JSON file.
            """
            MYCN_init_dummy = sum(df_processed.iloc[i, 11:17])
            MYCN_init.append(MYCN_init_dummy)
            clones = data_end_values['NB_living_count']
            MYCNclones = clones[6]+clones[7]+clones[8]+clones[9]+clones[10]+clones[11]
            Allclones = sum(clones)
            MYCN_final.append(MYCNclones/Allclones)
            MYCN_final_init.append(MYCNclones/Allclones/MYCN_init_dummy)
            Nnb_final.append(sum(clones))
            degdiff_final.append(data_end_values['NB_living_degdiff_average'])
            Nnbl_final.append(data_end_values['Nnbl_count'])
            Nscl_final.append(data_end_values['Nscl_count'])
            O2_final.append(data_end_values['O2'])
            O+=1            
    try: #Try to average over the current set of JSON files.
        df_processed.iloc[i, 29]=sum(MYCN_init)/len(MYCN_init)
        df_processed.iloc[i, 30]=sum(MYCN_final)/len(MYCN_final)
        df_processed.iloc[i, 31]=sum(MYCN_final_init)/len(MYCN_final_init)
        df_processed.iloc[i, 32]=sum(Nnb_final)/len(Nnb_final)
        df_processed.iloc[i, 33]=sum(degdiff_final)/len(degdiff_final)
        df_processed.iloc[i, 34]=sum(Nnbl_final)/len(Nnbl_final)
        df_processed.iloc[i, 35]=sum(Nscl_final)/len(Nscl_final)
        df_processed.iloc[i, 36]=sum(O2_final)/len(O2_final)
        df_processed.iloc[i, 37]=R
        df_processed.iloc[i, 38]=D
        df_processed.iloc[i, 39]=O
        df_processed.iloc[i, 40]=len(degdiff_final)
        if len(degdiff_final) != 10:
            print(len(degdiff_final), 'JSON files are incomplete!')
    except (ZeroDivisionError): #When all the JSON files are incomplete, this configuration requires special attention.
        df_processed.iloc[i, 29]=sum(MYCN_init)/len(MYCN_init)        
        df_processed.iloc[i, 30]=0
        df_processed.iloc[i, 31]=0
        df_processed.iloc[i, 32]=sum(Nnb_final)/len(Nnb_final)
        df_processed.iloc[i, 33]=0
        df_processed.iloc[i, 34]=sum(Nnbl_final)/len(Nnbl_final)
        df_processed.iloc[i, 35]=0
        df_processed.iloc[i, 36]=0
        df_processed.iloc[i, 37]=R
        df_processed.iloc[i, 38]=D
        df_processed.iloc[i, 39]=O
        df_processed.iloc[i, 40]=len(degdiff_final)        
        print('All JSON files are incomplete!')
        pass
os.chdir(cwd) #Return to the present working directory.
df_processed.to_pickle("hetNB_results.pkl") #Save the dataset.