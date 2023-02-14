# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:01:05 2020

@author: dbromley
"""

import numpy as np 
import matplotlib.pyplot as plt
import os

import wx

import pandas as pd

import lmfit.models as lm

from matplotlib.ticker import AutoMinorLocator


def file_finder():
    '''
    Inputs:
            
            N/A
       
    Description:
            
                Uses wxPython library to create a dialog box allowing for the choosing of files for data analysis
     
    Returns:
            
            'path' - this is the path of the data file that will be analysed
    '''
        
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    
    print('path = ', path)
    return path


def datacollector(filename):
    
    """
        Inputs:
            
            'filename' - the filename of the file you want to extract data from
    
        
        Description:
            
            Uses np.genfromtext to extract data from a file - IN NEAR FUTURE - will determine delimiter and not need to be told 
            
        Returns:
            
            'data' - this is the data in the file
            'label' - this is a list of the data headers - must contain 'Time', change if 'Time' not a column
            
            
        NEEDS SOME ERROR TESTING STUFF try except etc
            
        
    """
        
    with open(filename, 'r') as datafile:
        
        delimit = None
                
        for line in datafile:
                if 'Comment,Time Stamp (sec),Temperature (K)' in line:
                    if ',' in line:
                        delimit = ',' #CSV file delimiter
                    
                    if '	' in line:
                        delimit = '	' #txt files delimiter
                    
                    labels=line.rstrip() #removes \n from last label
                    labels = labels.split(delimit)
                    
                    break 
                else:
                    continue
                    
        data = np.genfromtxt(datafile,delimiter=delimit,skip_header=0,filling_values="nan")
            
    return data,labels

def data_sorter(filename):
    '''
    Gets data from the SQUID file and stores it in a pandas dataframe
    '''
    
    data, labels = datacollector(filename)
    
    labels = np.asarray(labels)
        
    df = pd.DataFrame(data, columns=labels)

    return df

def init_saturation_remover(df):
    '''
    Removes the initial saturation curve from the datafile
    '''
    
    H_0 = -1e9 #setting H_0 effectively sets it at -ve infinity - so there won't be a value more negative than this
    i = 0
    for H in df["Magnetic Field (Oe)"]:
        if H < H_0:
            break
        i += 1
        H_0 = H #this changes H_0 to the current H value if H hasn't decreased, when it does, it breaks the loop
                #therefore indicating we are at the end of the initial saturation
                
    df.drop([j for j in range(0,i)], inplace=True) #removes the rows containing the inital suturation values
    
    return df
    

    
def linear_fit(x,y):
    '''
    Linear fitter for the paramagnetic removal part returns the intercept and slope values
    '''

    model = lm.LinearModel(prefix='linear_') #Linear equation in to a fitting model

    slope_guess = (np.nanmax(y)-np.nanmin(y))/(np.nanmax(x)-np.nanmin(x))
    params = model.make_params(linear_slope=slope_guess,linear_intercept=1e-6) #some reasonable guesses

    result = model.fit(y,params,x=x)#results of fit
    
    '''
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.legend()
    plt.show()
    '''
    return result

def paramagnetic_remover(squid_dataframe, saturation_guess):
    '''
    Removes the paramagnetic response in the SQUID data and returns the corrected moment
    '''
    
    Hx = squid_dataframe['Magnetic Field (Oe)'].values
    forward_indices = np.where([int(sub1) < int(sub2) for sub1, sub2 in zip(Hx,Hx[1:])])[0] #split data in to forward and reverse parts of the  
    reverse_indices = np.where([int(sub2) < int(sub1) for sub1, sub2 in zip(Hx, Hx[1:])])[0] #hysteresis curve

    forward_df = squid_dataframe.copy()
    forward_df = forward_df.iloc[forward_indices] #find forward sweep
    reverse_df = squid_dataframe.copy()
    reverse_df = reverse_df.iloc[reverse_indices] #find reverse sweep
    
    forward_sat_subframe = forward_df.loc[forward_df["Magnetic Field (Oe)"]<saturation_guess]
    print('-saturation_guess', -saturation_guess)
    reverse_sat_subframe = reverse_df.loc[-saturation_guess<reverse_df["Magnetic Field (Oe)"]]

    forward_sat_fit = linear_fit(forward_sat_subframe["Magnetic Field (Oe)"],forward_sat_subframe["Moment (emu)"])
    reverse_sat_fit = linear_fit(reverse_sat_subframe["Magnetic Field (Oe)"],reverse_sat_subframe["Moment (emu)"])
    
    plt.figure()
    plt.title('SATURATION REMOVED + PARAMAGNETIC REGIONS FITTED \n' + title,fontsize=13)

    plt.plot(forward_df["Magnetic Field (Oe)"]/10,forward_df["Moment (emu)"],linestyle='solid',marker='None',label='Foward sweep data')
    plt.plot(reverse_df["Magnetic Field (Oe)"]/10,reverse_df["Moment (emu)"],linestyle='solid',marker='None',label='Reverse sweep data')

    plt.plot(forward_sat_subframe["Magnetic Field (Oe)"]/10,forward_sat_subframe["Moment (emu)"],'o',markersize=5,label='foward fit data')
    plt.plot(forward_sat_subframe["Magnetic Field (Oe)"]/10, forward_sat_fit.best_fit, '-', label='forward fit')
    
    plt.plot(reverse_sat_subframe["Magnetic Field (Oe)"]/10,reverse_sat_subframe["Moment (emu)"],'o',markersize=5,label='reverse fit data')
    plt.plot(reverse_sat_subframe["Magnetic Field (Oe)"]/10, reverse_sat_fit.best_fit, '-', label='reverse fit')
    plt.legend(loc='best',fontsize=9)

    plt.ylabel('Moment (emu)',fontsize=13)
    
    if 'OOP' in filename:
        plt.xlabel(u"\u00B5" + r'$_{0}$H$_{\perp}$ (mT)',fontsize=13)
    if 'IP' in filename:
        plt.xlabel(u"\u00B5" + r'$_{0}$H$_{\parallel}$ (mT)',fontsize=13)
    
    plt.tight_layout()
    plt.show()

    
    average_gradient = np.mean([forward_sat_fit.values['linear_slope'],reverse_sat_fit.values['linear_slope']])
    
    squid_dataframe["Moment (emu)"] = squid_dataframe["Moment (emu)"] - average_gradient*squid_dataframe["Magnetic Field (Oe)"]
    
    return squid_dataframe    

def normalise_moment(squid_df):
    '''
    Normalises moment between 1 and -1 
    underlying eq:     (b-a)*(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)) + a
    
    '''
    moment_max = np.nanmax(squid_df["Moment (emu)"].values)
    moment_min = np.nanmin(squid_df["Moment (emu)"].values)
    
    normalised_moment = 2*(squid_df["Moment (emu)"]-moment_min)/(moment_max-moment_min) - 1

    squid_df["Moment (emu)"] = normalised_moment
    
    return squid_df

def drift_removal(squid_df):
    '''
    Make forward and reverse curves have same M value at field extremeties
    '''

    Hx = squid_df['Magnetic Field (Oe)'].values
    forward_indices = np.where([int(sub1) < int(sub2) for sub1, sub2 in zip(Hx,Hx[1:])])[0] #split data in to forward and reverse parts of the  
    reverse_indices = np.where([int(sub2) < int(sub1) for sub1, sub2 in zip(Hx, Hx[1:])])[0] #hysteresis curve

    forward_df = squid_df.copy()
    forward_df = forward_df.iloc[forward_indices] #find forward sweep
    reverse_df = squid_df.copy()
    reverse_df = reverse_df.iloc[reverse_indices] #find reverse sweep

    
    pos_region_diff = abs(forward_df['Moment (emu)'].values[-1] - reverse_df['Moment (emu)'].values[0])

    reverse_df.loc[:,'Moment (emu)'] = reverse_df['Moment (emu)'].values - pos_region_diff
    
    new_df = pd.concat([forward_df,reverse_df],ignore_index=True) #combine the outlier and non outlier datframes

    return new_df

def OOP_data_fixer(df):
    
    new_df = init_saturation_remover(df) #remove initial saturation curves

    #take linear gradient of paramagnetic component and substract
    if PARA_REMOVE:
        new_df = paramagnetic_remover(new_df, -1000)
        
    if DRIFT_REMOVE:
        new_df = drift_removal(new_df)  

    if NORMALISE:
        new_df = normalise_moment(new_df)

    return new_df

def IP_data_fixer(df):
    
    new_df = init_saturation_remover(df)

    if PARA_REMOVE:
        new_df = paramagnetic_remover(new_df, -5000)
    
    if DRIFT_REMOVE:
        new_df = drift_removal(new_df)    
    
    if NORMALISE:
        new_df = normalise_moment(new_df)
    

    return new_df

def final_figure_maker(df,orientation):
    '''
    Plot M vs H 
    '''
    S1_colour = 'royalblue'
    
    cm_to_inch = 1/2.54
    
    fig, (ax0) = plt.subplots(1,1) #183mm is nature max width

    ax0.plot(df['Magnetic Field (Oe)']/10,df['Moment (emu)'],\
                   c=S1_colour,linestyle='solid',marker='o', linewidth=1.0, markersize = marker_size)

    for axis in ['top','left','right']:
        ax0.spines[axis].set_linewidth(axes_thickness)

    ax0.tick_params(direction='in',which='both',bottom=True,top=True,left=True,right=True)
    ax0.set_xlim(-1.1*np.nanmax(df['Magnetic Field (Oe)'])/10,1.1*np.nanmax(df['Magnetic Field (Oe)'])/10)
    
    #leg = ax0.legend(loc='lower right', frameon=False, fontsize='medium',handlelength=0, handletextpad=0.75)

    
    ax0.xaxis.set_minor_locator(AutoMinorLocator(5)) 
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5)) 

    if NORMALISE == True:
        ax0.set_ylabel(r'Moment (norm. units)',fontsize=13, **Afont)

    if NORMALISE == False:
        ax0.set_ylabel(r'Moment (emu)',fontsize=13, **Afont)
    
    if orientation == 'IP':
        ax0.set_xlabel(u"\u00B5" + r'$_{0}$H$_{\parallel}$ (mT)',fontsize=13, **Afont)
    if orientation == 'OOP':
        ax0.set_xlabel(u"\u00B5" + r'$_{0}$H$_{\perp}$ (mT)',fontsize=13, **Afont)
    
    # remove vertical gap between subplots
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)

    if save_fig:
        plt.savefig(path + title + r'_.png', dpi=800)
        
    plt.show()

if __name__ == "__main__":   
    
    NORMALISE = False #make True to normalise moment (y axis)
    DRIFT_REMOVE = False #make True to remove drift from M-H signal
    PARA_REMOVE = True #make True to remove paramagnetic component of signal
    
    save_fig = False
    
    Afont = {'fontname':'Arial'}
    
    axes_thickness = 1
    
    fontsize = 15.5
    
    cm_to_inch = 1/2.54

    marker_size = 4
    
    script_path = os.path.dirname(os.path.realpath(__file__))
        
    filename = file_finder()
    file_df = data_sorter(filename)

    path = os.path.dirname(os.path.abspath(filename)) #strips path from whole filepath
    path = path + '\\' #adds '\' so it can be used later to save figs in same folder as data file
    title = os.path.splitext(os.path.basename(filename))[0]

    
    plt.figure()
    plt.title('RAW DATA \n' + title,fontsize=13)
    plt.plot(file_df['Magnetic Field (Oe)']/10,file_df['Moment (emu)'])
    plt.ylabel('Moment (emu)',fontsize=13)
    
    if 'OOP' in filename:
        plt.xlabel(u"\u00B5" + r'$_{0}$H$_{\perp}$ (mT)',fontsize=13)
    if 'IP' in filename:
        plt.xlabel(u"\u00B5" + r'$_{0}$H$_{\parallel}$ (mT)',fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
    if 'OOP' in filename:
        new_df = OOP_data_fixer(file_df)
        final_figure_maker(new_df,'OOP')

    
    if 'IP' in filename:
        new_df = IP_data_fixer(file_df)
        final_figure_maker(new_df,'IP')
    
    
    
    
    
    