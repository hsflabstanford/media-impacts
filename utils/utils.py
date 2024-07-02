import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib.dates as mdates
from datetime import datetime, timedelta

"""
Assumes format: YYYY-, e.g. 2004-
"""
def add_time_month(df, date_col, add_day=False):
    df['date_col'] = pd.to_datetime(df[date_col])
    df['Year'] = df['date_col'].dt.year
    df['Month'] = df['date_col'].dt.month
    #df['Time'] = df.reset_index(drop=True).index
    if add_day:
        df['Day'] = df['date_col'].dt.day
      
    return df

def date_difference(date1, date2, unit, in_datetime=False):
    # Convert input strings to datetime objects
    if not in_datetime:
        date1 = datetime.strptime(date1, '%Y-%m-%d')
        date2 = datetime.strptime(date2, '%Y-%m-%d')

    # Calculate the difference in days
    delta = date2 - date1

    if unit == 'months':
        # Calculate the difference in months
        months_difference = (date2.year - date1.year) * 12 + date2.month - date1.month
        return months_difference
    elif unit == 'weeks':
        # Calculate the difference in weeks
        weeks_difference = delta.days // 7
        return weeks_difference
    else:
        raise ValueError("Invalid unit. Use 'months' or 'weeks'.")
    
def normalize_shocks(df, ratio_map):
    """
    TODO: an assert for everything being normalized?
    """
    for shock in ratio_map:
        if shock in df.columns:
            print('normalizing: ', shock, ratio_map[shock])
            df[shock] = df[shock]*ratio_map[shock]
    return df

"""
Example: 
ratio_map = {'okja': 1.0, 'tgc': 0.5, 'cowspiracy': 0.1}, 
map_reference = 'okja'
new_reference: 'tgc'
new_map = {'tgc': 1.0, 'okja': 2.0, 'cowspiracy': 0.2}
"""
def generate_new_ratio_map(ratio_map, map_reference, new_reference):
    new_map = {}
    for key in ratio_map:
        new_map[key] = ratio_map[key]/ratio_map[new_reference]
    
    return new_map

def plot_fig(merged, cols, month_interval=3, col_map=None, date_col='ds', title=None, logscale=False):
    dec_idx = 0
    plt.figure(figsize=(15,5))
    time_name = 'T'
    merged[time_name] = pd.to_datetime(merged[date_col], infer_datetime_format=True, errors='coerce')
        
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    print(merged[time_name])
    for col in cols:
        if col_map:
            plt.plot(merged[time_name], merged[col],linewidth=5,label=col_map[col], alpha=0.5)
        else:
            plt.plot(merged[time_name], merged[col],linewidth=5,label=col, alpha=0.5)            


    plt.gcf().autofmt_xdate()    
    plt.legend()
    if title:
        plt.title(title)
    
    if logscale:
        plt.yscale('log')
    
    plt.show()
   