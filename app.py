# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:59:22 2020

@author: sherangag
"""

import pandas as pd

import markov_model_functions as mmf

import_dataset=pd.read_csv('channel attribution example.csv')

data,dataset=mmf.markov_chain(data_set=import_dataset,no_iteration=10,no_of_simulation=10000,alpha=5)
