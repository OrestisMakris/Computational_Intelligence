"""Module for Alzheimer's disease prediction and analysis using neural networks.

This module contains helper functions for loading data, performing Genetech Algorithm (GA) 
operations for feature selection,preprocessing it, and evaluating the performance of neural network models.
Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def initialize_population(pop_size, num_features):      
    """ Δημιουργεί αρχικό πληθυσμό: τυχαία δυαδικά διανύσματα, 
        με τουλάχιστον ένα bit = 1."""
    
    population = []
    for _ in range(pop_size):
        individual = np.random.randint(0, 2, size=num_features)
        # Ensure at least one feature is selected
        if np.sum(individual) == 0:
            individual[np.random.randint(0, num_features)] = 1
        population.append(individual)
    return np.array(population)
       

