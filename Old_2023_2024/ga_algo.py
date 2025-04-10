import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv('./Data/iphi2802.csv', encoding='utf-8')

# Filter data by region_main_id = 1683
filtered_data = data[data['region_main_id'] == 1683]

#print
print(filtered_data.head())