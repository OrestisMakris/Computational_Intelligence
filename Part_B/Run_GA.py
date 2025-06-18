"""Module for Alzheimer's disease prediction and analysis using neural networks 
finding the best features using Genetic Algorithm (GA).

tjis is the main script to run the GA for feature selection.


Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""




import torch
import torch.nn as nn
import pandas as  pd
import numpy as np
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.compose import ColumnTransformer


from configGA import POPULATION_SIZE, NUM_FEATURES,  MAX_GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, TOURNAMENTSIZE, ELITISM, PENALTY_WEIGHT

from ga_utils  import genenetikos_main
from utils_nn_dataload import load_data, SimpleNet 


def run_ga():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_df, y_np, cont_cols, ord_cols, nom_cols, bin_cols = load_data('/home/st1084516/Computational_Intelligence/Part_B/alzheimers_disease_data.csv')


    problematic_column = 'DoctorInCharge'  

    x_df[problematic_column] = pd.to_numeric(x_df[problematic_column], errors='coerce').fillna(-1).astype(int)

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[

        ('cont', StandardScaler(), cont_cols),
        ('ord', StandardScaler(), ord_cols),
        ('nom', OneHotEncoder(sparse=False, handle_unknown='ignore'), nom_cols),
        ('bin', 'passthrough', bin_cols),

    ])

    x_processed = preprocessor.fit_transform(x_df)

    x_tensor = torch.tensor(x_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    best_model_part_a = SimpleNet(input_size=NUM_FEATURES)

    best_model_part_a.load_state_dict(torch.load('/home/st1084516/Computational_Intelligence/Part_B/best_model_part_a_orestis_makris.pth'))
    best_model_part_a.to(device)

    best_feature_mask, best_fitness, historyGA, num_gens = genenetikos_main(
        model=best_model_part_a, 
        x_val_full=x_tensor, 
        y_val=y_tensor, 
        device=device,
        num_features=NUM_FEATURES  # Η παράμετρος που έλειπε
    )



    print("\n--- Genetic Algorithm Finished ---")
    print(f"Best solution found after {num_gens} generations.")
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {np.sum(best_feature_mask)}")
    print(f"Feature Mask: {best_feature_mask.astype(int)}")

if __name__ == "__main__":
    run_ga()


