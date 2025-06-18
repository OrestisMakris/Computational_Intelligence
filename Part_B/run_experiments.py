"""Module for Alzheimer's disease prediction and analysis using neural networks 
finding the best features using Genetic Algorithm (GA).

this the experiment script to run all the experiments based on the configurations
for B3 quaestion of the project.

Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""

import torch
import pandas as pd

import  numpy as np
import matplotlib.pyplot  as plt
import  os
import  time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose  import ColumnTransformer


from  configGA  import *
from ga_utils import genenetikos_main
from utils_nn_dataload import load_data, SimpleNet

def run_all_experiments():

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x_df, y_np, cont_cols, ord_cols, nom_cols, bin_cols = load_data('/home/st1084516/Computational_Intelligence/Part_B/alzheimers_disease_data.csv')

    problematic_column = 'DoctorInCharge'  

    x_df[problematic_column] = pd.to_numeric(x_df[problematic_column], errors='coerce').fillna(-1).astype(int)


    preprocessor = ColumnTransformer(transformers=[

        ('cont', StandardScaler(), cont_cols), ('ord', StandardScaler(), ord_cols),
        ('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nom_cols),             
        ('bin', 'passthrough', bin_cols),

    ])


    x_processed = preprocessor.fit_transform(x_df)
    x_tensor = torch.tensor(x_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    model = SimpleNet(input_size=NUM_FEATURES)  

    model.load_state_dict(torch.load('/home/st1084516/Computational_Intelligence/Part_B/best_model_part_a_orestis_makris.pth'))
    model.to(device)
    
    results_data = []
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


    for config in EXPERIMENT_CONFIGS:

        start_time = time.time()
        print(f"\n--- expr run {config['id']}: POP={config['pop']}, CROSS={config['crosspro']}, MUT={config['mutpro']} ---")

        run_fitnesses, run_generations, run_histories = [], [], []


        for i in range(NUM_RUNS):
            print(f"  repeat {i+1}/{NUM_RUNS}...")
  
            _, best_fitness, history, num_gens = genenetikos_main(
                model=model, x_val_full=x_tensor, y_val=y_tensor, device=device,        
                population_size=config['pop'], 
                num_features=NUM_FEATURES,
                max_generations=MAX_GENERATIONS, 
                crossover_prob=config['crosspro'],
                mutation_prob=config['mutpro'], 
                tournamentsize=TOURNAMENTSIZE,
                elitism=ELITISM,        
                penalty_weight=PENALTY_WEIGHT, 
                patience=0.0001
            )
            run_fitnesses.append(best_fitness)      
            run_generations.append(num_gens)
            run_histories.append(history)

        # --- Επεξεργασία και Αποθήκευση Αποτελεσμάτων ---
        avg_fitness = np.mean(run_fitnesses)
        avg_generations = np.mean(run_generations)
        
        results_data.append({
            'A/A': config['id'],
            'ΜΕΓΕΘΟΣ ΠΛΗΘΥΣΜΟΥ': config['pop'],
            'ΠΙΘΑΝΟΤΗΤΑ ΔΙΑΣΤΑΥΡΩΣΗΣ': config['crosspro'],
            'ΠΙΘΑΝΟΤΗΤΑ ΜΕΤΑΛΛΑΞΗΣ': config['mutpro'],
            'ΜΕΣΗ ΤΙΜΗ ΒΕΛΤΙΣΤΟΥ': f"{avg_fitness:.4f}",
            'ΜΕΣΟΣ ΑΡΙΘΜΟΣ ΓΕΝΕΩΝ': f"{avg_generations:.1f}"
        })

        # --- Δημιουργία και Αποθήκευση Γραφήματος ---
        max_len = max(len(h) for h in run_histories)
        padded_histories = [np.pad(h, (0, max_len - len(h)), 'edge') for h in run_histories]
        mean_history = np.mean(padded_histories, axis=0)
        std_history = np.std(padded_histories, axis=0)
        
        plt.figure(figsize=(12, 7))
        plt.plot(mean_history, label='Μέση Βέλτιστη Καταλληλότητα', color="#0d00c2")
        plt.fill_between(range(max_len), mean_history - std_history, mean_history + std_history, alpha=0.2, label='Τυπική Απόκλιση')
        plt.title(f"Καμπιλη Εξελιξης - Πεεραμα {config['id']}\n(POP={config['pop']}, CROSS={config['crosspro']}, MUT={config['mutpro']})")
        plt.xlabel('Γενιά')
        plt.ylabel('Τιμη Καταλληλοτητας (Fitness)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"b3_exp_{config['id']}.png"))
        plt.close()
        
        end_time = time.time()
        print(f"Το πείραμα {config['id']} ολοκληροθηκε σε {end_time - start_time:.2f} δευτερολεπτα.")

    # --- Εμφάνιση Τελικού Πίνακα Αποτελεσμάτων ---
    results_df = pd.DataFrame(results_data)
    print("\n--- Τελικά αποτελεματα πειραμάτων ---")
    print(results_df.to_string(index=False))
    results_df.to_csv('experiment_results.csv', index=False)

if __name__ == "__main__":
    run_all_experiments()