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

from sklearn.model_selection import StratifiedKFold
from utils_nn_dataload import train, evaluate, plot_mean_loss_curves, plot_mean_accuracy_curves
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



    print("\n Retraining & Evaluating Final new and improved  Model--")

 # μασκα 

    mask_bool = best_feature_mask.astype(bool)

    x_reduced = x_processed[:, mask_bool]


    num_selected_features = mask_bool.sum()
    print(f"Number of selected features: {num_selected_features}")
                     
    
    num_selected_features = np.sum(best_feature_mask)           


    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    hist_tr_loss, hist_val_loss = [], []
    hist_tr_acc, hist_val_acc = [], []


    for fold, (train_ids, test_ids) in enumerate(kfold.split(x_reduced, y_np)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        

        x_train, x_test = x_reduced[train_ids], x_reduced[test_ids]
        y_train, y_test = y_np[train_ids], y_np[test_ids]

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        final_model = SimpleNet(input_size=num_selected_features)
        final_model.to(device)

        optimizer = torch.optim.SGD(final_model.parameters(), lr=0.1, momentum=0.6, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

   

        print(f"εκπαίδευση {fold+1}...")


        train_losses, val_losses, train_accs, val_accs = train(
            model=final_model, 
            optimizer=optimizer, 
            criterion=criterion, 
            x_train=x_train_tensor,                        
            y_train=y_train_tensor, 
            x_val=x_test_tensor,
            y_val=y_test_tensor,
            epochs=200,                      
            device=device,
            patience=20,
            min_delta=1e-3
        )

        hist_tr_loss.append(train_losses)
        hist_val_loss.append(val_losses)
        hist_tr_acc.append(train_accs)
        hist_val_acc.append(val_accs)


  
        accuracy = evaluate(final_model, x_test_tensor, y_test_tensor, device)
        fold_accuracies.append(accuracy)
        print(f"Accuracy for fold {fold+1}: {accuracy * 100:.2f}%")


    mean_accuracy = np.mean(fold_accuracies)
    print(f"nean accuracy over {n_splits} folds: {mean_accuracy * 100:.2f}%")



    min_ep_loss = min(len(l) for l in hist_tr_loss)
    mn_tr_loss  = np.mean([l[:min_ep_loss] for l in hist_tr_loss], axis=0)
    mn_val_loss = np.mean([l[:min_ep_loss] for l in hist_val_loss], axis=0)

    plot_mean_loss_curves(mn_tr_loss, mn_val_loss, f"loss_final_model_GA_retraine.png")

    min_ep_acc = min(len(a) for a in hist_tr_acc)
    mn_tr_acc   = np.mean([a[:min_ep_acc] for a in hist_tr_acc], axis=0)
    mn_val_acc  = np.mean([a[:min_ep_acc] for a in hist_val_acc], axis=0)

    plot_mean_accuracy_curves(mn_tr_acc, mn_val_acc, f"acc_final_model_GA_retraine.png")




if __name__ == "__main__":
    run_ga()