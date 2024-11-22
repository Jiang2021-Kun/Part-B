from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle
import argparse  # add parameter parsing

def feature_ranking_algorithm(X, y, n_jobs=None):
    features = list(X.columns)
    results = []
    n_features = len(features)
    
    for i in range(n_features):
        print(f'{i+1}/{n_features}')
        if n_jobs is None:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=n_jobs)
        rf.fit(X[features], y)
        
        oob_predictions = rf.oob_prediction_
        oob_rmse = np.sqrt(mean_squared_error(y, oob_predictions))
        oob_r2 = r2_score(y, oob_predictions)
        
        importance = pd.Series(rf.feature_importances_, index=features)
        least_important = importance.idxmin()
        
        results.append({
            'removed_feature': least_important,
            'importance': importance[least_important],
            'oob_rmse': oob_rmse,
            'oob_r2': oob_r2,
            'remaining_features': len(features),
            'features': features.copy(),
        })
        
        features.remove(least_important)
    
    return pd.DataFrame(results)


def process_synop_group(synop_code, group_data, feature_cols, target_type):
    print(f"\nStart handle SYNOP code {synop_code}")
    
    X_group = group_data[feature_cols].drop('SYNOPCode', axis=1)
    y_group = group_data[f'{target_type}_Att']  # use parameter to determine target variable
    results = feature_ranking_algorithm(X_group, y_group)
    
    # create result directory, include target type
    result_dir = f'results_{target_type.lower()}'
    os.makedirs(result_dir, exist_ok=True)
    
    # save results, include target type
    with open(f'{result_dir}/synop_{synop_code}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"SYNOP {synop_code} handle complete!")

if __name__ == '__main__':
    # set command line parameters
    parser = argparse.ArgumentParser(description='Feature selection for RFL/FSO')
    parser.add_argument('target_type', choices=['RFL', 'FSO'], 
                       help='Target type to process (RFL or FSO)')
    args = parser.parse_args()
    
    # load data
    with open('data/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)


    with open('data/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)


    target_synop_codes = [0, 3, 4, 5, 6, 7, 8]

    tasks = [
        (code, 
        processed_data[processed_data['SYNOPCode'] == code],
        feature_cols, 
        args.target_type)
        for code in target_synop_codes
    ]

    # use multiprocessing to handle
    with Pool() as pool:
        pool.starmap(process_synop_group, tasks)
    
    print(f"\nAll SYNOP groups handle complete for {args.target_type}!")



