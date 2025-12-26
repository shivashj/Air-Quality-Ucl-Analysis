"""
Main Pipeline for IoT Air Quality Prediction Project

This script orchestrates the full workflow:
- Data loading
- Cleaning
- Preprocessing
- Saving processed data
- Model training
- Evaluation
"""
from exceptions.custom_exceptions import (
    DataNotFoundError,
    EmptyDatasetError,
    PreprocessingError,
    ModelTrainingError
)

from pathlib import Path

from data_acquisition.load_data import load_raw_air_quality_data
from data_acquisition.cleaning import handle_missing_values

from preprocessing.preprocess_data import preprocess_data

from models.train_model import train_random_forest

from evaluation.evaluate import run_evaluation


##########################
from config.setting import PROCESSED_DATA_DIR
import pandas as pd

from data_acquisition.synthetic_air_quality import generate_synthetic_air_quality_data
from config.setting import RAW_DATA_DIR, SYNTHETIC_DATA_FILE
import os
def main():
    print("Starting IoT Air Quality Prediction Pipeline...")
    


    from config.setting import (
    USE_SYNTHETIC_DATA,
    REAL_DATA_FILE,
    SYNTHETIC_DATA_FILE
)


    def load_selected_dataset():
        
        # اگر فایل synthetic وجود ندارد، بسازش
        if not SYNTHETIC_DATA_FILE.exists():
            print(" Synthetic dataset not found. Generating it now...")
            generate_synthetic_air_quality_data(
                periods=2000,
                save_path=SYNTHETIC_DATA_FILE
            )
        else:
            print("Synthetic dataset already exists." )
        
        
        
        
        
        if USE_SYNTHETIC_DATA:
           print(" Using synthetic dataset")
           data_path = SYNTHETIC_DATA_FILE
        else:
          print(" Using real dataset")
          data_path = REAL_DATA_FILE

        return pd.read_csv(
          data_path,
          sep=";",
          decimal=","
    )

    # --------------------------------------------------
    # 1. Load Raw Data
    # --------------------------------------------------
    print(" Loading raw data...")
    df_raw = load_selected_dataset()
    if df_raw is None or df_raw.empty:
        raise DataNotFoundError("Raw dataset not found or empty.")

    # --------------------------------------------------
    # 2. Data Cleaning
    # --------------------------------------------------
    print(" Cleaning data...")
    df_clean = handle_missing_values(df_raw)

    # --------------------------------------------------
    # 3. Preprocessing & Feature Engineering
    # --------------------------------------------------
    print(" Preprocessing data...")
    try:
        X, y = preprocess_data(
            df=df_clean,
            timestamp_col="Date",  
             sensor_features=[
               "CO(GT)",
              "NO2(GT)",
              "NOx(GT)",
              "C6H6(GT)",
              "T",
              "RH",
              "AH"
          ],
          target_columns=["CO(GT)"],
          horizon=1,
          use_time_features=True
     )
    except Exception as e:
        raise PreprocessingError(f"Preprocessing failed: {e}")    
    if X.empty or y.empty:
        raise EmptyDatasetError("Dataset became empty after preprocessing.")

    #برای تست کردن داده های ساخته شده
    #print("@@@@@@@Label distribution:")
    #print(y.value_counts())

    # 4. Save Processed Data (IMPORTANT)

    print("Saving processed data...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_path = PROCESSED_DATA_DIR / "X_features.csv"
    y_path = PROCESSED_DATA_DIR / "y_labels.csv"

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    print(f" Features saved to: {X_path}")
    print(f" Labels saved to: {y_path}")

    # 5. Train Model

    print("Training Random Forest model...")
    try:

        model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_random_forest(
            X=X,
            y=y
        )
    except Exception as e:
        raise ModelTrainingError(f"Model training failed: {e}") 
    
        #Export trained model to ONNX
    from models.export_onnx import export_model_to_onnx

    export_model_to_onnx(
         model=model,
         n_features=X_train.shape[1],
         output_path="results/air_quality_model.onnx"
    )


    from config.setting import RESULTS_DIR
    import json

    # 7. Feature Importance

    print(" Plotting feature importance...")

        # 6. Evaluation
    print(" Evaluating model...")

    metrics = run_evaluation(
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob
    )

    print(" Evaluation completed.")

    print(" Pipeline completed successfully!")
    from evaluation.feature_importance import plot_feature_importance
    from config.setting import RESULTS_DIR

    print(" Plotting feature importance...")

    importance_df = plot_feature_importance(
        model=model,
        feature_names=X_train.columns,
        save_path=RESULTS_DIR / "feature_importance.png"
    )
    

if __name__ == "__main__":
        main()
