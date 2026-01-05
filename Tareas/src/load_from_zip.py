import zipfile
import os
import joblib
import json
import pandas as pd
import shutil

def load_processed_data_from_zip(zip_path, prinpath = False):
    """
    Carga los datos procesados y artefactos desde un archivo zip.
    Retorna train_final, val_final, test_final, num_pipe, preprocessor_cat, feature_names, metadata
    """
    temp_dir = 'temp_load_dir'
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(temp_dir)

    # Cargar artefactos
    num_pipe = joblib.load(os.path.join(temp_dir, 'num_pipe.joblib'))
    preprocessor_cat = joblib.load(os.path.join(temp_dir, 'cat_preprocessor.joblib'))
    feature_names = joblib.load(os.path.join(temp_dir, 'feature_names.joblib'))

    # Cargar datasets
    train_final = pd.read_csv(os.path.join(temp_dir, 'train_final.csv'))
    val_final = pd.read_csv(os.path.join(temp_dir, 'val_final.csv'))
    test_final = pd.read_csv(os.path.join(temp_dir, 'test_final.csv'))

    # Cargar metadata
    with open(os.path.join(temp_dir, 'metadata_preprocesamiento.json'), 'r') as f:
        metadata = json.load(f)

    # Limpiar
    shutil.rmtree(temp_dir)

    if prinpath:
        print(f"Datos extraidos desde {zip_path} correctamente")
    else: 
        print("Datos extraidos correctamente")
    return train_final, val_final, test_final, num_pipe, preprocessor_cat, feature_names, metadata