import zipfile
import os
import joblib
import json
import pandas as pd 


def save_processed_data_to_zip(zip_path, processed_data, printpath = False):
    """
    Guarda los datos procesados y artefactos en un archivo zip.
    processed_data es el diccionario retornado por process_mpg_data.
    """
    data = processed_data['data']
    artifacts = processed_data['artifacts']

    # Crear dfs con target
    train_final = data['X_train_df'].copy()
    train_final["target"] = data['y_train'].loc[data['X_train_df'].index].to_numpy()

    val_final = data['X_val_df'].copy()
    val_final["target"] = data['y_val'].loc[data['X_val_df'].index].to_numpy()

    test_final = data['X_test_df'].copy()
    test_final["target"] = data['y_test'].loc[data['X_test_df'].index].to_numpy()

    # Crear zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Guardar transformadores
        joblib.dump(artifacts['num_pipe'], 'num_pipe.joblib')
        zipf.write('num_pipe.joblib')
        os.remove('num_pipe.joblib')

        joblib.dump(artifacts['cat_preprocessor'], 'cat_preprocessor.joblib')
        zipf.write('cat_preprocessor.joblib')
        os.remove('cat_preprocessor.joblib')

        joblib.dump(artifacts['feature_names'], 'feature_names.joblib')
        zipf.write('feature_names.joblib')
        os.remove('feature_names.joblib')

        # Datasets
        train_final.to_csv('train_final.csv', index=False)
        zipf.write('train_final.csv')
        os.remove('train_final.csv')

        val_final.to_csv('val_final.csv', index=False)
        zipf.write('val_final.csv')
        os.remove('val_final.csv')

        test_final.to_csv('test_final.csv', index=False)
        zipf.write('test_final.csv')
        os.remove('test_final.csv')

        # Metadata
        with open('metadata_preprocesamiento.json', 'w') as f:
            json.dump(artifacts['metadata'], f, indent=2)
        zipf.write('metadata_preprocesamiento.json')
        os.remove('metadata_preprocesamiento.json')
   
    if printpath:
        print(f"Datos procesados guardados en {zip_path}")
    else: 
        print(f"Datos procesados guardados")