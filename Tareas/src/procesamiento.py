import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

SEED = 7
np.random.seed(SEED)

class MPGDataProcessor:
    def __init__(self):
        self.cols_num = ['displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'cylinders']
        self.cols_cat = ['origin']
        self.cols_onehot = ['origin']
        self.cols_ordinal = []
        self.categorias_ordinales = []
        self.target = "mpg"

    def load_and_split_data(self, csv_path):
        df = pd.read_csv(csv_path)
        y = df[self.target]
        X = df.drop(columns=[self.target])

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=SEED
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_categorical(self, X_train_cat, X_val_cat, X_test_cat):
        preprocessor_cat = ColumnTransformer(
            transformers=[
                (
                    'onehot',
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore"))
                    ]),
                    self.cols_onehot
                ),
                (
                    'ordinal',
                    Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy="most_frequent")),
                        ('encoder', OrdinalEncoder(
                            categories=self.categorias_ordinales,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        ))
                    ]),
                    self.cols_ordinal
                )
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

        preprocessor_cat.fit(X_train_cat)

        X_train_cat_proc = preprocessor_cat.transform(X_train_cat)
        X_val_cat_proc = preprocessor_cat.transform(X_val_cat)
        X_test_cat_proc = preprocessor_cat.transform(X_test_cat)

        cols_out_cat = list(preprocessor_cat.get_feature_names_out())

        # Renombrar One-Hot
        rename_map = {}
        if len(self.cols_onehot) > 0:
            ohe = preprocessor_cat.named_transformers_["onehot"].named_steps["encoder"]
            ohe_names = list(ohe.get_feature_names_out(self.cols_onehot))
            for name in ohe_names:
                for col in self.cols_onehot:
                    prefix = col + "_"
                    if name.startswith(prefix):
                        cat = name[len(prefix):]
                        rename_map[name] = f"{col}___{cat}"
                        break

        cols_out_cat = [rename_map.get(c, c) for c in cols_out_cat]

        df_train_cat_encode = pd.DataFrame(X_train_cat_proc, columns=cols_out_cat, index=X_train_cat.index)
        df_val_cat_encode = pd.DataFrame(X_val_cat_proc, columns=cols_out_cat, index=X_val_cat.index)
        df_test_cat_encode = pd.DataFrame(X_test_cat_proc, columns=cols_out_cat, index=X_test_cat.index)

        return df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, preprocessor_cat, cols_out_cat

    def preprocess_numerical(self, X_train_num, X_val_num, X_test_num):
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        num_pipe.fit(X_train_num)

        T_train_num = num_pipe.transform(X_train_num)
        T_val_num = num_pipe.transform(X_val_num)
        T_test_num = num_pipe.transform(X_test_num)

        T_train_num_df = pd.DataFrame(T_train_num, columns=self.cols_num, index=X_train_num.index)
        T_val_num_df = pd.DataFrame(T_val_num, columns=self.cols_num, index=X_val_num.index)
        T_test_num_df = pd.DataFrame(T_test_num, columns=self.cols_num, index=X_test_num.index)

        return T_train_num_df, T_val_num_df, T_test_num_df, num_pipe

    def combine_and_finalize(self, T_train_num_df, T_val_num_df, T_test_num_df, df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, y_train, y_val, y_test, cols_out_cat):
        X_train_final_df = pd.concat([T_train_num_df, df_train_cat_encode], axis=1)
        X_val_final_df = pd.concat([T_val_num_df, df_val_cat_encode], axis=1)
        X_test_final_df = pd.concat([T_test_num_df, df_test_cat_encode], axis=1)

        X_train_final = X_train_final_df.to_numpy(dtype=np.float32)
        X_val_final = X_val_final_df.to_numpy(dtype=np.float32)
        X_test_final = X_test_final_df.to_numpy(dtype=np.float32)

        feature_names = list(X_train_final_df.columns)

        metadata = {
            "cols_num": self.cols_num,
            "cols_cat": self.cols_cat,
            "cols_onehot": self.cols_onehot,
            "cols_ordinal": self.cols_ordinal,
            "cat_out_cols": cols_out_cat,
            "feature_names": feature_names,
            "target": self.target
        }

        print("X_train_final:", X_train_final.shape)
        print("X_val_final  :", X_val_final.shape)
        print("X_test_final :", X_test_final.shape)

        return X_train_final, X_val_final, X_test_final, X_train_final_df, X_val_final_df, X_test_final_df, feature_names, metadata

    def process(self, csv_path):
        # Paso 1: Cargar y dividir
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_split_data(csv_path)

        # Subsets
        X_train_num = X_train[self.cols_num]
        X_train_cat = X_train[self.cols_cat]
        X_val_num = X_val[self.cols_num]
        X_val_cat = X_val[self.cols_cat]
        X_test_num = X_test[self.cols_num]
        X_test_cat = X_test[self.cols_cat]

        # Paso 2: Procesar categóricas
        df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, preprocessor_cat, cols_out_cat = self.preprocess_categorical(X_train_cat, X_val_cat, X_test_cat)

        # Paso 3: Procesar numéricas
        T_train_num_df, T_val_num_df, T_test_num_df, num_pipe = self.preprocess_numerical(X_train_num, X_val_num, X_test_num)

        # Paso 4: Combinar y finalizar
        X_train_final, X_val_final, X_test_final, X_train_final_df, X_val_final_df, X_test_final_df, feature_names, metadata = self.combine_and_finalize(
            T_train_num_df, T_val_num_df, T_test_num_df, df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, y_train, y_val, y_test, cols_out_cat
        )

        # Retornar diccionario
        result = {
            'data': {
                'X_train': X_train_final,
                'X_val': X_val_final,
                'X_test': X_test_final,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'X_train_df': X_train_final_df,
                'X_val_df': X_val_final_df,
                'X_test_df': X_test_final_df
            },
            'artifacts': {
                'num_pipe': num_pipe,
                'cat_preprocessor': preprocessor_cat,
                'feature_names': feature_names,
                'metadata': metadata
            }
        }

        return result

def process_mpg_data(csv_path):
    """
    Procesa el conjunto de datos mpg desde un path CSV.
    Retorna un diccionario con datos procesados y artefactos.
    """
    processor = MPGDataProcessor()
    return processor.process(csv_path)

def process_new_data_with_artifacts(new_df, num_pipe, preprocessor_cat, metadata, feature_names):
    """
    Procesa nuevos datos usando los artefactos ya entrenados.
    Retorna el DataFrame procesado listo para predicción.
    """
    cols_num = metadata["cols_num"]
    cols_cat = metadata["cols_cat"]
    cols_onehot = metadata["cols_onehot"]
    cat_out_cols = metadata["cat_out_cols"]
    target = metadata["target"]

    # Si por error viene el target, lo quitamos
    for possible_target in [target]:
        if possible_target in new_df.columns:
            new_df = new_df.drop(columns=[possible_target])

    # Asegurar columnas crudas (faltantes -> NA; extras se ignoran)
    expected_raw = cols_num + cols_cat
    for c in expected_raw:
        if c not in new_df.columns:
            new_df[c] = pd.NA
    new_df = new_df[expected_raw]

    X_new_num = new_df[cols_num]
    X_new_cat = new_df[cols_cat]

    # === Categóricas (mismo encoder, sin re-ajustar) ===
    X_new_cat_proc = preprocessor_cat.transform(X_new_cat)

    # reconstruir nombres de salida categórica EXACTOS como en entrenamiento
    cols_out_cat = list(preprocessor_cat.get_feature_names_out())

    rename_map = {}
    if len(cols_onehot) > 0:
        ohe = preprocessor_cat.named_transformers_["onehot"].named_steps["encoder"]
        ohe_names = list(ohe.get_feature_names_out(cols_onehot))
        for name in ohe_names:
            for col in cols_onehot:
                prefix = col + "_"
                if name.startswith(prefix):
                    cat = name[len(prefix):]
                    rename_map[name] = f"{col}___{cat}"
                    break

    cols_out_cat = [rename_map.get(c, c) for c in cols_out_cat]

    df_new_cat_encode = pd.DataFrame(X_new_cat_proc, columns=cols_out_cat, index=new_df.index)

    # Alinear a cat_out_cols (si faltan columnas porque no apareció alguna categoría -> 0)
    for c in cat_out_cols:
        if c not in df_new_cat_encode.columns:
            df_new_cat_encode[c] = 0.0
    df_new_cat_encode = df_new_cat_encode[cat_out_cols]

    # === Numéricas ===
    T_new_num = num_pipe.transform(X_new_num)
    T_new_num_df = pd.DataFrame(T_new_num, columns=cols_num, index=new_df.index)

    # === Final (num + cat) ===
    T_new_final = pd.concat([T_new_num_df, df_new_cat_encode], axis=1)

    # Forzar el orden final exacto
    for c in feature_names:
        if c not in T_new_final.columns:
            T_new_final[c] = 0.0
    T_new_final = T_new_final[feature_names]

    print("Nuevos datos procesados:", T_new_final.shape)

    return T_new_final