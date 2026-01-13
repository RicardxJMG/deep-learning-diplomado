import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from typing import List, Union, Optional, Dict, Any, Tuple, Union
from pathlib import Path

SEED = 7
np.random.seed(SEED)



class DataProcessor:
   
    def __init__(
            self,
            cols_num: Optional[List[str]] = None,
            cols_cat: Optional[List[str]] = None,
            cols_onehot: Optional[List[str]] = None,
            cols_ordinal: Optional[List[str]] = None,
            ordinal_categories: Optional[List[List[str]]] = None,
            target: Optional[str] = None,
            stratify: Union[bool, str] = False ) -> None: 
        """
        Procesador genérico para transformación de datos.
        
        Parámetros:
        -----------
        cols_num : list
            Lista de columnas numéricas
        cols_cat : list
            Lista de columnas categóricas
        cols_onehot : list
            Subconjunto de cols_cat para codificar con OneHotEncoder
        cols_ordinal : list
            Subconjunto de cols_cat para codificar con OrdinalEncoder
        categorias_ordinales : list of lists
            Categorías en orden para cada variable ordinal
        target : str
            Nombre de la columna objetivo
        stratify : bool or str
            Si es True, estratifica por la columna objetivo (para clasificación).
            Si es string, usa esa columna para estratificar.
            Si es False, no estratifica.
        """
        self.cols_num = cols_num or []
        self.cols_cat = cols_cat or []
        self.cols_onehot = cols_onehot or []
        self.cols_ordinal = cols_ordinal or []
        self.ordinal_categories = ordinal_categories or []
        self.target = target
        self.stratify = stratify
        
        
        if self.cols_cat:
            if not set(self.cols_onehot + self.cols_ordinal).issubset(set(self.cols_cat)):
                raise ValueError("cols_onehot y cols_ordinal deben ser subconjuntos de cols_cat")
                
        if self.cols_ordinal and self.ordinal_categories:
            if len(self.cols_ordinal) != len(self.ordinal_categories):
                raise ValueError("cols_ordinal y categorias_ordinales deben tener la misma longitud")


    def load_and_split_data(self, csv_path:str | Path, test_size: float = 0.2, val_size:float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame,  pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        
        df = pd.read_csv(csv_path)
        y = df[self.target]
        X = df.drop(columns=[self.target])

        stratify_column  = y if self.stratify else None
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size =test_size, stratify=stratify_column ,random_state=SEED
        )
 
        stratify_column  = y_train_full if self.stratify else None
            
        X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,test_size=val_size, stratify = stratify_column, random_state=SEED
        )
            

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_categorical(self, X_train_cat: pd.DataFrame, X_val_cat: pd.DataFrame, X_test_cat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[ColumnTransformer], List[str]]:
        
        # en caso de que la lista de categóricas estén vacias 
        if not self.cols_cat:
            return pd.DataFrame(index=X_train_cat.index), pd.DataFrame(index=X_val_cat.index), \
                   pd.DataFrame(index=X_test_cat.index), None, []
        
        cat_preprocessor = ColumnTransformer(
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
                            categories=self.ordinal_categories,
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

        cat_preprocessor.fit(X_train_cat)

        X_train_cat_proc = cat_preprocessor.transform(X_train_cat)
        X_val_cat_proc = cat_preprocessor.transform(X_val_cat)
        X_test_cat_proc = cat_preprocessor.transform(X_test_cat)

        cols_out_cat = list(cat_preprocessor.get_feature_names_out())

        # Renombrar One-Hot
        rename_map = {}
        if len(self.cols_onehot) > 0:
            ohe = cat_preprocessor.named_transformers_["onehot"].named_steps["encoder"]
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

        return df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, cat_preprocessor, cols_out_cat

    def preprocess_numerical(self, X_train_num: pd.DataFrame, X_val_num: pd.DataFrame, X_test_num:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Pipeline]]:
        
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

    def combine_and_finalize(
        self,
        T_train_num_df: pd.DataFrame,
        T_val_num_df: pd.DataFrame,
        T_test_num_df: pd.DataFrame,
        df_train_cat_encode: pd.DataFrame,
        df_val_cat_encode: pd.DataFrame,
        df_test_cat_encode: pd.DataFrame,
        y_train: pd.Series, 
        y_val: pd.Series,
        y_test: pd.Series, 
        cols_out_cat: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        X_train_final_df = pd.concat([T_train_num_df, df_train_cat_encode], axis=1)
        X_val_final_df = pd.concat([T_val_num_df, df_val_cat_encode], axis=1)
        X_test_final_df = pd.concat([T_test_num_df, df_test_cat_encode], axis=1)

        X_train_final = X_train_final_df.to_numpy(dtype=np.float32)
        X_val_final = X_val_final_df.to_numpy(dtype=np.float32)
        X_test_final = X_test_final_df.to_numpy(dtype=np.float32)

        feature_names = list(X_train_final_df.columns)

        metadata = {
            "feature_names": feature_names,
            "target": self.target,
            "cols_num": self.cols_num,
            "cols_cat": self.cols_cat,
            "cols_onehot": self.cols_onehot,
            "cols_ordinal": self.cols_ordinal,
            "cat_out_cols": cols_out_cat
        }

        print("\n" + "="*50)
        print("Resumen del preprocesamiento:")
        print("="*50)
        print(f"X_train_final: {X_train_final.shape}")
        print(f"X_val_final  : {X_val_final.shape}")
        print(f"X_test_final : {X_test_final.shape}")
        print(f"Total de características: {len(feature_names)}")
        print(f"  - Numéricas: {len(self.cols_num)}")
        print(f"  - Categóricas procesadas: {len(cols_out_cat)}")
        
        data = { 
                'X_train': X_train_final,
                'X_val': X_val_final,
                'X_test': X_test_final,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'X_train_df': X_train_final_df,
                'X_val_df': X_val_final_df,
                'X_test_df': X_test_final_df
        }
        

        return data, metadata

    def process(self, csv_path: str | Path, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Proceso completo de preprocesamiento.
        
        Parámetros:
        -----------
        csv_path : str | Path
            Ruta al archivo CSV
        test_size : float
            Proporción para test (0-1)
        val_size : float
            Proporción para validación del train (0-1)
        """
        
        # Paso 1: Cargar y dividir
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_split_data(csv_path, test_size, val_size)
        
        # Subsets
        X_train_num = X_train[self.cols_num]
        X_train_cat = X_train[self.cols_cat]
        X_val_num = X_val[self.cols_num]
        X_val_cat = X_val[self.cols_cat]
        X_test_num = X_test[self.cols_num]
        X_test_cat = X_test[self.cols_cat]

        # Paso 2: Procesar categóricas
        df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, cat_preprocessor, cols_out_cat = self.preprocess_categorical(X_train_cat, X_val_cat, X_test_cat)

        # Paso 3: Procesar numéricas
        T_train_num_df, T_val_num_df, T_test_num_df, num_pipe = self.preprocess_numerical(X_train_num, X_val_num, X_test_num)

        # Paso 4: Combinar y finalizar
        data_final, metadata = self.combine_and_finalize(
            T_train_num_df, T_val_num_df, T_test_num_df, 
            df_train_cat_encode, df_val_cat_encode, df_test_cat_encode, 
            y_train, y_val, y_test, cols_out_cat
        )

        return  {
            'data': data_final,
            'artifacts': {
                'num_pipe': num_pipe,
                'cat_preprocessor': cat_preprocessor,
                'metadata': metadata
            }
        }

# def process_mpg_data(csv_path):
#     """
#     Procesa el conjunto de datos mpg desde un path CSV.
#     Retorna un diccionario con datos procesados y artefactos.
#     """
#     processor =  DataProcessor()
#     return processor.process(csv_path)


def process_new_data_with_artifacts(
    new_df: pd.DataFrame,
    artifacts: dict
) -> pd.DataFrame:
    """
    Procesa nuevos datos usando artefactos entrenados.
    Versión completamente defensiva.
    """

    # ========= Artefactos =========
    num_pipe = artifacts.get("num_pipe")
    cat_preprocessor = artifacts.get("cat_preprocessor")
    metadata = artifacts["metadata"]

    feature_names = metadata["feature_names"]
    cols_num = metadata.get("cols_num", [])
    cols_cat = metadata.get("cols_cat", [])
    cols_onehot = metadata.get("cols_onehot", [])
    cat_out_cols = metadata.get("cat_out_cols", [])
    target = metadata.get("target")

    new_df = new_df.copy()

    # ========= Eliminar target =========
    if target and target in new_df.columns:
        new_df.drop(columns=[target], inplace=True)

    # ========= Asegurar columnas =========
    expected_raw_cols = cols_num + cols_cat
    for col in expected_raw_cols:
        if col not in new_df.columns:
            new_df[col] = pd.NA

    new_df = new_df[expected_raw_cols]

    # ========= NUMÉRICAS =========
    if hasattr(num_pipe, "transform") and len(cols_num) > 0:
        X_new_num = new_df[cols_num]
        T_new_num_df = pd.DataFrame(
            num_pipe.transform(X_new_num),
            columns=cols_num,
            index=new_df.index
        )
    else:
        T_new_num_df = pd.DataFrame(index=new_df.index)

    # ========= CATEGÓRICAS =========
    if hasattr(cat_preprocessor, "transform") and len(cols_cat) > 0:
        X_new_cat = new_df[cols_cat]
        X_new_cat_proc = cat_preprocessor.transform(X_new_cat)

        cols_out = list(cat_preprocessor.get_feature_names_out())

        # Renombrado OneHot consistente
        rename_map = {}
        if len(cols_onehot) > 0:
            ohe = (
                cat_preprocessor
                .named_transformers_["onehot"]
                .named_steps["encoder"]
            )
            for name in ohe.get_feature_names_out(cols_onehot):
                for col in cols_onehot:
                    if name.startswith(col + "_"):
                        rename_map[name] = f"{col}___{name[len(col)+1:]}"
                        break

        cols_out = [rename_map.get(c, c) for c in cols_out]

        df_new_cat_encode = pd.DataFrame(
            X_new_cat_proc,
            columns=cols_out,
            index=new_df.index
        )

        # Alinear a entrenamiento
        for c in cat_out_cols:
            if c not in df_new_cat_encode.columns:
                df_new_cat_encode[c] = 0.0

        df_new_cat_encode = df_new_cat_encode[cat_out_cols]

    else:
        df_new_cat_encode = pd.DataFrame(index=new_df.index)

    # ========= FINAL =========
    T_new_final = pd.concat(
        [T_new_num_df, df_new_cat_encode],
        axis=1
    )

    # Forzar orden exacto
    for col in feature_names:
        if col not in T_new_final.columns:
            T_new_final[col] = 0.0

    T_new_final = T_new_final[feature_names]

    print("Nuevos datos procesados:", T_new_final.shape)

    return T_new_final
