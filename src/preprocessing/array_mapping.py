import numpy as np

def map_labels(y: np.ndarray, mapping_dict=None) -> tuple:
    # Generar el diccionario de mapeo a partir de los labels de train
    if mapping_dict is None:
        unique_labels = np.unique(y)
        mapping_dict = {old: new for new, old in enumerate(unique_labels)}
    
    # Función vectorizada para aplicar el mapeo
    mapper = np.vectorize(mapping_dict.get)
    
    mapped_y = mapper(y)
    
    return mapped_y, mapping_dict


def unmap_labels(mapped_train: np.ndarray,
                       mapping_dict: dict,
                       mapped_val: np.ndarray = None,
                       mapped_test: np.ndarray = None) -> tuple:
    """
    Revert mapping from consecutive labels back to the original labels for train,
    and optionally for validation and test sets.
    
    Args:
        mapped_train (np.ndarray): Array of mapped training labels.
        mapping_dict (dict): Dictionary used for mapping {original: mapped}.
        mapped_val (np.ndarray, optional): Array of mapped validation labels.
        mapped_test (np.ndarray, optional): Array of mapped test labels.
    
    Returns:
        train_labels (np.ndarray): Unmapped training labels.
        unmapped_val (np.ndarray or None): Unmapped validation labels (None if not provided).
        unmapped_test (np.ndarray or None): Unmapped test labels (None if not provided).
    """
    # Construir el diccionario inverso
    reverse_mapping = {v: k for k, v in mapping_dict.items()}
    unmapper = np.vectorize(reverse_mapping.get)
    
    train_labels = unmapper(mapped_train)
    unmapped_val = unmapper(mapped_val) if mapped_val is not None else None
    unmapped_test = unmapper(mapped_test) if mapped_test is not None else None
    
    return train_labels, unmapped_val, unmapped_test

def map_df(df, label_column):
    label_mapping = {3: 0, 4: 1, 6: 2, 7: 3, 8: 4, 9: 5, 11: 6, 12: 7, 13: 8}
    df[label_column] = df[label_column].map(label_mapping)
    return df

def group_sdgs(y):
    """
    Agrupa índices de ejemplos en función de la agrupación fija de ODS:
      - ODS 3, 6, 11, 13 -> etiqueta 0
      - ODS 7, 9, 12   -> etiqueta 1
      - ODS 4, 8      -> etiqueta 2
      
    Si se encuentra un ODS diferente, se lanza un ValueError.
    
    Parámetros:
      y : iterable
          Etiquetas de entrenamiento.
      y_val : iterable, opcional
          Etiquetas de validación.
      y_test : iterable, opcional
          Etiquetas de test.
          
    Retorna:
      groups : dict
          Diccionario con las claves 'train', 'val' y 'test' (si se proporcionan), en el que cada valor es
          un diccionario que mapea la etiqueta agrupada (0, 1 o 2) a una lista de índices.
          
          Ejemplo:
          {
              'train': {0: [índices], 1: [índices], 2: [índices]},
              'val': {0: [índices], 1: [índices], 2: [índices]},
              'test': {0: [índices], 1: [índices], 2: [índices]}
          }
    """
    # Define mapping
    mapping = {3: 0, 6: 0, 11: 0, 13: 0,
               7: 1, 9: 1, 12: 1,
               4: 2, 8: 2}

    mapper = np.vectorize(mapping.get)

    y_grouped = mapper(y)

    return y_grouped



    



