from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def resample_df(original_df: pd.DataFrame, aug_df: pd.DataFrame, label_col: str, sampling_strategy: dict, seed: int = None) -> pd.DataFrame:
    """
    Resamples a DataFrame based on a class-wise sampling strategy, optionally using augmented data.

    Args:
        original_df (pd.DataFrame): Original labeled data.
        aug_df (pd.DataFrame): Augmented data to use when more samples are needed.
        label_col (str): Name of the label column.
        sampling_strategy (dict): Dict mapping label -> desired sample count.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    resampled_dfs = []

    for label, target_count in sampling_strategy.items():
        orig_class_df = original_df[original_df[label_col] == label]
        count_orig = len(orig_class_df)

        if target_count <= count_orig:
            sampled_orig = orig_class_df.sample(n=target_count, random_state=seed)
            resampled_dfs.append(sampled_orig)
        else:
            needed = target_count - count_orig
            aug_class_df = aug_df[aug_df[label_col] == label]

            if len(aug_class_df) >= needed:
                sampled_aug = aug_class_df.sample(n=needed, random_state=seed)
            else:
                sampled_aug = aug_class_df

            combined = pd.concat([orig_class_df, sampled_aug], axis=0)
            resampled_dfs.append(combined)

    return pd.concat(resampled_dfs, axis=0).reset_index(drop=True)


def resample_data(X, y, sampling_strategy='auto', random_state=42):
    """
    Applies undersampling followed by SMOTE oversampling.

    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target vector.
        sampling_strategy (str or dict): Strategy for SMOTE (oversampling). 
                                         Undersampling is done to the minority class by default.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Resampled (X_res, y_res)
    """
    class_counts = Counter(y)
    
    # Step 1: Undersample all classes to the size of the smallest class
    rus_strategy = {
        cls: min(class_counts[cls], sampling_strategy.get(cls, class_counts[cls]))
        for cls in class_counts
    }
    rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)

    # Step 2: SMOTE with user-specified strategy
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_res, y_res)

    return X_res, y_res

def resample_data_with_gpt(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_gpt: np.ndarray,
        y_gpt: np.ndarray,
        sampling_strategy: dict, 
        seed: int):
    
    rng = np.random.default_rng(seed)
    x_resampled, y_resampled = [], []

    for label, target_count in sampling_strategy.items():
        # Original samples of this class
        idx_orig = np.where(y_train == label)[0]
        count_orig = len(idx_orig)

        if target_count <= count_orig:
            sampled_idx = rng.choice(idx_orig, size=target_count, replace=False)
            x_resampled.append(x_train[sampled_idx])
            y_resampled.append(y_train[sampled_idx])
        else:
            needed = target_count - count_orig
            idx_aug = np.where(y_gpt == label)[0]

            if len(idx_aug) >= needed:
                sampled_aug_idx = rng.choice(idx_aug, size=needed, replace=False)
            else:
                sampled_aug_idx = idx_aug  # use all available

            x_combined = np.vstack([x_train[idx_orig], x_gpt[sampled_aug_idx]])
            y_combined = np.hstack([y_train[idx_orig], y_gpt[sampled_aug_idx]])

            x_resampled.append(x_combined)
            y_resampled.append(y_combined)

    x_final = np.vstack(x_resampled)
    y_final = np.hstack(y_resampled)

    return x_final, y_final

import numpy as np

def resample_sequence_data_with_gpt(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_gpt: np.ndarray,
        y_gpt: np.ndarray,
        sampling_strategy: dict,
        seed: int):

    rng = np.random.default_rng(seed)
    x_resampled, y_resampled = [], []

    for label, target_count in sampling_strategy.items():
        idx_orig = np.where(y_train == label)[0]
        count_orig = len(idx_orig)

        if target_count <= count_orig:
            sampled_idx = rng.choice(idx_orig, size=target_count, replace=False)
            x_resampled.append(x_train[sampled_idx])
            y_resampled.append(y_train[sampled_idx])
        else:
            needed = target_count - count_orig
            idx_aug = np.where(y_gpt == label)[0]

            if len(idx_aug) >= needed:
                sampled_aug_idx = rng.choice(idx_aug, size=needed, replace=False)
            else:
                sampled_aug_idx = idx_aug  # usar todos los que haya

            x_combined = np.concatenate([x_train[idx_orig], x_gpt[sampled_aug_idx]], axis=0)
            y_combined = np.concatenate([y_train[idx_orig], y_gpt[sampled_aug_idx]], axis=0)

            x_resampled.append(x_combined)
            y_resampled.append(y_combined)

    x_final = np.concatenate(x_resampled, axis=0)
    y_final = np.concatenate(y_resampled, axis=0)

    return x_final, y_final
    




    


