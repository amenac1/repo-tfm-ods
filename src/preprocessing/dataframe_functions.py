def get_superlabel2(label: int) -> int:
    """Assigns a superlabel based on the provided label.

    Args:
        label (int): The input label.

    Returns:
        int: The assigned superlabel.
    """
    superlabel = -1

    # Se ignoran [1, 2, 5, 10, 14, 16, 17] y 13 y 15 se pueden combinar

    if label in [3, 6, 11, 13]:  # Salud y medioambiente
        superlabel = 0
    elif label in [7, 9, 12]:  # Industria y consumo
        superlabel = 1
    elif label in [4, 8]:  # Sociedad
        superlabel = 2
    return superlabel


def remap_dataframe_labels(df, label_col: str = 'labels', mapping: dict = None):
    """Remap DataFrame labels to contiguous integers starting at 0, using an optional provided mapping.

    If no mapping is provided, one is created from the sorted unique labels in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_col (str, optional): Name of the column containing the labels. Defaults to 'labels'.
        mapping (dict, optional): A dictionary mapping original labels to new labels. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with remapped labels in the specified column.
        dict: The mapping of original labels to new labels.
    """
    if mapping is None:
        unique_labels = sorted(df[label_col].unique())
        mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_labels)}

    df[label_col] = df[label_col].map(mapping)
    
    return df, mapping


def remap_dataset_labels(dataset, label_column: str = 'labels', mapping: dict = None):
    from datasets import Dataset, disable_progress_bar, enable_progress_bar
    """Remap dataset labels to contiguous integers starting at 0, using an optional provided mapping.

    If no mapping is provided, one is created from the sorted unique labels in the dataset.
    This function handles cases where labels are stored as tensors.

    Args:
        dataset (datasets.Dataset or similar): The input dataset.
        label_column (str, optional): Name of the column with original labels. Defaults to 'labels'.
        mapping (dict, optional): A dictionary mapping original labels to new labels. Defaults to None.

    Returns:
        tuple: A tuple (remapped_dataset, mapping) where:
            - remapped_dataset: Dataset with remapped labels.
            - mapping (dict): Dictionary mapping original labels to new labels.
    """
    if mapping is None:
        unique_labels = sorted({int(x) if hasattr(x, 'item') else int(x) for x in dataset[label_column]})
        mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_labels)}

    def map_label(example):
        label = example[label_column]
        if hasattr(label, 'item'):
            label = label.item()
        example[label_column] = mapping[label]
        return example

    disable_progress_bar
    remapped_dataset = dataset.map(map_label)
    enable_progress_bar
    return remapped_dataset, mapping


def tokenize_test_dataframe(df, text_column: str, label_column: str, tokenizer, max_length: int = 512):
    from datasets import Dataset, disable_progress_bar, enable_progress_bar
    import torch
    """Convert a pandas DataFrame to a tokenized Hugging Face dataset without labels and return a tensor of labels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing the text.
        label_column (str): Name of the column containing the labels.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        max_length (int, optional): Maximum sequence length. Defaults to 512.

    Returns:
        tokenized_dataset (datasets.Dataset): Tokenized dataset without the 'labels' column.
        labels (torch.Tensor): Tensor containing the labels.
    """
    labels = torch.tensor(df[label_column].values)
    dataset = Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset, labels



