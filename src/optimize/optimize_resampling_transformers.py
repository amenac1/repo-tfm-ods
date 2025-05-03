from src.transformers_torch.bert_torch import train_bert_with_validation
from src.preprocessing.dataframe_functions import dataframe_to_dataset

def optimize_transformer_parameters(train_dataset, val_dataset, model, ):
    def objective(trial):
        trained_model = train_bert_with_validation(train_dataset, val_dataset, model, verbose=False)