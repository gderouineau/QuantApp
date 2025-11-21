import pandas as pd

def month_odd_even_split(index: pd.DatetimeIndex):
    """
    Renvoie deux masques booléens (train, val) selon mois impairs/pairs.
    """
    months = index.month
    train_mask = months % 2 == 1  # impairs
    val_mask = ~train_mask        # pairs
    return train_mask, val_mask

def week_odd_even_split(index: pd.DatetimeIndex):
    """
    Split par semaines iso (1..52), impaires/paires.
    """
    weeks = index.isocalendar().week.astype(int)
    train_mask = weeks % 2 == 1
    val_mask = ~train_mask
    return train_mask, val_mask
