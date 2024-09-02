import pandas as pd


def build_df_from_dict(d: dict, id_cols: list[str], value_col: str) -> pd.DataFrame:
    """Build a DataFrame with a multi-index from a multi-level dictionary of uniform depth."""

    def _recurse(d: dict) -> pd.DataFrame:
        vals = list(d.values())
        if isinstance(vals[0], dict):
            dfs = {k: _recurse(v) for k, v in d.items()}
            df = pd.concat(dfs.values(), keys=dfs.keys())
        else:
            df = pd.DataFrame.from_dict(d, orient="index")
        return df

    df = _recurse(d).reset_index()
    df.columns = id_cols + [value_col]
    return df


def flatten_dict(d: dict, parent_key: str = None, sep: str = "_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
