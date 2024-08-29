import pandas as pd


def build_df_from_dict(d: dict, cols: list[str]) -> pd.DataFrame:
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
    df.columns = cols
    return df
