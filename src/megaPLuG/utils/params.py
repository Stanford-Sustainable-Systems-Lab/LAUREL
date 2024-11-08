import numpy as np
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet


def set_entity_params(
    entities: pd.DataFrame | DwellSet, params: dict
) -> pd.DataFrame | DwellSet:
    """Set entity parameters in advance of simulation."""
    # Seed is based on master seed and entity's id to ensure that entities are
    # individually controllable without impacting all other entities.
    entity_is_dwellset = isinstance(entities, DwellSet)
    if entity_is_dwellset:
        return_val = entities
        entities = entities.data

    orig_idx = entities.index.names
    entities = entities.reset_index()

    for k, v in params.items():
        if isinstance(v, dict) and set(v.keys()) == {"id_columns", "values"}:
            # If this is a merge-type param, sensitive to already-defined parameters
            par_df = build_df_from_dict(
                d=v["values"], id_cols=v["id_columns"], value_col=k
            )
            entities = entities.merge(
                par_df, how="left", on=v["id_columns"], indicator="_mrg"
            )
            if np.any(entities["_mrg"] == "left_only"):
                raise RuntimeError(
                    f"Parameter values for {k} do not cover all entities."
                )
            else:
                entities = entities.drop(columns=["_mrg"])
        elif k == "random_seed":
            # Each entity gets its own independent seed controlled by the master
            entities["random_seed"] = entities[v["seed_id_col"]] + v["master_seed"]
        else:
            # If this is a parameter with the same value across entities
            flat = flatten_dict({k: v})
            for col, val in flat.items():
                entities[col] = val

    entities = entities.set_index(orig_idx)

    if entity_is_dwellset:
        return_val.data = entities
        return return_val
    else:
        return entities


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
