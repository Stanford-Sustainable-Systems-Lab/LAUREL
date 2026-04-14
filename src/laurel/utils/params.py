"""Utilities for setting per-entity parameters and extracting scenario configs.

In LAUREL, each vehicle (and, in some sub-pipelines, each hexagon) carries
its own parameter values — battery size, charging power, random seed — stored
as columns on the entity DataFrame.  This module provides the helpers needed
to populate those columns from YAML parameter dicts, to extract select
parameters from nested scenario configs for reporting, and to import classes
by dotted-path string (used for dynamically loading model components from
YAML).

Key design decisions
--------------------
- **Three param patterns**: :func:`set_entity_params` recognises three shapes
  of parameter value: (1) a flat scalar/string applied uniformly to all
  entities, (2) a ``{id_columns, values}`` dict that maps entity-level ids to
  different values via a left-join, and (3) a special ``random_seed`` pattern
  that derives a per-entity seed from a master seed plus an entity ID, ensuring
  reproducible but independent stochasticity per vehicle.
- **Integrity check on merge-type params**: if any entity fails to match the
  provided lookup table, :func:`set_entity_params` raises immediately rather
  than silently propagating NaNs.
"""

import importlib
from typing import Any

import numpy as np
import pandas as pd

from laurel.models.dwell_sets import DwellSet


# ruff: noqa: PLR0912
def set_entity_params(
    entities: pd.DataFrame | DwellSet, params: dict
) -> pd.DataFrame | DwellSet:
    """Attach scenario parameters to an entity DataFrame as new columns.

    Iterates over each key–value pair in ``params`` and adds the corresponding
    column to ``entities`` according to one of three patterns:

    1. **Merge-type** — if the value is a dict with keys ``id_columns`` and
       ``values``, builds a lookup table via :func:`build_df_from_dict` and
       left-joins it onto ``entities``.  Raises if any entity is unmatched.
    2. **Random seed** — if the key is ``"random_seed"``, computes
       ``entity[seed_id_col] + master_seed`` so every entity has an
       independent but deterministic seed.
    3. **Scalar** — any other value is flattened (nested dicts become
       ``"parent_child"`` column names) and broadcast uniformly.

    ``DwellSet`` inputs are handled transparently: the underlying DataFrame is
    modified and wrapped back into the original ``DwellSet``.

    Args:
        entities: DataFrame or DwellSet of entities to parameterise.
        params: Mapping of column name to value.  Each value may be:

            - A scalar (int, float, str) applied uniformly to all rows.
            - A nested dict, which is flattened with ``_`` separators.
            - A dict ``{"id_columns": [...], "values": {...}}`` for entity-
              specific lookups (see :func:`build_df_from_dict`).
            - A dict ``{"seed_id_col": str, "master_seed": int}`` when the key
              is ``"random_seed"``.

    Returns:
        ``entities`` with one new column per key in ``params``.

    Raises:
        RuntimeError: If a merge-type param does not cover all entities.
    """
    # Seed is based on master seed and entity's id to ensure that entities are
    # individually controllable without impacting all other entities.
    entity_is_dwellset = isinstance(entities, DwellSet)
    if entity_is_dwellset:
        return_val = entities
        entities = entities.data

    orig_idx = entities.index.names
    if orig_idx != [None]:
        entities = entities.reset_index()

    for k, v in params.items():
        if isinstance(v, dict) and set(v.keys()) == {"id_columns", "values"}:
            # If this is a merge-type param, sensitive to already-defined parameters
            par_df = build_df_from_dict(
                d=v["values"], id_cols=v["id_columns"], value_col=k
            )
            if par_df[k].dtype == np.dtype("O"):
                par_df[k] = pd.Categorical(par_df[k])
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
                if entities[col].dtype == np.dtype("O"):
                    entities[col] = pd.Categorical(entities[col])
    if orig_idx != [None]:
        entities = entities.set_index(orig_idx)

    if entity_is_dwellset:
        return_val.data = entities
        return return_val
    else:
        return entities


def build_df_from_dict(d: dict, id_cols: list[str], value_col: str) -> pd.DataFrame:
    """Build a flat DataFrame from a uniformly-nested dict, with one column per id level.

    Recursively expands nested dicts of uniform depth into a DataFrame whose
    first columns are the nested keys (matching ``id_cols``) and whose last
    column is the leaf value (``value_col``).  Leaf values may be scalars or
    lists (the latter are stored as a single array-valued column).

    Args:
        d: Nested dict of uniform depth.  Keys at each level become one id
           column.
        id_cols: Column names for the key levels, in nesting order.  Must have
            length equal to the nesting depth of ``d``.
        value_col: Name for the leaf-value column.

    Returns:
        Flat DataFrame with columns ``id_cols + [value_col]``.
    """

    def _recurse(d: dict) -> pd.DataFrame:
        vals = list(d.values())
        if isinstance(vals[0], dict):
            dfs = {k: _recurse(v) for k, v in d.items()}
            df = pd.concat(dfs.values(), keys=dfs.keys())
        else:
            if isinstance(vals[0], list):
                d = {
                    k: [np.array(v)] for k, v in d.items()
                }  # Wrapping in lists to create an array column
            df = pd.DataFrame.from_dict(d, orient="index")
        return df

    df = _recurse(d).reset_index()
    df.columns = id_cols + [value_col]
    return df


def flatten_dict(d: dict, parent_key: str = None, sep: str = "_") -> dict:
    """Flatten a nested dict to a single level using ``sep``-joined keys.

    Args:
        d: Arbitrarily nested dictionary.
        parent_key: Prefix to prepend to all keys at this level (used in
            recursion; pass ``None`` for the top-level call).
        sep: Separator inserted between parent and child key names.

    Returns:
        Single-level dict whose keys are ``sep``-joined paths through the
        original nesting.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def tabularize_params(
    cfgs: dict[Any, dict], read_keys: dict[str, list[str]], idx_name: str
) -> pd.DataFrame:
    """Read configurations from their partitions into a DataFrame.

    Args:
        cfgs: The dictionary of parameters for each scenario in a set, with the key
            representing the scenario identifier.
        read_keys: The sequences of parameters to read from each of the the multi-level
            dictionaries within cfgs.
        idx_name: The name to give the resulting DataFrame index.

    Returns: DataFrame of select parameters from each scenario, indexed by scenario identifier.
    """
    ext = {k: extract_params(v, read_keys) for k, v in cfgs.items()}
    col_wise = {
        col: {k: row[col] for k, row in ext.items()} for col in next(iter(ext.values()))
    }
    cfg_df = pd.DataFrame.from_dict(col_wise)
    cfg_df.index.name = idx_name
    cfg_df = cfg_df.sort_index()
    return cfg_df


def extract_params(params: dict, key_map: dict) -> dict:
    """Extract the parameters of interest from a config for this scenario.

    Args:
        params: the dict of parameters used directly by `kedro`
        key_map: the dict of (reporting key, kedro parameter key) pairs. The kedro
            parameter keys may be tuples of any length, which will be interpreted as the
            key at each level of the dictionary.

    Returns: A dictionary of only the selected parameters which is only one level deep.
    """

    def get_value_from_dict(d, keys):
        cum_keys = []
        for key in keys:
            cum_keys.append(key)
            if key in d:
                d = d[key]
            else:
                raise ValueError(f"Key {cum_keys} not found in parameters.")
        return d

    res = {k: get_value_from_dict(params, v) for k, v in key_map.items()}
    return res


def import_from_config(import_path: str):
    """Import a Python class (or any object) from a dotted import path string.

    Allows YAML configuration files to specify model components by their fully
    qualified Python path (e.g. ``"laurel.models.charging.LinearCharger"``),
    which is then imported at runtime.

    Args:
        import_path: Fully qualified dotted path to the object
            (e.g. ``"package.module.ClassName"``).

    Returns:
        The imported class or object.
    """
    module_path, obj_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
