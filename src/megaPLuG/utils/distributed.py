import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from megaPLuG.models.dwell_sets import DwellSet


def start_dask_node(params: dict) -> tuple[LocalCluster, Client]:
    """Start a Dask LocalCluster and client."""
    if ("use_dask" in params and params["use_dask"]) or ("use_dask" not in params):
        cluster = LocalCluster(**params["cluster"])
        client = Client(cluster)
        return cluster, client
    else:
        return "None", "None"


def stop_dask_node(cluster: LocalCluster, client: Client, result: object) -> None:
    """Stop a Dask LocalCluster and client.

    result is used to ensure that this node runs last, after all desired results have
    been computed. Pass the final dataset which requires Dask to this node.
    """
    if client != "None" and isinstance(client, Client):
        client.close()

    if cluster != "None" and isinstance(cluster, LocalCluster):
        cluster.close()


def load_in_memory_node(ddf: dd.DataFrame | DwellSet) -> pd.DataFrame | DwellSet:
    """Force computation to bring the input dataframe into memory."""
    if isinstance(ddf, DwellSet):
        ddf_new = ddf.copy_without_data()
        ddf_new.data = ddf.data.compute()
        return ddf_new
    elif isinstance(ddf, dd.DataFrame):
        return ddf.compute()
    else:
        raise NotImplementedError(
            "Load-in-memory is not yet implemented for this type."
        )
