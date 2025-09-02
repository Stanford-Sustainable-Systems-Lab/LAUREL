from dask.distributed import Client, LocalCluster


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
