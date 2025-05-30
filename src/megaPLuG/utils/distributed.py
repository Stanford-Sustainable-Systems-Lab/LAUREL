from dask.distributed import Client, LocalCluster


def start_dask_node(params: dict) -> tuple[LocalCluster, Client]:
    """Start a Dask LocalCluster and client."""
    cluster = LocalCluster(**params["cluster"])
    client = Client(cluster)
    return cluster, client


def stop_dask_node(cluster: LocalCluster, client: Client, result: object) -> None:
    """Stop a Dask LocalCluster and client.

    result is used to ensure that this node runs last, after all desired results have
    been computed. Pass the final dataset which requires Dask to this node.
    """
    cluster.close()
    client.close()
