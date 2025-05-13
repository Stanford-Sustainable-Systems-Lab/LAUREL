import asyncio

import geopandas as gpd
import pandas as pd
import shapely as shp
from routingpy.exceptions import RouterApiError
from tqdm.asyncio import trange

from .parser import AsyncGraphhopper
from .server import GraphhopperContainerRouter

DIST_COL = "trip_meters_route"
TIME_COL = "trip_seconds_route"
ROUTE_COL = "trip_geom_route"


async def get_routes_async(
    trips: gpd.GeoDataFrame,
    orig_col: str,
    dest_col: str,
    server_params: dict,
    conc_mult: int = 5,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Set up the server and client for routing, then iterate through groups asynchronously."""
    trips[DIST_COL] = pd.Series(data=pd.NA, dtype=pd.Float64Dtype())
    trips[TIME_COL] = pd.Series(data=pd.NA, dtype=pd.Float64Dtype())
    trips[ROUTE_COL] = gpd.GeoSeries(data=None, crs=trips.crs)

    idx = {col_name: idx for idx, col_name in enumerate(trips.columns)}
    conc_max = server_params["max_concurrent_requests"]

    with GraphhopperContainerRouter(
        image=server_params["image"], graph_dir=server_params["graph_dir"]
    ) as server:
        async with AsyncGraphhopper(
            base_url=server.base_url,
            max_concurrent_requests=conc_max,
        ) as router:
            # After completion, you can analyze the semaphore usage
            sem = router.client.request_semaphore
            max_concurrent = max(sem.usage_history) if sem.usage_history else 0
            avg_concurrent = sum(sem.usage_history) / len(sem.usage_history) if sem.usage_history else 0
            
            print(f"Max concurrent tasks: {max_concurrent}")
            print(f"Avg concurrent tasks: {avg_concurrent:.2f}")
    return trips


async def _get_routes_async_core(
    dwells: gpd.GeoDataFrame, router: AsyncGraphhopper, **kwargs
) -> gpd.GeoDataFrame:
    """Get all routes for an individual vehicle asynchronously."""
    geos = dwells.geometry
    directs = [
        _report_route(pd.NA, pd.NA, None)
    ]  # Initial location has no prior location

    if len(geos) > 1:  # So we have at least two geos to compare
        tasks = []
        for i in range(1, len(geos)):
            orig = geos.iloc[i - 1]
            dest = geos.iloc[i]
            cur_task = asyncio.create_task(
                _get_route_async(orig=orig, dest=dest, router=router, **kwargs)
            )
            tasks.append(cur_task)

        results = await asyncio.gather(*tasks)
        directs.extend(results)

    routes = pd.DataFrame.from_records(directs, index=dwells.index)
    return pd.concat([dwells, routes], axis=1)


async def _get_route_async(
    orig: shp.Point, dest: shp.Point, router: AsyncGraphhopper, **kwargs
) -> dict:
    """Get a single route asynchronously"""
    if orig == dest:
        return _report_route(0.0, 0.0, None)
    if orig is None or dest is None:
        return _report_route(pd.NA, pd.NA, None)

    try:
        coords = (tuple(orig.coords)[0], tuple(dest.coords)[0])
        rte = await router.directions(locations=coords, **kwargs)
        return _report_route(rte.distance, rte.duration, shp.LineString(rte.geometry))
    except RouterApiError:
        return _report_route(pd.NA, pd.NA, None)


def _report_route(meters: float, seconds: float, geom: shp.Geometry) -> dict:
    res = {
        DIST_COL: meters,
        TIME_COL: seconds,
        ROUTE_COL: geom,
    }
    return res
