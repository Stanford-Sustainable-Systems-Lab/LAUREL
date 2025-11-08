import asyncio
import logging

import geopandas as gpd
import numpy as np
import shapely as shp
from routingpy.exceptions import RouterApiError

from .parser import AsyncGraphhopper

logger = logging.getLogger(__name__)

DIST_COL = "trip_meters_route"
TIME_COL = "trip_seconds_route"
ROUTE_COL = "trip_geom_route"


def get_routes(gdf: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
    """Get routes for an individual GeoPandas dataframe."""
    res = asyncio.run(_get_routes_async(trips=gdf, **kwargs))
    logger.info("Completed a routing partition.")
    return res


async def _get_routes_async(
    trips: gpd.GeoDataFrame,
    orig_col: str,
    dest_col: str,
    server_url: str,
    max_concurrent_requests: int = 200,
    batch_size: int = 5000,
    timeout: int = 10,
    verbose: bool = False,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Set up the server and client for routing, then iterate through groups asynchronously."""
    trips[DIST_COL] = np.nan
    trips[TIME_COL] = np.nan
    trips[ROUTE_COL] = gpd.GeoSeries(data=None, crs=trips.crs)

    idx = {col_name: idx for idx, col_name in enumerate(trips.columns)}

    # This wrapper ensures the semaphore is properly used
    async def process_route(i):
        res_dict = await _get_route_async(
            orig=trips.iat[i, idx[orig_col]],
            dest=trips.iat[i, idx[dest_col]],
            router=router,
            **kwargs,
        )
        trips.iat[i, idx[DIST_COL]] = res_dict[DIST_COL]
        trips.iat[i, idx[TIME_COL]] = res_dict[TIME_COL]
        trips.iat[i, idx[ROUTE_COL]] = res_dict[ROUTE_COL]

    async with AsyncGraphhopper(
        base_url=server_url,
        timeout=timeout,
        max_concurrent_requests=max_concurrent_requests,
    ) as router:
        # Process in batches
        total_trips = len(trips)

        for batch_start in range(0, total_trips, batch_size):
            batch_end = min(batch_start + batch_size, total_trips)

            # Create tasks for this batch and process them together
            batch_tasks = [process_route(i) for i in range(batch_start, batch_end)]
            await asyncio.gather(*batch_tasks)

        if verbose:
            # After completion, you can analyze the semaphore usage
            sem = router.client.request_semaphore
            max_concurrent = max(sem.usage_history) if sem.usage_history else 0
            avg_concurrent = (
                sum(sem.usage_history) / len(sem.usage_history)
                if sem.usage_history
                else 0
            )

            logger.debug(f"Max concurrent tasks: {max_concurrent}")
            logger.debug(f"Avg concurrent tasks: {avg_concurrent:.2f}")
    return trips


async def _get_route_async(
    orig: shp.Point, dest: shp.Point, router: AsyncGraphhopper, **kwargs
) -> dict:
    """Get a single route asynchronously"""
    if orig == dest:
        return _report_route(0.0, 0.0, None)
    if orig is None or dest is None:
        return _report_route(np.nan, np.nan, None)

    try:
        coords = (tuple(orig.coords)[0], tuple(dest.coords)[0])
        rte = await router.directions(locations=coords, **kwargs)
        linestring = shp.LineString(rte.geometry)
        return _report_route(rte.distance, rte.duration, linestring)
    except RouterApiError:
        return _report_route(np.nan, np.nan, None)
    except shp.lib.GEOSException:
        logger.warning(f"Interpretation of route caused GEOSException: {rte.geometry}")
        return _report_route(np.nan, np.nan, None)


def _report_route(meters: float, seconds: float, geom: shp.Geometry) -> dict:
    res = {
        DIST_COL: meters,
        TIME_COL: seconds,
        ROUTE_COL: geom,
    }
    return res
