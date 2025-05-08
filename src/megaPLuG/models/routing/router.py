import asyncio

import geopandas as gpd
import pandas as pd
import shapely as shp
from routingpy.exceptions import RouterApiError
from tqdm.asyncio import tqdm_asyncio

from .parser import AsyncGraphhopper
from .server import GraphhopperContainerRouter


async def get_routes_for_trips_wrapper(
    dwells: gpd.GeoDataFrame, server_params: dict
) -> gpd.GeoDataFrame:
    with GraphhopperContainerRouter(
        image=server_params["image"], graph_dir=server_params["graph_dir"]
    ) as server:
        async with AsyncGraphhopper(
            base_url=server.base_url,
            max_concurrent_requests=server_params["max_concurrent_requests"],
        ) as router:
            tasks = []
            for name, grp in dwells.groupby("veh_id"):
                cur_task = asyncio.create_task(
                    get_routes_async(grp, router, profile="car")
                )
                tasks.append(cur_task)
            results = await tqdm_asyncio.gather(*tasks)
            with_routes = pd.concat(results, axis=0)
            with_routes["route_geometry"] = gpd.GeoSeries(with_routes["route_geometry"], crs=dwells.crs)
    return with_routes


async def get_routes_async(
    dwells: gpd.GeoDataFrame, router: AsyncGraphhopper, **kwargs
):
    """Get all routes for an individual vehicle asynchronously."""
    geos = dwells.geometry
    directs = [
        report_route(pd.NA, pd.NA, None)
    ]  # Initial location has no prior location

    if len(geos) > 1:  # So we have at least two geos to compare
        tasks = []
        for i in range(1, len(geos)):
            orig = geos.iloc[i - 1]
            dest = geos.iloc[i]
            cur_task = asyncio.create_task(
                get_route_async(orig=orig, dest=dest, router=router, **kwargs)
            )
            tasks.append(cur_task)

        results = await asyncio.gather(*tasks)
        directs.extend(results)

    routes = pd.DataFrame.from_records(directs, index=dwells.index)
    return pd.concat([dwells, routes], axis=1)


async def get_route_async(
    orig: shp.Point, dest: shp.Point, router: AsyncGraphhopper, **kwargs
):
    """Get a single route asynchronously"""
    if orig == dest:
        return report_route(0.0, 0.0, None)

    try:
        coords = (tuple(orig.coords)[0], tuple(dest.coords)[0])
        rte = await router.directions(locations=coords, **kwargs)
        return report_route(rte.distance, rte.duration, shp.LineString(rte.geometry))
    except RouterApiError:
        return report_route(pd.NA, pd.NA, None)


def report_route(meters: float, seconds: float, geom: shp.Geometry) -> dict:
    res = {
        "trip_meters_route": meters,
        "trip_seconds_route": seconds,
        "route_geometry": geom,
    }
    return res
