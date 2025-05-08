# Copyright (C) 2021 GIS OPS UG
#
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

from typing import Self  # noqa: F401

from aiohttp import ClientSession
from routingpy import utils
from routingpy.client_base import DEFAULT
from routingpy.direction import Direction, Directions

from .client import AsyncClient


class AsyncGraphhopper:
    """Performs requests to the Graphhopper API services."""

    _DEFAULT_BASE_URL = "https://graphhopper.com/api/1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = _DEFAULT_BASE_URL,
        user_agent: str | None = None,
        timeout: int | None = DEFAULT,
        retry_timeout: int | None = None,
        retry_over_query_limit: bool | None = False,
        skip_api_error: bool | None = None,
        client=AsyncClient,
        **client_kwargs,
    ):
        """
        Initializes an graphhopper client.

        :param api_key: GH API key. Required if https://graphhopper.com/api is used.
        :type api_key: str

        :param base_url: The base URL for the request. Defaults to the ORS API
            server. Should not have a trailing slash.
        :type base_url: str

        :param user_agent: User Agent to be used when requesting.
            Default :attr:`routingpy.routers.options.default_user_agent`.
        :type user_agent: str

        :param timeout: Combined connect and read timeout for HTTP requests, in
            seconds. Specify ``None`` for no timeout. Default :attr:`routingpy.routers.options.default_timeout`.
        :type timeout: int or None

        :param retry_timeout: Timeout across multiple retriable requests, in
            seconds.  Default :attr:`routingpy.routers.options.default_retry_timeout`.
        :type retry_timeout: int

        :param retry_over_query_limit: If True, client will not raise an exception
            on HTTP 429, but instead jitter a sleeping timer to pause between
            requests until HTTP 200 or retry_timeout is reached.
            Default :attr:`routingpy.routers.options.default_retry_over_query_limit`.
        :type retry_over_query_limit: bool

        :param skip_api_error: Continue with batch processing if a :class:`routingpy.exceptions.RouterApiError` is
            encountered (e.g. no route found). If False, processing will discontinue and raise an error.
            Default :attr:`routingpy.routers.options.default_skip_api_error`.
        :type skip_api_error: bool

        :param client: A client class for request handling. Needs to be derived from :class:`routingpy.client_base.BaseClient`
        :type client: abc.ABCMeta

        :param client_kwargs: Additional arguments passed to the client, such as headers or proxies.
        :type client_kwargs: dict

        """

        if base_url == self._DEFAULT_BASE_URL and api_key is None:
            raise KeyError("API key must be specified.")
        self.key = api_key

        self.client = client(
            base_url,
            user_agent,
            timeout,
            retry_timeout,
            retry_over_query_limit,
            skip_api_error,
            **client_kwargs,
        )

    async def __aenter__(self: Self) -> Self:
        """Start the routing client-side service."""
        self.client._session = ClientSession()
        return self

    async def __aexit__(self: Self, exc_type, exc_val, exc_tb):
        """Stop the routing client-side service."""
        await self.client._session.close()

    async def directions(  # noqa: C901
        self,
        locations: list[list[float]] | tuple[tuple[float]],
        profile: str,
        format: str | None = None,
        optimize: bool | None = None,
        instructions: bool | None = None,
        locale: str | None = None,
        elevation: bool | None = None,
        points_encoded: bool | None = True,
        calc_points: bool | None = None,
        debug: bool | None = None,
        point_hints: list[str] | None = None,
        details: list[str] | None = None,
        ch_disable: bool | None = None,
        custom_model: dict | None = None,
        headings: list[int] | None = None,
        heading_penalty: int | None = None,
        pass_through: bool | None = None,
        algorithm: str | None = None,
        round_trip_distance: int | None = None,
        round_trip_seed: int | None = None,
        alternative_route_max_paths: int | None = None,
        alternative_route_max_weight_factor: float | None = None,
        alternative_route_max_share_factor: float | None = None,
        dry_run: bool | None = None,
        snap_preventions: list[str] | None = None,
        curbsides: list[str] | None = None,
        **direction_kwargs,
    ):
        """Get directions between an origin point and a destination point.

        Use ``direction_kwargs`` for any missing ``directions`` request options.

        For more information, visit https://docs.graphhopper.com/#operation/postRoute.

        :param locations: The coordinates tuple the route should be calculated
            from in order of visit.
        :type locations: list of list or tuple of tuple

        :param profile: The vehicle for which the route should be calculated. One of ["car" "bike" "foot" "hike" "mtb"
            "racingbike" "scooter" "truck" "small_truck"]. Default "car".
        :type profile: str

        :param format: Specifies the resulting format of the route, for json the content type will be application/json.
            Default "json".
        :type format: str

        :param locale: Language for routing instructions. The locale of the resulting turn instructions.
            E.g. pt_PT for Portuguese or de for German. Default "en".
        :type locale: str

        :param optimize: If false the order of the locations will be identical to the order of the point parameters.
            If you have more than 2 points you can set this optimize parameter to ``True`` and the points will be sorted
            regarding the minimum overall time - e.g. suiteable for sightseeing tours or salesman.
            Keep in mind that the location limit of the Route Optimization API applies and the credit costs are higher!
            Note to all customers with a self-hosted license: this parameter is only available if your package includes
            the Route Optimization API. Default False.
        :type optimize: bool

        :param instructions: Specifies whether to return turn-by-turn instructions.
            Default True.
        :type instructions: bool

        :param elevation: If true a third dimension - the elevation - is included in the polyline or in the GeoJson.
            IMPORTANT: If enabled you have to use a modified version of the decoding method or set points_encoded to false.
            See the points_encoded attribute for more details. Additionally a request can fail if the vehicle does not
            support elevation. See the features object for every vehicle.
            Default False.
        :type elevation: bool

        :param points_encoded: If ``False`` the coordinates in point and snapped_waypoints are returned as array using the order
            [lon,lat,elevation] for every point. If true the coordinates will be encoded as string leading to less bandwith usage.
            Default True.
        :type points_encoded: bool

        :param calc_points: If the points for the route should be calculated at all, printing out only distance and time.
            Default True.
        :type calc_points: bool

        :param debug: If ``True``, the output will be formated.
            Default False.
        :type debug: bool

        :param point_hints: The point_hints is typically a road name to which the associated point parameter should be
            snapped to. Specify no point_hint parameter or the same number as you have locations. Optional.
        :type point_hints: list of str

        :param details: Optional parameter to retrieve path details. You can request additional details for the route:
            street_name, time, distance, max_speed, toll, road_class, road_class_link, road_access, road_environment,
            lanes, and surface.
        :type details: list of str

        :param ch_disable: Always use ch_disable=true in combination with one or more parameters of this table.
            Default False.
        :type ch_disable: bool

        :param custom_model: The custom_model modifies the routing behaviour of the specified profile.
            See https://docs.graphhopper.com/#section/Custom-Model
        :type custom_model: dict

        :param headings: Optional parameter. Favour a heading direction for a certain point. Specify either one heading for the start point or as
            many as there are points. In this case headings are associated by their order to the specific points.
            Headings are given as north based clockwise angle between 0 and 360 degree.
        :type headings: list of int

        :param heading_penalty: Optional parameter. Penalty for omitting a specified heading. The penalty corresponds to the accepted time
            delay in seconds in comparison to the route without a heading.
            Default 120.
        :type heading_penalty: int

        :param pass_through: Optional parameter. If true u-turns are avoided at via-points with regard to the heading_penalty.
            Default False.
        :type pass_through: bool

        :param algorithm: Optional parameter. round_trip or alternative_route.
        :type algorithm: str

        :param round_trip_distance: If algorithm=round_trip this parameter configures approximative length of the resulting round trip.
            Default 10000.
        :type round_trip_distance: int

        :param round_trip_seed: If algorithm=round_trip this parameter introduces randomness if e.g. the first try wasn't good.
            Default 0.
        :type round_trip_seed: int

        :param alternative_route_max_paths: If algorithm=alternative_route this parameter sets the number of maximum paths
            which should be calculated. Increasing can lead to worse alternatives.
            Default 2.
        :type alternative_route_max_paths: int

        :param alternative_route_max_weight_factor: If algorithm=alternative_route this parameter sets the factor by which the alternatives
            routes can be longer than the optimal route. Increasing can lead to worse alternatives.
            Default 1.4.
        :type alternative_route_max_weight_factor: float

        :param alternative_route_max_share_factor: If algorithm=alternative_route this parameter specifies how much alternatives
            routes can have maximum in common with the optimal route. Increasing can lead to worse alternatives.
            Default 0.6.
        :type alternative_route_max_share_factor: float

        :param dry_run: Print URL and parameters without sending the request.
        :type dry_run: bool

        :param snap_preventions: Optional parameter to avoid snapping to a certain road class or road environment.
            Currently supported values are motorway, trunk, ferry, tunnel, bridge and ford. Optional.
        :type snap_preventions: list of str

        :param curbsides: One of "any", "right", "left". It specifies on which side a point should be relative to the driver
            when she leaves/arrives at a start/target/via point. You need to specify this parameter for either none
            or all points. Only supported for motor vehicles and OpenStreetMap.
        :type curbsides: list of str

        :returns: One or multiple route(s) from provided coordinates and restrictions.
        :rtype: :class:`routingpy.direction.Direction` or :class:`routingpy.direction.Directions`

        .. versionchanged:: 0.3.0
           `point_hint` used to be bool, which was not the right usage.

        .. versionadded:: 0.3.0
           ``snap_prevention``, ``curb_side``, ``turn_costs`` parameters

        .. versionchanged:: 1.2.0
           Renamed `point_hint` to `point_hints`, `heading` to `headings`,
           `snap_prevention` to `snap_preventions`, `curb_side` to `curbsides`,

        .. versionadded:: 1.2.0
           Added `custom_model` parameter

        .. deprecated:: 1.2.0
           Removed `weighting`, `block_area`, `avoid`, `turn_costs` parameters
        """

        params = {"profile": profile}

        if locations is not None:
            params["points"] = locations

        get_params = {}

        if self.key is not None:
            get_params["key"] = self.key

        if format is not None:
            params["type"] = format

        if optimize is not None:
            params["optimize"] = optimize

        if instructions is not None:
            params["instructions"] = instructions

        if locale is not None:
            params["locale"] = locale

        if elevation is not None:
            params["elevation"] = elevation

        if points_encoded is not None:
            params["points_encoded"] = points_encoded

        if calc_points is not None:
            params["calc_points"] = calc_points

        if debug is not None:
            params["debug"] = debug

        if point_hints is not None:
            params["point_hints"] = point_hints

        if snap_preventions:
            params["snap_preventions"] = snap_preventions

        if curbsides:
            params["curbsides"] = curbsides

        ### all below params will only work if ch is disabled

        if details is not None:
            params["details"] = details

        if ch_disable is not None:
            params["ch.disable"] = ch_disable

        if custom_model is not None:
            params["custom_model"] = custom_model

        if headings is not None:
            params["headings"] = headings

        if heading_penalty is not None:
            params["heading_penalty"] = heading_penalty

        if pass_through is not None:
            params["pass_through"] = pass_through

        if algorithm is not None:
            params["algorithm"] = algorithm

            if algorithm == "round_trip":
                if round_trip_distance is not None:
                    params["round_trip.distance"] = round_trip_distance

                if round_trip_seed is not None:
                    params["round_trip.seed"] = round_trip_seed

            if algorithm == "alternative_route":
                if alternative_route_max_paths is not None:
                    params["alternative_route.max_paths"] = alternative_route_max_paths

                if alternative_route_max_weight_factor is not None:
                    params["alternative_route.max_weight_factor"] = (
                        alternative_route_max_weight_factor
                    )

                if alternative_route_max_share_factor:
                    params["alternative_route_max_share_factor"] = (
                        alternative_route_max_share_factor
                    )

        params.update(direction_kwargs)

        async with self.client.request_semaphore:
            response = await self.client._request(
                "/route", get_params=get_params, post_params=params, dry_run=dry_run
            )

        return self.parse_directions_json(
            response,
            algorithm,
            elevation,
            points_encoded,
        )

    @staticmethod
    def parse_directions_json(response, algorithm, elevation, points_encoded):
        if response is None:  # pragma: no cover
            if algorithm == "alternative_route":
                return Directions()
            else:
                return Direction()

        if algorithm == "alternative_route":
            routes = []
            for route in response["paths"]:
                geometry = (
                    utils.decode_polyline5(route["points"], elevation)
                    if points_encoded
                    else route["points"]["coordinates"]
                )
                routes.append(
                    Direction(
                        geometry=geometry,
                        duration=int(route["time"] / 1000),
                        distance=int(route["distance"]),
                        raw=route,
                    )
                )
            return Directions(routes, response)
        else:
            geometry = (
                utils.decode_polyline5(response["paths"][0]["points"], elevation)
                if points_encoded
                else response["paths"][0]["points"]["coordinates"]
            )
            return Direction(
                geometry=geometry,
                duration=int(response["paths"][0]["time"] / 1000),
                distance=int(response["paths"][0]["distance"]),
                raw=response,
            )
