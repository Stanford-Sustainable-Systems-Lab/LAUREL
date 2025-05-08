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

import asyncio
import copy
import json
import random
import warnings
from datetime import datetime

import aiohttp
from routingpy import exceptions
from routingpy.client_base import _RETRIABLE_STATUSES, DEFAULT, BaseClient, options
from routingpy.utils import get_ordinal


class AsyncClient(BaseClient):
    """Asynchronous client class for requests handling, which is passed to each router. Uses the aiohttp package."""

    def __init__(
        self,
        base_url,
        user_agent=None,
        timeout=DEFAULT,
        retry_timeout=None,
        retry_over_query_limit=None,
        skip_api_error=None,
        max_concurrent_requests=10,
        **kwargs,
    ):
        """
        :param base_url: The base URL for the request. All routers must provide a default.
            Should not have a trailing slash.
        :type base_url: string

        :param user_agent: User-Agent to send with the requests to routing API.
            Overrides ``options.default_user_agent``.
        :type user_agent: string

        :param timeout: Combined connect and read timeout for HTTP requests, in
            seconds. Specify "None" for no timeout.
        :type timeout: int

        :param retry_timeout: Timeout across multiple retriable requests, in
            seconds.
        :type retry_timeout: int

        :param retry_over_query_limit: If True, client will not raise an exception
            on HTTP 429, but instead jitter a sleeping timer to pause between
            requests until HTTP 200 or retry_timeout is reached.
        :type retry_over_query_limit: bool

        :param skip_api_error: Continue with batch processing if a :class:`routingpy.exceptions.RouterApiError` is
            encountered (e.g. no route found). If False, processing will discontinue and raise an error. Default False.
        :type skip_api_error: bool

        :param kwargs: Additional arguments, such as headers or proxies.
        :type kwargs: dict
        """
        super().__init__(
            base_url,
            user_agent=user_agent,
            timeout=timeout,
            retry_timeout=retry_timeout,
            retry_over_query_limit=retry_over_query_limit,
            skip_api_error=skip_api_error,
            **kwargs,
        )
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.kwargs = kwargs or {}
        try:
            self.headers.update(self.kwargs["headers"])
        except KeyError:
            pass

        self.kwargs["headers"] = self.headers
        self.kwargs["timeout"] = self.timeout

        self.proxies = self.kwargs.get("proxies") or options.default_proxies
        if self.proxies:
            self.kwargs["proxies"] = self.proxies

    async def _request(
        self,
        url,
        get_params={},
        post_params=None,
        first_request_time=None,
        retry_counter=0,
        dry_run=None,
    ):
        """Performs HTTP GET/POST with credentials, returning the body as
        JSON.

        :param url: URL path for the request. Should begin with a slash.
        :type url: string

        :param get_params: HTTP GET parameters.
        :type get_params: dict or list of tuples

        :param post_params: HTTP POST parameters. Only specified by calling method.
        :type post_params: dict

        :param first_request_time: The time of the first request (None if no
            retries have occurred).
        :type first_request_time: :class:`datetime.datetime`

        :param retry_counter: The number of this retry, or zero for first attempt.
        :type retry_counter: int

        :param dry_run: If true, only prints URL and parameters. true or false.
        :type dry_run: bool

        :raises routingpy.exceptions.RouterApiError: when the API returns an error due to faulty configuration.
        :raises routingpy.exceptions.RouterServerError: when the API returns a server error.
        :raises routingpy.exceptions.RouterError: when anything else happened while requesting.
        :raises routingpy.exceptions.JSONParseError: when the JSON response can't be parsed.
        :raises routingpy.exceptions.Timeout: when the request timed out.
        :raises routingpy.exceptions.TransportError: when something went wrong while trying to
            execute a request.

        :returns: raw JSON response or GeoTIFF image
        :rtype: dict or bytes
        """

        if not first_request_time:
            first_request_time = datetime.now()

        elapsed = datetime.now() - first_request_time
        if elapsed > self.retry_timeout:
            raise exceptions.Timeout()

        if retry_counter > 0:
            # 0.5 * (1.5 ^ i) is an increased sleep time of 1.5x per iteration,
            # starting at 0.5s when retry_counter=1. The first retry will occur
            # at 1, so subtract that first.
            delay_seconds = 1.5 ** (retry_counter - 1)

            # Jitter this value by 50% and pause.
            asyncio.sleep(delay_seconds * (random.random() + 0.5))

        authed_url = self._generate_auth_url(url, get_params)

        final_requests_kwargs = copy.copy(self.kwargs)

        # Determine GET/POST.
        requests_method = self._session.get
        if post_params is not None:
            requests_method = self._session.post
            if final_requests_kwargs["headers"]["Content-Type"] == "application/json":
                final_requests_kwargs["json"] = post_params
            else:
                # Send as x-www-form-urlencoded key-value pair string (e.g. Mapbox API)
                final_requests_kwargs["data"] = post_params

        # Only print URL and parameters for dry_run
        if dry_run:
            print(
                f"url:\n{self.base_url + authed_url}\nParameters:\n{json.dumps(final_requests_kwargs, indent=2)}"
            )
            return

        try:
            response = await requests_method(
                self.base_url + authed_url, **final_requests_kwargs
            )
            self._req = response.request_info

        except TimeoutError:
            raise exceptions.Timeout()

        tried = retry_counter + 1

        if response.status in _RETRIABLE_STATUSES:
            # Retry request.
            warnings.warn(
                f"Server down.\nRetrying for the {tried}{get_ordinal(tried)} time.",
                UserWarning,
            )
            return await self._request(
                url, get_params, post_params, first_request_time, retry_counter + 1
            )

        try:
            return await self._get_body(response)

        except exceptions.RouterApiError:
            if self.skip_api_error:
                txt = await response.text()
                warnings.warn(
                    f"Router {self.__class__.__name__} returned an API error with "
                    f"the following message:\n{txt}"
                )
                return

            raise

        except exceptions.RetriableRequest as e:
            if (
                isinstance(e, exceptions.OverQueryLimit)
                and not self.retry_over_query_limit
            ):
                raise

            warnings.warn(
                f"Rate limit exceeded.\nRetrying for the {tried}{get_ordinal(tried)} time.",
                UserWarning,
            )
            # Retry request.
            return await self._request(
                url, get_params, post_params, first_request_time, retry_counter + 1
            )

    @property
    def req(self):
        """Holds the :class:`requests.PreparedRequest` property for the last request."""
        return self._req

    @staticmethod
    async def _get_body(response: aiohttp.ClientResponse):
        status_code = response.status
        content_type = response.headers["content-type"]

        if status_code == 200:
            if content_type == "image/tiff":
                return response.content

            else:
                try:
                    js = await response.json()
                    return js

                except json.decoder.JSONDecodeError:
                    txt = await response.text()
                    raise exceptions.JSONParseError(f"Can't decode JSON response:{txt}")

        if status_code == 429:
            txt = await response.text()
            raise exceptions.OverQueryLimit(status_code, txt)

        if 400 <= status_code < 500:
            txt = await response.text()
            raise exceptions.RouterApiError(status_code, txt)

        if 500 <= status_code:
            txt = await response.text()
            raise exceptions.RouterServerError(status_code, txt)

        if status_code != 200:
            txt = await response.text()
            raise exceptions.RouterError(status_code, txt)
