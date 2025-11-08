"""This file should be copied into the nodes.py file of whatever `kedro` pipeline
you want to run it in. This is so that you can define scenario builders outside of
this package and then import them."""

import inspect

from .build import ScenarioBuilder  # noqa: F401


def generate_scenario_configs(scen_params: dict, all_params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        scen_params: just the item in the "parameters" dictionary dedicated to giving
            parameters for scenario building (e.g. builder name)
        all_params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions. Usually this would be
    saved to a `kedro` partitioned dataset.
    """
    bldr_map = {}
    base_cls = ScenarioBuilder
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls:
            bldr_map.update({name: obj})

    bldr_name = scen_params["builder"]
    try:
        builder_cls = bldr_map[bldr_name]
    except KeyError:
        raise NotImplementedError(f"Scenario builder {bldr_name} not imported.")
    builder = builder_cls(scen_params=scen_params, all_params=all_params)
    parts = builder.build_configs()
    return (parts, builder)
