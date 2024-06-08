"""
This is a boilerplate pipeline 'build_scenario_runners'
generated using Kedro 0.18.13
"""

from pathlib import Path

from .manage_scenarios import generate_configs


@generate_configs
def build_test_configs(params: dict) -> tuple[list[Path], list[dict]]:
    """Build test scenario."""
    scen_params = params["scenario_params"]
    scen_name = scen_params["name"]
    paths = [Path(scen_name)]
    scens = [{}]
    return (paths, scens)


def generate_scenario_configs(params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions.
    """
    scen_name = params["scenario_params"]["name"]
    if scen_name == "test":
        func = build_test_configs
    else:
        raise NotImplementedError(
            f"Scenario config generator for {scen_name} scenarios not yet implemented."
        )
    parts = func(params)
    return parts
