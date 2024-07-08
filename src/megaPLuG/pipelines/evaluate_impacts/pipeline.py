"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    aggregate_regional_loads,
    calc_derived_trip_cols,
    get_events_from_trips,
    get_load_profiles,
    plot_hourly_load,
    plot_peak_load_evolution,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=calc_derived_trip_cols,
                inputs=["trips_with_charging", "params:derived_cols"],
                outputs="trips_derived",
                name="calc_derived_trip_cols",
            ),
            node(
                func=get_events_from_trips,
                inputs=["trips_derived", "params:events_from_trips"],
                outputs="events",
                name="get_events_from_trips",
            ),
            node(
                func=get_load_profiles,
                inputs=["events", "params:load_profiles"],
                outputs="load_profiles",
                name="get_load_profiles",
            ),
            node(
                func=aggregate_regional_loads,
                inputs=["charging_sessions_sampled", "grid_regions"],
                outputs="regional_loads",
                name="aggregate_regional_loads",
            ),
            node(
                func=plot_peak_load_evolution,
                inputs=["regional_loads", "baseline_load"],
                outputs="peak_load_evolution",
                name="plot_peak_load_evolution",
            ),
            node(
                func=plot_hourly_load,
                inputs=["regional_loads", "baseline_load"],
                outputs="hourly_load",
                name="plot_hourly_load",
            ),
        ],
    )
    return pipe
