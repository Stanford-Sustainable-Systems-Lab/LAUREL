"""
This is a boilerplate pipeline 'evaluate_impacts'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from megaPLuG.models.dwell_sets import load_dwell_set

from .nodes import (
    add_geometries,
    aggregate_regional_loads,
    calc_derived_dwell_cols,
    drop_vehicles,
    get_hex_events_from_dwells,
    plot_hourly_load,
    plot_peak_load_evolution,
    report_by_hex,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=load_dwell_set,
                inputs=["dwells_with_charging", "params:load_dwell_set"],
                outputs="dwell_obj_loaded",
                name="load_dwell_set_again",
            ),
            node(
                func=drop_vehicles,
                inputs=["dwell_obj_loaded", "params:drop_vehicles"],
                outputs="dwells_dropped",
                name="drop_vehicles",
            ),
            node(
                func=calc_derived_dwell_cols,
                inputs=["dwells_dropped", "params:derived_cols"],
                outputs="dwells_derived",
                name="calc_derived_dwell_cols",
            ),
            node(
                func=get_hex_events_from_dwells,
                inputs=["dwells_derived", "params:events_from_dwells"],
                outputs="events",
                name="get_hex_events_from_dwells",
            ),
            node(
                func=report_by_hex,
                inputs=["events", "params:report_by_hex"],
                outputs="report_by_hex",
                name="report_by_hex",
            ),
            node(
                func=add_geometries,
                inputs=["report_by_hex", "params:add_geometries"],
                outputs="report_by_hex_with_geoms",
                name="add_geometries",
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
