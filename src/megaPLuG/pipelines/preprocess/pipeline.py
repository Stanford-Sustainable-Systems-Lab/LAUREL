"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calc_derived_trip_cols,
    clean_vius,
    create_dwells,
    format_trips_columns,
    strip_vehicle_attrs,
)
from .process_role_consumption import calc_energy_consump_rate
from .process_role_energy import label_charging_sessions


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=format_trips_columns,
                inputs=["navistar", "params:format_columns"],
                outputs="trips_formatted",
                name="format_trips_columns",
            ),
            node(
                func=strip_vehicle_attrs,
                inputs=["trips_formatted", "params:strip_vehicle_attrs"],
                outputs=["trips_stripped", "vehicles_raw"],
                name="strip_vehicle_attrs",
            ),
            node(
                func=calc_derived_trip_cols,
                inputs=["trips_stripped", "params:trip_derived_cols"],
                outputs="trips_derived",
                name="calc_derived_trip_cols",
            ),
            node(
                func=create_dwells,
                inputs=["trips_derived", "params:create_dwells"],
                outputs="dwells",
                name="create_dwells",
            ),
            node(
                func=clean_vius,
                inputs="vius_raw",
                outputs="vius_clean",
                name="clean_vius",
            ),
            node(
                func=label_charging_sessions,
                inputs=["role_energy_readings", "params:charge_sessions"],
                outputs="energy_sessions",
                name="preprocess_energy_sessions",
            ),
            node(
                func=calc_energy_consump_rate,
                inputs=["energy_sessions", "role_gps", "params:consump_rate"],
                outputs="energy_consumption",
                name="preprocess_energy_consumption",
            ),
        ],
    )
    return pipe
