"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_h3_polygons, clean_vius, format_navistar_columns
from .process_role_consumption import calc_energy_consump_rate
from .process_role_energy import label_charging_sessions


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=format_navistar_columns,
                inputs=["navistar", "params:navistar"],
                outputs="trips",
                name="format_navistar_columns",
            ),
            node(
                func=build_h3_polygons,
                inputs="us_outline",
                outputs="h3_8_polygons",
                name="build_h3_polygons",
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
