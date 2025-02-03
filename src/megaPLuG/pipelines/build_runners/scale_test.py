import re
from copy import deepcopy
from pathlib import Path
from typing import Self

from megaPLuG.scenarios.build import ScenarioBuilder
from megaPLuG.scenarios.read import ScenarioReader


class ScaleTestScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the scaling test truck model."""

    display_name = "scale_test"
    partition_level_names = (
        "run_name",
        "is_scaled",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        scale_levels = self.scen_params["is_scaled"]

        for is_scaled in scale_levels:
            pth = Path(
                self.display_name,
                f"is_scaled_{is_scaled}",
            )

            cur_scale = deepcopy(self.params["assign_scale_up_factor"])
            cur_scale["apply_scaling"] = is_scaled

            scn = {
                "assign_scale_up_factor": cur_scale,
            }

            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class ScaleTestScenarioReader(ScenarioReader):
    """Read scenarios for the scaling test truck model."""

    builder = ScaleTestScenarioBuilder
    metadata_level_names = ("is_scaled",)

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        task_id = re.search(r"(?<=is_scaled_)(.+)", meta["is_scaled"]).group()
        return (task_id,)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        task_id = re.search(r"(?<=is_scaled_)(.+)", meta["is_scaled"]).group()
        return task_id
