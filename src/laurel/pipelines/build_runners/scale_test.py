import re
from copy import deepcopy
from pathlib import Path
from typing import Self

from laurel.scenarios.build import ScenarioBuilder
from laurel.scenarios.read import ScenarioReader


class ScaleTestScenarioBuilder(ScenarioBuilder):
    """Build scenarios for the scaling test truck model."""

    partition_level_names = (
        "run_name",
        "sample_source",
        "task_id",
    )

    def _build_param_dicts(self) -> tuple[list[Path], list[dict]]:
        paths, scens = [], []
        sources = self.scen_params["sample_source"]

        for sample_source in sources:
            pth = Path(
                self.display_name,
                f"sample_source_{sample_source}",
            )

            cur_samp = deepcopy(self.params["sample_profiles"])
            if sample_source == "self":
                cur_samp["sample_self"] = True
                cur_samp["sample_class"] = False
            elif sample_source == "class":
                cur_samp["sample_self"] = False
                cur_samp["sample_class"] = True

            scn = {
                "sample_profiles": cur_samp,
            }

            paths.append(pth)
            scens.append(scn)
        return (paths, scens)


class ScaleTestScenarioReader(ScenarioReader):
    """Read scenarios for the scaling test truck model."""

    builder = ScaleTestScenarioBuilder
    metadata_level_names = ("sample_source",)

    def extract_metadata(self: Self, path: Path) -> tuple:
        meta = self.get_metadata_values(path=path)
        task_id = re.search(r"(?<=sample_source_)(.+)", meta["sample_source"]).group()
        return (task_id,)

    def name_scenario(self: Self, path: Path) -> str:
        meta = self.get_metadata_values(path=path)
        task_id = re.search(r"(?<=sample_source_)(.+)", meta["sample_source"]).group()
        return task_id
