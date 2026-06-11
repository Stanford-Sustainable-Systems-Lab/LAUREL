# Applied Energy 2026 — figure & dataset reproduction notebooks

These notebooks reproduce the figures, the one computed table, and the in-text
summary numbers for the manuscript. They are organized by manuscript section. Maps are assembled in QGIS from
GeoPackages the notebooks export; the other figures are generated inline.

**To find where a figure is made:** locate it in the index below, open the listed
notebook, and search for its bold marker (e.g. `Figure 8a`) — each figure/table cell is
a markdown heading like `### Figure N: …` immediately above the code that produces it.
Every notebook's top cell also has a **Generates:** line summarizing its outputs.

## Figure & table index

| Manuscript output | Notebook | How it's produced |
|---|---|---|
| Figure 1 — LAUREL model-flow schematic | — | conceptual diagram, not generated in a notebook |
| Figure 2a — data structure schematic | — | conceptual diagram |
| Figure 2b — example-vehicle stop/route map | [2.2](2.2_augment_dwells_data.ipynb) | exports a trajectory GeoPackage → QGIS |
| Figure 2c — operating-distance distribution vs VIUS | [2.2](2.2_augment_dwells_data.ipynb) | inline bar chart |
| Figure 3a — SF Bay Area TAZ map | [2.3](2.3_augment_taz_data.ipynb) | exports a TAZ GeoPackage → QGIS |
| Figure 3b — freight-class selection schematic | — | conceptual diagram |
| Figure 4a — robustness/upgrade-policy schematic | — | conceptual diagram |
| Figure 4b — robust peak-load ECDF | [3.1.1](3.1.1_upgrades_across_us.ipynb) | inline ECDF |
| Figure 5 — continental peak-load choropleth | [3.1.1](3.1.1_upgrades_across_us.ipynb) | exports per-substation GeoPackages → QGIS |
| Figure 6a — PG&E substation overload-exposure map | [3.1.2](3.1.2_upgrades_known_conditions.ipynb) | exports a GeoPackage → QGIS |
| Figure 6b — Laytonville stacked load-profile envelopes | [3.1.2](3.1.2_upgrades_known_conditions.ipynb) | inline functional boxplot |
| Figure 7 — SF Bay Area peak-load inset | [3.1.1](3.1.1_upgrades_across_us.ipynb) | exports per-substation GeoPackages → QGIS |
| Figure 8a/8b — PRIM factor importance | [3.3](3.3_upgrade_indicators_technoeconomic.ipynb) | inline bar charts |
| Figure 8c/8d — minimum-adoption-trigger ECDFs | [3.3](3.3_upgrade_indicators_technoeconomic.ipynb) | inline ECDFs |
| Figure 9 — minimum-adoption-trigger map | [3.3](3.3_upgrade_indicators_technoeconomic.ipynb) | exports per-substation GeoPackages → QGIS |
| Table — PRIM employment thresholds (no-truck-stop subs) | [3.2](3.2_upgrade_indicators_geographic.ipynb) | `box.limits` |
| Figure A.10 — substation territory-area distribution | [2.3](2.3_augment_taz_data.ipynb) | inline ECDF |
| Figure A.11 — adoption correlation scatter matrix | [2.1](2.1_select_states_of_the_world.ipynb) | inline pair grid |
| Figure A.12 — trip-distance distribution | [2.2](2.2_augment_dwells_data.ipynb) | inline ECDF |
| Figure A.13 — validation vs Broga et al. (2024) | [4](4_validation.ipynb) | inline bar chart |

Notebook [3.2](3.2_upgrade_indicators_geographic.ipynb) also produces the
example-substation employment values and percentile ranks cited in the §3.2 text;
[3.3](3.3_upgrade_indicators_technoeconomic.ipynb) produces the per-substation PRIM
example numbers (Newark, Westley) cited in the §3.3 text.

## Notes

- **QGIS maps:** for map figures the notebook computes and exports a GeoPackage via
  `catalog.save(...)`; the final cartographic layout (basemap, colors, insets) is built
  in QGIS and is not reproduced here.
