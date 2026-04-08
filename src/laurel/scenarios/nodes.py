"""Kedro node for dynamic scenario configuration dispatch.

Provides a single node function :func:`generate_scenario_configs` that
introspects the module's global namespace for :class:`ScenarioBuilder`
subclasses, instantiates the one named in ``scen_params["builder"]``, and
delegates to its :meth:`~laurel.scenarios.build.ScenarioBuilder.build_configs`
method.

Usage
-----
This file is designed to be **copied verbatim** into any Kedro pipeline's
``nodes.py`` alongside the relevant builder import, enabling out-of-package
builder definitions::

    # In your pipeline's nodes.py:
    from laurel.scenarios.nodes import generate_scenario_configs  # noqa: F401
    from my_project.builders import MySoW512Builder  # noqa: F401

Any :class:`ScenarioBuilder` subclass imported into the file's namespace
becomes automatically discoverable — no registration step is needed.

Key design decisions
--------------------
- **Global-namespace introspection**: the function discovers available builders
  via ``inspect.isclass`` + ``issubclass`` rather than a static registry.
  This makes adding a new builder as simple as importing it; the dispatch
  logic does not need to be updated.
- **``# noqa: F401`` import of ``ScenarioBuilder``**: this import is
  intentional even though the name is not referenced directly.  It ensures
  the base class is present in the namespace so the ``issubclass`` check
  can exclude it (and any other indirect imports) from the builder map.
"""

import inspect

from .build import ScenarioBuilder  # noqa: F401


def generate_scenario_configs(scen_params: dict, all_params: dict) -> dict:
    """Dispatch to the named ScenarioBuilder and return configuration partitions.

    Scans all names in the module's global namespace for concrete subclasses of
    :class:`~laurel.scenarios.build.ScenarioBuilder`, instantiates the one
    whose class name matches ``scen_params["builder"]``, calls
    :meth:`~laurel.scenarios.build.ScenarioBuilder.build_configs`, and
    returns the resulting partitions dict together with the builder instance.

    Args:
        scen_params: Scenario-specific parameter dict.  Must contain:

            - ``"builder"`` *(str)*: class name of the concrete
              :class:`ScenarioBuilder` subclass to use (must be importable
              into this module's namespace).
            - ``"display_name"`` *(str)*: human-readable name forwarded to the
              builder constructor.
            - Any additional keys consumed by the chosen builder's
              ``_build_param_dicts`` implementation.

        all_params: The complete Kedro ``parameters`` dict for the run,
            forwarded unchanged to the builder constructor.

    Returns:
        Two-tuple ``(partitions, builder)`` where ``partitions`` is the dict
        of ``{config_file_path: param_dict}`` entries returned by
        :meth:`~laurel.scenarios.build.ScenarioBuilder.build_configs`, and
        ``builder`` is the instantiated builder (whose
        :attr:`~laurel.scenarios.build.ScenarioBuilder.n_tasks_generated`
        is set and available for downstream nodes such as
        :func:`~laurel.scenarios.cmd.generate_bash_script`).

    Raises:
        NotImplementedError: If no builder matching ``scen_params["builder"]``
            is found in the module's global namespace.
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
