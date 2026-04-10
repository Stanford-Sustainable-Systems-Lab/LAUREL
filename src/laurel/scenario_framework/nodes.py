"""Kedro node for dynamic scenario configuration dispatch.

Provides a single node function :func:`generate_scenario_configs` that looks
up the requested :class:`ScenarioBuilder` subclass in the class-level registry,
instantiates it, and delegates to its
:meth:`~laurel.scenario_framework.build.ScenarioBuilder.build_configs` method.

Usage
-----
Import this function into any Kedro pipeline's ``nodes.py`` alongside the
builder imports that should be available for that pipeline::

    # In your pipeline's nodes.py:
    import laurel.scenario_builders  # noqa: F401 — registers all bundled builders
    from laurel.scenario_framework.nodes import generate_scenario_configs  # noqa: F401

Any :class:`ScenarioBuilder` subclass imported (anywhere) before
:func:`generate_scenario_configs` is called becomes discoverable — importing
the module is the registration step.

Key design decisions
--------------------
- **``__init_subclass__`` registry**: builders self-register when their module
  is imported via :meth:`~laurel.scenario_framework.build.ScenarioBuilder.__init_subclass__`.
  This replaces the previous ``globals()``/``inspect`` approach, which required
  the function and all builder imports to live in the same module.
- **Import-as-registration**: callers signal intent by importing builder
  modules.  A plain ``import laurel.scenario_builders`` is enough to register
  all bundled builders; custom builders only need to be imported once anywhere
  before the node runs.
"""

from .build import ScenarioBuilder


def generate_scenario_configs(scen_params: dict, all_params: dict) -> tuple:
    """Dispatch to the named ScenarioBuilder and return configuration partitions.

    Looks up ``scen_params["builder"]`` in
    :attr:`~laurel.scenario_framework.build.ScenarioBuilder._registry`, instantiates the
    matching class, calls
    :meth:`~laurel.scenario_framework.build.ScenarioBuilder.build_configs`, and returns
    the resulting partitions dict together with the builder instance.

    The registry is populated automatically when a
    :class:`~laurel.scenario_framework.build.ScenarioBuilder` subclass module is
    imported.  Callers are responsible for ensuring the desired builders are
    imported before this function is called (typically via a module-level
    ``import laurel.scenario_builders`` in the pipeline's ``nodes.py``).

    Args:
        scen_params: Scenario-specific parameter dict.  Must contain:

            - ``"builder"`` *(str)*: class name of the concrete
              :class:`ScenarioBuilder` subclass to use.
            - ``"display_name"`` *(str)*: human-readable name forwarded to the
              builder constructor.
            - Any additional keys consumed by the chosen builder's
              ``_build_param_dicts`` implementation.

        all_params: The complete Kedro ``parameters`` dict for the run,
            forwarded unchanged to the builder constructor.

    Returns:
        Two-tuple ``(partitions, builder)`` where ``partitions`` is the dict
        of ``{config_file_path: param_dict}`` entries returned by
        :meth:`~laurel.scenario_framework.build.ScenarioBuilder.build_configs`, and
        ``builder`` is the instantiated builder (whose
        :attr:`~laurel.scenario_framework.build.ScenarioBuilder.n_tasks_generated`
        is set and available for downstream nodes such as
        :func:`~laurel.scenario_framework.cmd.generate_bash_script`).

    Raises:
        NotImplementedError: If no builder matching ``scen_params["builder"]``
            is found in :attr:`~laurel.scenario_framework.build.ScenarioBuilder._registry`.
    """
    bldr_name = scen_params["builder"]
    if bldr_name not in ScenarioBuilder._registry:
        raise NotImplementedError(
            f"Scenario builder '{bldr_name}' is not registered. "
            "Ensure its module is imported before calling generate_scenario_configs."
        )
    builder_cls = ScenarioBuilder._registry[bldr_name]
    builder = builder_cls(scen_params=scen_params, all_params=all_params)
    parts = builder.build_configs()
    return (parts, builder)
