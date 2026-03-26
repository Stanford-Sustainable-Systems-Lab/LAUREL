"""Bash and SLURM script generation for batch scenario execution.

Provides :class:`ScenarioBashWriter` for generating shell scripts that launch
Kedro scenario arrays either locally or via SLURM, and a thin pipeline-facing
wrapper :func:`generate_bash_script` used from the ``build_runners`` pipeline.

Key design decisions
--------------------
- **Three command modes**: ``sbatch`` generates a self-contained SLURM batch
  script with ``#SBATCH`` directives and dynamic config-directory discovery;
  ``salloc`` generates an interactive allocation command (the ``--array``
  option is stripped since ``salloc`` does not support job arrays);
  ``local`` wraps the Kedro call in a shell ``for`` loop that iterates over
  ``SLURM_ARRAY_TASK_ID`` values, mimicking SLURM array behaviour for local
  testing.
- **Runtime config discovery**: rather than hardcoding task paths, the
  generated script uses ``find`` to locate the ``task_$SLURM_ARRAY_TASK_ID``
  directory under ``conf/scenarios/<name>/`` at runtime.  This keeps the
  script independent of the absolute project path on the HPC cluster.
- **Option building via string template**: :meth:`ScenarioBashWriter.build_opts`
  uses a simple ``KEY``/``VALUE`` placeholder template rather than a
  CLI-parsing library, keeping the output predictable and easy to inspect.
"""

from pathlib import Path
from typing import Self

from .build import ScenarioBuilder


class ScenarioBashWriter:
    """Generates shell scripts that launch Kedro scenario arrays.

    Supports three execution modes (``sbatch``, ``salloc``, ``local``)
    controlled at construction time.  Use :meth:`compile` to obtain the
    complete script string, then write it to disk or pass it to the
    ``build_runners`` pipeline for storage.

    Args:
        name: Human-readable scenario name (matches
            :attr:`~megaplug.scenarios.build.ScenarioBuilder.display_name`).
            Used as the SLURM job name and the config search path.
        command: Execution mode — one of ``"sbatch"``, ``"salloc"``, or
            ``"local"``.
    """

    def __init__(self: Self, name: str, command: str) -> None:
        self.name = name
        assert command in ["salloc", "sbatch", "local"]
        self.command = command

    def build_slurm_request(self: Self, resources: dict, reporting: dict = None) -> str:
        """Build the SLURM resource-request block for ``sbatch`` or ``salloc`` mode.

        For ``sbatch``, produces ``#SBATCH --key=value`` header lines (one per
        resource option) and injects ``job-name`` automatically.  For
        ``salloc``, produces a single ``salloc --key=value ...`` command string
        and strips the ``array`` option, which ``salloc`` does not support.

        Args:
            resources: Dict of SLURM resource options (e.g.
                ``{"ntasks": 4, "mem": "64G", "array": "0-511"}``).
                Modified in-place to add ``job-name`` for ``sbatch`` mode.
            reporting: Optional dict of additional SLURM options (e.g. email
                and output-file settings) merged into ``resources`` for
                ``sbatch`` mode only.

        Returns:
            Multi-line string of ``#SBATCH`` directives (``sbatch`` mode) or
            a single ``salloc ...`` command line (``salloc`` mode).

        Raises:
            RuntimeError: If ``resources`` is ``None``.
        """
        # Set up resources
        if resources is None:
            raise RuntimeError(
                "Requested resources must be specified when the command is 'sbatch' or 'salloc'."
            )
        res_opts = resources
        if self.command == "sbatch":
            prefix = ""
            template = "#SBATCH --KEY=VALUE\n"
            res_opts.update({"job-name": self.name})
            if reporting is not None:
                res_opts.update(reporting)
        elif self.command == "salloc":
            prefix = "salloc"
            template = " --KEY=VALUE"
            res_opts.pop("array", None)  # Remove the array argument if it's given
        slurm_opts = self.build_opts(d=res_opts, template=template)
        slurm_opts = "".join([prefix, slurm_opts])
        return slurm_opts

    def build_kedro_run(
        self: Self,
        params: dict,
        prefix: str = "",
        n_tasks: int = 1,
    ) -> str:
        """Build the shell commands that discover the config dir and invoke Kedro.

        For ``sbatch`` mode, produces two lines: a ``find`` command that sets
        ``conf_dir`` from ``SLURM_ARRAY_TASK_ID``, then a ``kedro run`` call
        using ``--env=$conf_dir``.  For ``local`` mode, wraps these lines in a
        ``for`` loop over ``seq 0 <n_tasks-1>`` that simulates a SLURM array.

        Args:
            params: Dict of Kedro CLI options to pass to ``kedro run`` (e.g.
                ``{"pipeline": "electrify_trips"}``).  A ``"prefix"`` key is
                expected at the call site and extracted before this method is
                called; the remaining keys are forwarded here.  The ``"env"``
                key is injected automatically and should not be included.
            prefix: Shell command prefix inserted before ``kedro run`` (e.g.
                ``"srun"`` for MPI or ``"python -m cProfile"`` for profiling).
                Defaults to ``""``.
            n_tasks: Total number of tasks in the array; used only for
                ``local`` mode to set the ``for`` loop range.  Defaults to
                ``1``.

        Returns:
            Multi-line shell string containing the config-discovery and
            ``kedro run`` invocation(s).
        """
        # If we're doing scenario runs, then find the appropriate config environment
        conf_lines = ["cd conf"]
        scen_pth = Path("scenarios") / self.name
        conf_dir_cmd = (
            f'conf_dir=$(find {scen_pth} -type d -name "task_$SLURM_ARRAY_TASK_ID")'
        )
        conf_lines.append(conf_dir_cmd)
        conf_lines.append("cd ..")
        conf_finder = " && ".join(conf_lines)

        if params is None:
            params = {}
        params.update({"env": "$conf_dir"})

        # Write kedro call
        kedro_opts = self.build_opts(d=params, template=" --KEY=VALUE")
        cmds = f"{prefix} kedro run{kedro_opts}"

        lines = []
        if self.command == "sbatch":
            lines.append(conf_finder)
            lines.append(cmds)
        else:
            lines.append(f"for SLURM_ARRAY_TASK_ID in `seq 0 {n_tasks - 1}`; do")
            lines.append(f"\t{conf_finder}")
            lines.append(f"\t{cmds}")
            lines.append("done")
        kedro_runner = "\n".join(lines)
        return kedro_runner

    def compile(
        self: Self,
        params: dict,
        resources: dict = None,
        reporting: dict = None,
        n_tasks: int = None,
    ) -> str:
        """Compile all script sections into a complete shell script string.

        Assembles the shebang line, optional SLURM directives, and the Kedro
        run block into a single multi-line string suitable for writing to a
        ``.sh`` file.

        Args:
            params: Dict with two required keys:

                - ``"prefix"``: Shell command prefix forwarded to
                  :meth:`build_kedro_run` (e.g. ``"srun"`` or ``""``).
                - ``"kedro"``: Dict of Kedro CLI options forwarded to
                  :meth:`build_kedro_run`.

            resources: SLURM resource dict forwarded to
                :meth:`build_slurm_request`.  Required when ``command`` is
                ``"sbatch"`` or ``"salloc"``; ignored for ``"local"``.
            reporting: Optional SLURM reporting options (email, output paths)
                forwarded to :meth:`build_slurm_request`.
            n_tasks: Total number of array tasks; forwarded to
                :meth:`build_kedro_run` for ``local`` mode loop sizing.

        Returns:
            Complete shell script string starting with ``#!/bin/bash``.
        """
        lines = ["#!/bin/bash"]
        if self.command in ["sbatch", "salloc"]:
            slurm_opts = self.build_slurm_request(
                resources=resources, reporting=reporting
            )
            lines += [slurm_opts]

        kedro_run = self.build_kedro_run(
            prefix=params["prefix"],
            params=params["kedro"],
            n_tasks=n_tasks,
        )
        lines += [kedro_run]
        sh = "\n".join(lines)
        return sh

    @staticmethod
    def build_opts(d: dict[str : str | int | float], template: str) -> str:
        """Render a dict of key-value pairs into a shell option string.

        Substitutes each ``(key, value)`` pair into ``template`` by replacing
        the literal strings ``KEY`` and ``VALUE``.  Concatenates all rendered
        lines into a single string.

        Args:
            d: Ordered dict of option names to values (e.g.
                ``{"ntasks": 4, "mem": "64G"}``).
            template: Format string containing the placeholders ``KEY`` and
                ``VALUE`` (e.g. ``" --KEY=VALUE"`` or ``"#SBATCH --KEY=VALUE\\n"``).

        Returns:
            Concatenated option string (e.g. ``" --ntasks=4 --mem=64G"``).
        """
        opt_lines = []
        for k, v in d.items():
            line = template.replace("KEY", str(k))
            line = line.replace("VALUE", str(v))
            opt_lines.extend(line)
        out_str = "".join(opt_lines)
        return out_str


def generate_bash_script(
    command: str,
    builder: ScenarioBuilder,
    cmd_params: dict = None,
    resources: dict = None,
    reporting: dict = None,
) -> dict[Path:str]:
    """Generate a Bash script for running a full scenario array.

    Constructs a :class:`ScenarioBashWriter` from the builder's display name
    and compiles the complete script.  Intended to be called as a Kedro node
    in the ``build_runners`` pipeline after :meth:`ScenarioBuilder.build_configs`
    has been run (so that :attr:`~ScenarioBuilder.n_tasks_generated` is set).

    Args:
        command: Execution mode passed to :class:`ScenarioBashWriter` —
            one of ``"sbatch"``, ``"salloc"``, or ``"local"``.
        builder: Configured :class:`ScenarioBuilder` instance whose
            ``display_name`` and ``n_tasks_generated`` are used.
        cmd_params: Dict with keys ``"prefix"`` and ``"kedro"`` forwarded to
            :meth:`ScenarioBashWriter.compile`.
        resources: SLURM resource dict forwarded to
            :meth:`ScenarioBashWriter.compile`.
        reporting: Optional SLURM reporting dict forwarded to
            :meth:`ScenarioBashWriter.compile`.

    Returns:
        Single-entry dict ``{builder.display_name: script_string}`` suitable
        for saving as a Kedro ``PartitionedDataset``.
    """
    writer = ScenarioBashWriter(name=builder.display_name, command=command)
    sh = writer.compile(
        params=cmd_params,
        n_tasks=builder.n_tasks_generated,
        resources=resources,
        reporting=reporting,
    )
    return {builder.display_name: sh}
