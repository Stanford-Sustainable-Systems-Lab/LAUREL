import logging
from abc import ABC
from typing import Self

from megaPLuG.models.routing.containers import (
    AbstractContainerRunner,
    DockerContainerRunner,
    ApptainerContainerRunner,
)

logger = logging.getLogger(__name__)


class GraphhopperContainerRouter(ABC):
    """Abstract base class for container-based routing services."""

    # Known defaults from the GraphHopper image
    target_port: int = 8989
    target_graph_dir: str = "/data/default-gh"

    def __init__(
        self: Self,
        image: str,
        graph_dir: str,
        config_path: str,
        port: int = 8989,
        container_name: str = "graphhopper",
        startup_delay: int = 5,
        mem_max_gb: int = 32,
        mem_start_gb: int = 2,
        runner_class: AbstractContainerRunner = ApptainerContainerRunner,
        **cmd_kwargs,
    ):
        self.image = image
        self.container_name = container_name
        self.port = port
        self.graph_dir = graph_dir
        self.config_path = config_path
        self.startup_delay = startup_delay
        self.mem_max_gb = mem_max_gb
        self.mem_start_gb = mem_start_gb
        self.runner_class = runner_class
        self.cmd_kwargs = cmd_kwargs
        self.process = None
        self.base_url = None

    def __enter__(self: Self) -> Self:
        """Start the container and return a GraphHopper client."""
        self.container = self.runner_class(
            name=self.container_name,
            image=self.image,
            port_map={self.port: self.target_port},
            bind_map={self.graph_dir: self.target_graph_dir},
            env_vars={"JAVA_OPTS": f"-Xmx{self.mem_max_gb}g -Xms{self.mem_start_gb}g"},
        )
        self.container.stop_existing()
        if isinstance(self.container, ApptainerContainerRunner):
            cmd_dict = {
                "--config": self.config_path,
                "--graph-cache": self.graph_dir,
            }
        elif isinstance(self.container, DockerContainerRunner):
            logger.warning("Passed config will be overridden by the container's internal config.")
        else:
            raise RuntimeError("Unsupported AbstractContainerRunner class passed.")

        cmd_rout = self._build_router_command(cmd_dict=cmd_dict)
        logger.info("Starting GraphHopper routing server...")
        self.container.start(
            cmd=cmd_rout,
            startup_delay_secs=self.startup_delay,
        )
        self.base_url = f"http://localhost:{self.port}"
        return self

    def __exit__(self: Self, exc_type, exc_val, exc_tb):
        """Stop the container when exiting the context."""
        logger.info("Stopping GraphHopper routing server...")
        self.container.stop_existing()
        self.base_url = None

    def import_graph(self: Self, input_file: str) -> None:
        """Import the graph from the given input file (.pbf) and process it."""
        self.container = self.runner_class(
            name=self.container_name,
            image=self.image,
            port_map={self.port: self.target_port},
            bind_map={self.graph_dir: self.target_graph_dir},
            env_vars={"JAVA_OPTS": f"-Xmx{self.mem_max_gb}g -Xms{self.mem_start_gb}g"},
        )
        self.container.stop_existing()
        if isinstance(self.container, ApptainerContainerRunner):
            cmd_dict = {
                "--input": input_file,
                "--config": self.config_path,
                "--graph-cache": self.graph_dir,
            }
        elif isinstance(self.container, DockerContainerRunner):
            logger.warning("Passed config will be overridden by the container's internal config.")
        else:
            raise RuntimeError("Unsupported AbstractContainerRunner class passed.")

        cmd_rout = self._build_router_command(cmd_dict=cmd_dict)
        cmd_import = ["--import"] + cmd_rout
        logger.info("Starting GraphHopper graph import...")
        self.container.start(
            cmd=cmd_import,
            wait_for_completion=True,
            startup_delay_secs=self.startup_delay,
        )
        self.container.stop_existing()

    def _build_router_command(self: Self, cmd_dict: dict) -> list[str]:
        """Build the command args that go to the routing container itself."""
        # Add command arguments for within the container itself
        cmd = []
        if len(cmd_dict) > 0:
            cmd_ls = [item for pair in cmd_dict.items() for item in pair]
            cmd.extend(cmd_ls)
        return cmd
