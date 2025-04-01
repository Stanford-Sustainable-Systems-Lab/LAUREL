import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

from routingpy import Graphhopper


class GraphhopperContainerRouter(ABC):
    """Abstract base class for container-based routing services."""

    # Known defaults from the GraphHopper image
    target_port: int = 8989
    target_graph_dir: str = "/data/default-gh"

    def __init__(
        self: Self,
        image: str,
        graph_dir: str,
        port: int = 8989,
        container_name: str = "graphhopper",
        startup_delay: int = 5,
        **cmd_kwargs,
    ):
        self.image = image
        self.container_name = container_name
        self.port = port
        self.graph_dir = graph_dir
        self.startup_delay = startup_delay
        self.cmd_kwargs = cmd_kwargs
        self.process = None

    def __enter__(self: Self):
        """Start the container and return a GraphHopper client."""
        self.container = DockerContainerRunner(
            name=self.container_name,
            image=self.image,
            port_map={self.port: self.target_port},
            bind_map={self.graph_dir: self.target_graph_dir},
        )

        # Stop existing container if running
        self.container.stop_existing()

        # Start the container
        print("Starting GraphHopper...")
        cmd_rout = self._build_router_command()
        self.container.start(cmd=cmd_rout)

        # Wait for service to be ready
        print(f"Waiting {self.startup_delay} seconds for initialization...")
        time.sleep(self.startup_delay)

        # Check if container is running
        if not self.container.is_running():
            stderr = (
                self.container.process.stderr.read()
                if self.container.process
                else "No process error"
            )
            raise RuntimeError(
                f"Failed to start GraphHopper container. Error: {stderr}"
            )

        # Return the client
        return Graphhopper(base_url=f"http://localhost:{self.port}")

    def __exit__(self: Self, exc_type, exc_val, exc_tb):
        """Stop the container when exiting the context."""
        print("Stopping GraphHopper...")
        self.container.stop_existing()

    def _build_router_command(self: Self) -> list[str]:
        """Build the command args that go to the routing container itself."""
        # Add command arguments for within the container itself
        cmd = []
        if len(self.cmd_kwargs) > 0:
            cmd_ls = [item for pair in self.cmd_kwargs.items() for item in pair]
            cmd.extend(cmd_ls)
        return cmd


class AbstractContainerRunner(ABC):
    """An abstract runner for containers. It assumes that there is a single container
    that is running or not running.
    """

    def __init__(
        self: Self, name: str, image: str, port_map: dict = None, bind_map: dict = None
    ) -> None:
        """Initialize the container runner, but do not start the container yet.

        port_map and bind_map give the ports and directories, respectively to map. The
        keys of these dicts represent the local machine (source), and the values
        represent the container (target).
        """
        self.process = None
        self.name = name
        self.image = image
        self.port_map = port_map
        self.bind_map = bind_map
        self.up_cmd = self.build_command()

    @abstractmethod
    def build_command(self: Self) -> list[str]:
        """Build the container run command."""
        pass

    @staticmethod
    def _build_map_str_list(
        map: dict, str_func: Callable[[str, str], str]
    ) -> list[str]:
        """Build a list of strings for port and bind mappings."""
        ls = []
        for local, cont in map.items():
            ls.append(str_func(local, cont))
        return ls

    @staticmethod
    @abstractmethod
    def _build_port_map_str(local: str, cont: str) -> str:
        """Build the string version of a port mapping."""
        pass

    @staticmethod
    @abstractmethod
    def _build_bind_map_str(local: str, cont: str) -> str:
        """Build the string version of a bind mount mapping."""
        pass

    def start(self: Self, cmd: list[str]) -> None:
        """Start the container running using the command passed."""
        run_cmd = self.up_cmd + cmd
        self.process = subprocess.Popen(
            run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    @abstractmethod
    def is_running(self: Self) -> bool:
        """Check if the container is running."""
        pass

    @abstractmethod
    def stop_existing(self: Self) -> None:
        """Stop the container if it's running."""
        pass


class DockerContainerRunner(AbstractContainerRunner):
    """Docker implementation of container runner."""

    def build_command(self: Self) -> list[str]:
        """Build the docker run command."""
        port_maps = self._build_map_str_list(self.port_map, self._build_port_map_str)
        bind_maps = self._build_map_str_list(self.bind_map, self._build_bind_map_str)

        cmd = ["docker", "run", "-d"]
        cmd.extend(["--name", self.name])
        cmd.extend(["--publish"] + port_maps)
        cmd.extend(["--mount"] + bind_maps)
        cmd.extend([self.image])
        return cmd

    @staticmethod
    def _build_port_map_str(local, cont):
        return f"{local}:{cont}"

    @staticmethod
    def _build_bind_map_str(local, cont):
        return f"type=bind,src={local},dst={cont}"

    def is_running(self: Self) -> bool:
        """Check if the docker container is running."""
        cmd = [
            "docker",
            "ps",
            "--filter",
            f"name={self.name}",
            "--format",
            "{{.ID}}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())

    def stop_existing(self: Self) -> None:
        """Stop the docker container if it's running."""
        if self.is_running():
            subprocess.run(
                ["docker", "stop", self.name],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                check=False,
            )


class SingularityContainerRunner(AbstractContainerRunner):
    """Singularity implementation of container runner."""

    def build_command(self: Self) -> list[str]:
        """Build the singularity run command."""
        port_maps = self._build_map_str_list(self.port_map, self._build_port_map_str)
        bind_maps = self._build_map_str_list(self.bind_map, self._build_bind_map_str)

        port_maps = "portmap=" + ",".join(port_maps)

        cmd = ["singularity", "instance", "run"]
        cmd.extend(["--net", "--network-args"] + port_maps)
        cmd.extend(["--bind"] + bind_maps)
        cmd.extend([self.image])
        cmd.extend([self.name])
        return cmd

    @staticmethod
    def _build_port_map_str(local, cont):
        return f"{local}:{cont}/tcp"

    @staticmethod
    def _build_bind_map_str(local, cont):
        return f"{local}:{cont}"

    def is_running(self: Self) -> bool:
        """Check if the singularity container is running."""
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, check=False
        )
        return self.name in result.stdout

    def stop_existing(self: Self) -> None:
        """Stop the singularity container if it's running."""
        if self.is_running():
            subprocess.run(
                ["singularity", "instance", "stop", self.name],
                check=False,
            )
