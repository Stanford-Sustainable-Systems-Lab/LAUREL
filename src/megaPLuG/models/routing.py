import logging
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

from routingpy import Graphhopper

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
        port: int = 8989,
        container_name: str = "graphhopper",
        startup_delay: int = 5,
        mem_max_gb: int = 32,
        mem_start_gb: int = 2,
        **cmd_kwargs,
    ):
        self.image = image
        self.container_name = container_name
        self.port = port
        self.graph_dir = graph_dir
        self.startup_delay = startup_delay
        self.mem_max_gb = mem_max_gb
        self.mem_start_gb = mem_start_gb
        self.cmd_kwargs = cmd_kwargs
        self.process = None

    def __enter__(self: Self):
        """Start the container and return a GraphHopper client."""
        self.container = DockerContainerRunner(
            name=self.container_name,
            image=self.image,
            port_map={self.port: self.target_port},
            bind_map={self.graph_dir: self.target_graph_dir},
            env_vars={"JAVA_OPTS": f"-Xmx{self.mem_max_gb}g -Xms{self.mem_start_gb}g"},
        )
        self.container.stop_existing()
        cmd_rout = self._build_router_command()
        logger.info("Starting GraphHopper routing server...")
        self.container.start(cmd=cmd_rout)
        return Graphhopper(base_url=f"http://localhost:{self.port}")

    def __exit__(self: Self, exc_type, exc_val, exc_tb):
        """Stop the container when exiting the context."""
        logger.info("Stopping GraphHopper routing server...")
        self.container.stop_existing()

    def import_graph(self: Self, url: str) -> None:
        """Import the graph from the given URL and process it."""
        self.container = DockerContainerRunner(
            name=self.container_name,
            image=self.image,
            port_map={self.port: self.target_port},
            bind_map={self.graph_dir: self.target_graph_dir},
            env_vars={"JAVA_OPTS": f"-Xmx{self.mem_max_gb}g -Xms{self.mem_start_gb}g"},
        )
        self.container.stop_existing()
        cmd_import = ["--import", "--url", url]
        logger.info("Starting GraphHopper graph import...")
        self.container.start(cmd_import, wait_for_completion=True)
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
        self: Self,
        name: str,
        image: str,
        port_map: dict = None,
        bind_map: dict = None,
        env_vars: dict = None,
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
        self.env_vars = env_vars

    @abstractmethod
    def build_command(self: Self, wait_for_completion: bool = False) -> list[str]:
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

    @staticmethod
    @abstractmethod
    def _build_env_var_str(local: str, cont: str) -> str:
        """Build the string version of an environment variable mapping."""
        pass

    def start(
        self: Self,
        cmd: list[str],
        wait_for_completion: bool = False,
        startup_delay_secs: int = 5,
    ) -> None:
        """Start the container running using the command passed."""
        up_cmd = self.build_command(wait_for_completion=wait_for_completion)
        run_cmd = up_cmd + cmd

        logger.info("Starting container...")
        if wait_for_completion:
            self.process = subprocess.Popen(
                run_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            logger.info("Process output:")
            # Print output in real-time while process is running
            while self.process.poll() is None:
                self._log_container_prints(self.process.stdout.readline())
                self._log_container_prints(self.process.stderr.readline())

            # Read any remaining output after process completed
            for line in self.process.stdout:
                self._log_container_prints(line)

            for line in self.process.stderr:
                self._log_container_prints(line)

            # Check return code
            if self.process.returncode != 0:
                raise RuntimeError(
                    f"Process failed with return code {self.process.returncode}."
                )

            logger.info("Process completed successfully")

        else:
            self.process = subprocess.Popen(
                run_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Wait for service to be ready
            logger.info(f"Waiting {startup_delay_secs} seconds for initialization...")
            time.sleep(startup_delay_secs)
            self.check_is_running()
            logger.info("Container is running.")

    @staticmethod
    def _log_container_prints(line: str | None) -> None:
        if line:
            logger.info(f"[green]CONTAINER:[/] {line.rstrip()}", extra={"markup": True})

    @abstractmethod
    def is_running(self: Self) -> bool:
        """Check if the container is running."""
        pass

    def check_is_running(self: Self) -> None:
        """Report if the container is running."""
        # Check if container is running
        if not self.is_running():
            stderr = self.process.stderr.read() if self.process else "No process error"
            raise RuntimeError(f"Failed to start container. Error: {stderr}")

    @abstractmethod
    def stop_existing(self: Self) -> None:
        """Stop the container if it's running."""
        pass


class DockerContainerRunner(AbstractContainerRunner):
    """Docker implementation of container runner."""

    def build_command(self: Self, wait_for_completion: bool = False) -> list[str]:
        """Build the docker run command."""
        port_maps = self._build_map_str_list(self.port_map, self._build_port_map_str)
        bind_maps = self._build_map_str_list(self.bind_map, self._build_bind_map_str)
        env_vars = self._build_map_str_list(self.env_vars, self._build_env_var_str)

        cmd = ["docker", "run"]
        if not wait_for_completion:
            cmd.extend(["-d"])
        cmd.extend(["--name", self.name])
        cmd.extend(["--publish"] + port_maps)
        cmd.extend(["--mount"] + bind_maps)
        cmd.extend(["--env"] + env_vars)
        cmd.extend([self.image])
        return cmd

    @staticmethod
    def _build_port_map_str(local, cont):
        return f"{local}:{cont}"

    @staticmethod
    def _build_bind_map_str(local, cont):
        return f"type=bind,src={local},dst={cont}"

    @staticmethod
    def _build_env_var_str(key, val):
        return f"{key}={val}"

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

    def build_command(self: Self, wait_for_completion: bool = False) -> list[str]:
        """Build the singularity run command."""
        port_maps = self._build_map_str_list(self.port_map, self._build_port_map_str)
        bind_maps = self._build_map_str_list(self.bind_map, self._build_bind_map_str)
        # env_vars = self._build_map_str_list(self.env_vars, self._build_env_var_str)

        port_maps = "portmap=" + ",".join(port_maps)

        cmd = ["singularity", "instance", "run"]
        cmd.extend(["--net", "--network-args"] + port_maps)
        cmd.extend(["--bind"] + bind_maps)
        # TODO: Learn how to set environment variables for Singularity
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
