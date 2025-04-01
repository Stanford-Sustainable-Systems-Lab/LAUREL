import subprocess
import time
from abc import ABC, abstractmethod
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
        # Stop existing container if running
        self._stop_existing_container()

        # Build the command based on container type
        cmd_cont = self._build_container_command()
        cmd_rout = self._build_router_command()
        cmd = cmd_cont + cmd_rout

        # Start the container
        print("Starting GraphHopper...")
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for service to be ready
        print(f"Waiting {self.startup_delay} seconds for initialization...")
        time.sleep(self.startup_delay)

        # Check if container is running
        if not self._is_container_running():
            stderr = self.process.stderr.read() if self.process else "No process error"
            raise RuntimeError(
                f"Failed to start GraphHopper container. Error: {stderr}"
            )

        # Return the client
        return Graphhopper(base_url=f"http://localhost:{self.port}")

    def __exit__(self: Self, exc_type, exc_val, exc_tb):
        """Stop the container when exiting the context."""
        print("Stopping GraphHopper...")
        self._stop_existing_container()

    def _build_router_command(self: Self) -> list[str]:
        """Build the command args that go to the routing container itself."""
        # Add command arguments for within the container itself
        cmd = []
        if len(self.cmd_kwargs) > 0:
            cmd_ls = [item for pair in self.cmd_kwargs.items() for item in pair]
            cmd.extend(cmd_ls)
        return cmd

    @abstractmethod
    def _build_container_command(self: Self) -> list[str]:
        """Build the container run command."""
        pass

    @abstractmethod
    def _is_container_running(self: Self) -> bool:
        """Check if the container is running."""
        pass

    @abstractmethod
    def _stop_existing_container(self: Self) -> None:
        """Stop the container if it's running."""
        pass


class GraphhopperDockerRouter(GraphhopperContainerRouter):
    """Docker implementation of GraphHopper container router."""

    def _build_container_command(self: Self) -> list[str]:
        """Build the docker run command."""
        cmd = ["docker", "run", "-d"]
        cmd.extend(["--name", self.container_name])
        cmd.extend(["--publish", f"{self.port}:{self.target_port}"])
        cmd.extend(
            ["--mount", f"type=bind,src={self.graph_dir},dst={self.target_graph_dir}"]
        )
        cmd.extend([self.image])
        return cmd

    def _is_container_running(self: Self) -> bool:
        """Check if the docker container is running."""
        cmd = [
            "docker",
            "ps",
            "--filter",
            f"name={self.container_name}",
            "--format",
            "{{.ID}}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())

    def _stop_existing_container(self: Self) -> None:
        """Stop the docker container if it's running."""
        if self._is_container_running():
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                check=False,
            )


class GraphhopperSingularityRouter(GraphhopperContainerRouter):
    """Singularity implementation of GraphHopper container router."""

    def _build_container_command(self: Self) -> list[str]:
        """Build the singularity run command."""
        cmd = ["singularity", "instance", "run"]
        cmd.extend(
            [
                "--net",
                "--network-args",
                f"portmap={self.port}:{self.target_port}/tcp",
            ]
        )
        cmd.extend(["--bind", f"{self.graph_dir}:{self.target_graph_dir}"])
        cmd.extend([self.image])
        cmd.extend([self.container_name])
        return cmd

    def _is_container_running(self: Self) -> bool:
        """Check if the singularity container is running."""
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, check=False
        )
        return self.container_name in result.stdout

    def _stop_existing_container(self: Self) -> None:
        """Stop the singularity container if it's running."""
        if self._is_container_running():
            subprocess.run(
                ["singularity", "instance", "stop", self.container_name],
                check=False,
            )


# Factory function to maintain backward compatibility
def GraphhopperContainerRouter(
    image: str,
    graph_dir: str,
    port: int = 8989,
    container_engine: str = "docker",
    container_name: str = "graphhopper",
    startup_delay: int = 5,
    **cmd_kwargs,
) -> GraphhopperContainerRouter:
    """Factory function to create the appropriate container router."""
    if container_engine == "docker":
        return GraphhopperDockerRouter(
            image=image,
            graph_dir=graph_dir,
            port=port,
            container_name=container_name,
            startup_delay=startup_delay,
            **cmd_kwargs,
        )
    elif container_engine == "singularity":
        return GraphhopperSingularityRouter(
            image=image,
            graph_dir=graph_dir,
            port=port,
            container_name=container_name,
            startup_delay=startup_delay,
            **cmd_kwargs,
        )
    else:
        raise ValueError(f"Unsupported container engine: {container_engine}")
