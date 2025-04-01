import subprocess
import time
from typing import Self

from routingpy import Graphhopper


class GraphhopperContainerRouter:
    """Context manager class to handle GraphHopper container lifecycle.

    This class can run a Docker container using either Docker or Singularity.

    This class starts a GraphHopper routing container, waits for it to initialize,
    and cleans it up when done.
    """

    # Known defaults from the GraphHopper image
    target_port: int = 8989
    target_graph_dir: str = "/data/default-gh"

    def __init__(
        self: Self,
        image: str,
        graph_dir: str,
        port: int = 8989,
        container_engine: str = "docker",  # "docker" or "singularity"
        container_name: str = "graphhopper",
        startup_delay: int = 5,
        **cmd_kwargs,
    ):
        if container_engine not in ["docker", "singularity"]:
            raise ValueError(f"Unsupported container engine: {self.container_engine}")
        self.container_engine = container_engine
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
        cmd = self._build_run_command()

        # Start the container
        print(f"Starting GraphHopper with {self.container_engine}...")
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
        print(f"Stopping GraphHopper with {self.container_engine}...")
        self._stop_existing_container()

    def _build_run_command(self: Self):
        """Build the container command based on the container type."""
        cmd = [self.container_engine]

        if self.container_engine == "docker":
            cmd.extend(["run", "-d"])
            cmd.extend(["--name", self.container_name])
            cmd.extend(["--publish", f"{self.port}:{self.target_port}"])
            cmd.extend(
                [
                    "--mount",
                    f"type=bind,src={self.graph_dir},dst={self.target_graph_dir}",
                ]
            )  # Read-only option causes container crash for unknown reason
            cmd.extend([self.image])

        elif self.container_engine == "singularity":
            cmd.extend(["instance", "run"])
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

        # Add command arguments for within the container itself
        if len(self.cmd_kwargs) > 0:
            cmd_ls = [item for pair in self.cmd_kwargs.items() for item in pair]
            cmd.extend(cmd_ls)

        return cmd

    def _is_container_running(self):
        """Check if the container is running."""
        if self.container_engine == "docker":
            cmd = [
                "docker",
                "ps",
                "--filter",
                f"name={self.container_name}",
                "--format",
                "{{.ID}}",
            ]
            # If I run into trouble with the processs hanging, consider adding a timeout argument
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return bool(result.stdout.strip())

        elif self.container_engine == "singularity":
            # Consider using https://docs.sylabs.io/guides/4.2/user-guide/cli/singularity_instance_stats.html

            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, check=False
            )
            return self.container_name in result.stdout

    def _stop_existing_container(self):
        """Stop the container if it's running."""
        if self._is_container_running():
            if self.container_engine == "docker":
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
            elif self.container_engine == "singularity":
                subprocess.run(
                    ["singularity", "instance", "stop", self.container_name],
                    check=False,
                )
