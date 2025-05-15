import logging
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

logger = logging.getLogger(__name__)


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


class ApptainerContainerRunner(AbstractContainerRunner):
    """Apptainer implementation of container runner."""

    def build_command(self: Self, wait_for_completion: bool = False) -> list[str]:
        """Build the Apptainer run command."""
        port_maps = self._build_map_str_list(self.port_map, self._build_port_map_str)
        bind_maps = self._build_map_str_list(self.bind_map, self._build_bind_map_str)
        # env_vars = self._build_map_str_list(self.env_vars, self._build_env_var_str)

        port_maps = "portmap=" + ",".join(port_maps)

        cmd = ["apptainer", "instance", "run"]
        cmd.extend(["--compat", "--no-home"])
        cmd.extend(["--bind"] + bind_maps)
        # TODO: Learn how to set environment variables for Apptainer
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
        """Check if the Apptainer container is running."""
        result = subprocess.run(
            ["apptainer", "instance", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return self.name in result.stdout

    def stop_existing(self: Self) -> None:
        """Stop the Apptainer container if it's running."""
        if self.is_running():
            subprocess.run(
                ["apptainer", "instance", "stop", self.name],
                check=False,
            )
