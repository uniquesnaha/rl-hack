"""
DSAR Environment Client.

Provides a client for connecting to a running DSAR Environment server.
"""

from openenv.core import EnvClient


class DSAREnv(EnvClient):
    """Client for the DSAR Compliance Environment.

    Connects to a running DSAR environment server via WebSocket.

    Example:
        >>> with DSAREnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     result = env.step({"action_type": "query_silo", "silo_name": "billing"})
        ...     print(result.observation)
    """

    pass  # EnvClient provides all needed functionality
