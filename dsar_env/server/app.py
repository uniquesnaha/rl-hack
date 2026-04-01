"""
FastAPI application for the DSAR Environment.

Uses OpenEnv's create_app() helper to expose the environment
over HTTP and WebSocket endpoints.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

from dsar_env.models import DSARAction, DSARObservation
from .dsar_environment import DSAREnvironment

# Create the app using OpenEnv's factory helper.
# Pass the CLASS (not instance) — the framework creates instances per session.
app = create_app(
    DSAREnvironment, DSARAction, DSARObservation, env_name="dsar_env"
)


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
