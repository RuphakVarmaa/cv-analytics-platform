"""
Shared mutable application state.
Populated during the lifespan startup and read by routers/services.
"""
from typing import Any, Dict

app_state: Dict[str, Any] = {}
