"""
LawEdu AI — Tool Registry (Placeholder).
Future: CRM, external API, database tool calling.
"""
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for external tools that can be called by the LLM.
    Currently a placeholder for future expansion.
    """

    def __init__(self):
        self._tools = {}

    def register(self, name: str, handler, description: str = ""):
        """Register a tool with a name and handler function."""
        self._tools[name] = {
            "handler": handler,
            "description": description,
        }
        logger.info(f"🔧 Registered tool: {name}")

    def call(self, name: str, **kwargs):
        """Call a registered tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]["handler"](**kwargs)

    def list_tools(self) -> list:
        """List all registered tools."""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self._tools.items()
        ]


# Global registry
tool_registry = ToolRegistry()
