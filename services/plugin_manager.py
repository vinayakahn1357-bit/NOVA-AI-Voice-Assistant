"""
services/plugin_manager.py — Plugin System for NOVA (Phase 7)
Defines the PluginInterface contract and PluginManager registry.

Phase 7 upgrades:
- Plugin categories: utility, data, web, automation
- Plugin metadata: input_schema, output_schema
- Category filtering
- Backward compatible: existing plugins without new properties still work

Plugin Contract:
    execute(input: dict) -> {"success": bool, "result": any, "error": str | None}
"""

import os
import importlib
import inspect
from abc import ABC, abstractmethod

from utils.logger import get_logger

log = get_logger("plugins")

# Required keys in plugin responses
_REQUIRED_RESPONSE_KEYS = {"success", "result", "error"}

# Valid plugin categories
PLUGIN_CATEGORIES = {"utility", "data", "web", "automation"}


class PluginInterface(ABC):
    """
    Abstract base class for NOVA plugins.

    Required (must implement):
        - name (property) → str
        - description (property) → str
        - execute(input: dict) → dict with keys: success, result, error

    Optional (Phase 7, with defaults):
        - category (property) → str (default: "utility")
        - input_schema (property) → dict | None
        - output_schema (property) → dict | None
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this plugin does."""
        ...

    @property
    def category(self) -> str:
        """
        Plugin category. One of: utility, data, web, automation.
        Default: "utility". Override in subclass to set.
        """
        return "utility"

    @property
    def input_schema(self) -> dict | None:
        """
        JSON-schema-like description of expected input keys.
        Example: {"query": "str", "max_results": "int"}
        Default: None (untyped).
        """
        return None

    @property
    def output_schema(self) -> dict | None:
        """
        JSON-schema-like description of output result structure.
        Example: {"items": "list", "total": "int"}
        Default: None (untyped).
        """
        return None

    @abstractmethod
    def execute(self, input: dict) -> dict:
        """
        Execute the plugin with given input.

        Returns:
            {
                "success": bool,
                "result": any,
                "error": str | None
            }
        """
        ...


class PluginManager:
    """
    Registry and executor for NOVA plugins.
    Validates plugin interface compliance before registration.
    Validates response shape after execution.
    Phase 7: supports categories, metadata, and filtering.
    """

    def __init__(self, plugins_dir: str = None):
        self._plugins: dict[str, PluginInterface] = {}
        self._plugins_dir = plugins_dir

        if plugins_dir:
            self._auto_discover(plugins_dir)

        log.info("PluginManager initialized: %d plugins registered", len(self._plugins))

    def register(self, plugin: PluginInterface) -> bool:
        """
        Register a plugin instance.
        Validates interface compliance before accepting.
        Returns True if registration was successful.
        """
        if not isinstance(plugin, PluginInterface):
            log.warning("Plugin rejected: %s does not implement PluginInterface",
                        type(plugin).__name__)
            return False

        name = None
        try:
            name = plugin.name
            desc = plugin.description
        except Exception as e:
            log.warning("Plugin rejected: failed to read name/description (%s)", e)
            return False

        if not name or not isinstance(name, str):
            log.warning("Plugin rejected: invalid or empty name")
            return False

        if name in self._plugins:
            log.warning("Plugin rejected: '%s' already registered", name)
            return False

        if not callable(getattr(plugin, "execute", None)):
            log.warning("Plugin '%s' rejected: execute() is not callable", name)
            return False

        # Validate category
        cat = getattr(plugin, "category", "utility")
        if cat not in PLUGIN_CATEGORIES:
            log.warning("Plugin '%s': unknown category '%s', defaulting to 'utility'", name, cat)

        self._plugins[name] = plugin
        log.info("Plugin registered: '%s' [%s] — %s", name, cat, desc or "(no description)")
        return True

    def unregister(self, name: str) -> bool:
        """Remove a registered plugin."""
        if name in self._plugins:
            del self._plugins[name]
            log.info("Plugin unregistered: '%s'", name)
            return True
        return False

    def execute(self, name: str, input: dict) -> dict:
        """
        Execute a registered plugin by name.
        Validates the response shape before returning.
        """
        plugin = self._plugins.get(name)
        if not plugin:
            return {
                "success": False,
                "result": None,
                "error": f"Plugin '{name}' not found",
            }

        try:
            response = plugin.execute(input)

            if not isinstance(response, dict):
                log.warning("Plugin '%s' returned non-dict response: %s",
                            name, type(response).__name__)
                return {
                    "success": False,
                    "result": None,
                    "error": "Plugin returned invalid response (not a dict)",
                }

            missing = _REQUIRED_RESPONSE_KEYS - set(response.keys())
            if missing:
                log.warning("Plugin '%s' response missing keys: %s", name, missing)
                return {
                    "success": False,
                    "result": None,
                    "error": f"Plugin returned invalid response (missing keys: {missing})",
                }

            return response

        except Exception as e:
            log.warning("Plugin '%s' execution error: %s", name, e, exc_info=True)
            return {
                "success": False,
                "result": None,
                "error": f"Plugin execution error: {str(e)}",
            }

    def has_plugin(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def list_plugins(self) -> list[dict]:
        """
        List all registered plugins with metadata.
        Phase 7: includes category, input_schema, output_schema.
        """
        results = []
        for p in self._plugins.values():
            entry = {
                "name": p.name,
                "description": p.description,
                "category": getattr(p, "category", "utility"),
            }
            # Include schemas if defined
            input_schema = getattr(p, "input_schema", None)
            output_schema = getattr(p, "output_schema", None)
            if input_schema:
                entry["input_schema"] = input_schema
            if output_schema:
                entry["output_schema"] = output_schema
            results.append(entry)
        return results

    def get_by_category(self, category: str) -> list[dict]:
        """Get all plugins matching a category."""
        return [
            p for p in self.list_plugins()
            if p.get("category") == category
        ]

    def get_plugin(self, name: str) -> PluginInterface | None:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def _auto_discover(self, plugins_dir: str):
        """
        Auto-discover plugins from a directory.
        Each .py file in the directory is imported.
        Any class that subclasses PluginInterface is registered.
        """
        if not os.path.isdir(plugins_dir):
            log.info("Plugins directory not found: %s (skipping discovery)", plugins_dir)
            return

        for filename in os.listdir(plugins_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            module_name = f"plugins.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)

                for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(attr_value, PluginInterface)
                            and attr_value is not PluginInterface):
                        try:
                            instance = attr_value()
                            self.register(instance)
                        except Exception as e:
                            log.warning("Failed to instantiate plugin %s.%s: %s",
                                        module_name, attr_name, e)

            except Exception as e:
                log.warning("Failed to load plugin module %s: %s", module_name, e)
