"""Central declarative registry for context scout language and connector plugins."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from typing import ClassVar

from context_scout.plugins.base import DomainPlugin, LanguagePlugin, SourceConnectorPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Thread-safe declarative registry with priority-ordered plugin dispatch."""

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _language_plugins: ClassVar[dict[str, LanguagePlugin]] = {}
    _specialized_plugins: ClassVar[list[tuple[int, LanguagePlugin]]] = []
    _extension_map: ClassVar[dict[str, LanguagePlugin]] = {}
    _connector_plugins: ClassVar[dict[str, SourceConnectorPlugin]] = {}
    _domain_plugins: ClassVar[dict[str, DomainPlugin]] = {}
    _unregistered_languages: ClassVar[set[str]] = set()
    _initialized: ClassVar[bool] = False
    _suppress_auto_init: ClassVar[bool] = False

    @classmethod
    def register_language(cls, plugin: LanguagePlugin, priority: int = 100) -> None:
        """Registers a language plugin.

        Lower priority numbers indicate higher precedence during dispatch (e.g. 10 before 100).
        """
        with cls._lock:
            if not cls._initialized and not cls._suppress_auto_init:
                cls._ensure_initialized()
            lang_key = plugin.language_id.lower()
            cls._unregistered_languages.discard(lang_key)
            cls._language_plugins[lang_key] = plugin
            # Remove any existing registration for this plugin before adding
            cls._specialized_plugins = [
                (prio, p)
                for prio, p in cls._specialized_plugins
                if p.language_id.lower() != lang_key
            ]
            cls._specialized_plugins.append((priority, plugin))
            cls._specialized_plugins.sort(key=lambda item: item[0])
            for ext in plugin.file_extensions:
                cls._extension_map[ext.lower()] = plugin
            logger.debug(
                "Registered language plugin '%s' (priority=%d, extensions=%s)",
                plugin.language_id,
                priority,
                plugin.file_extensions,
            )

    @classmethod
    def unregister_language(cls, language_id: str) -> None:
        """Removes a language plugin from the registry."""
        with cls._lock:
            lang_key = language_id.lower()
            cls._unregistered_languages.add(lang_key)
            plugin = cls._language_plugins.pop(lang_key, None)
            if plugin:
                cls._specialized_plugins = [
                    (prio, p)
                    for prio, p in cls._specialized_plugins
                    if p.language_id.lower() != lang_key
                ]
                for ext in plugin.file_extensions:
                    if cls._extension_map.get(ext.lower()) == plugin:
                        cls._extension_map.pop(ext.lower(), None)

    @classmethod
    def register_connector(cls, plugin: SourceConnectorPlugin) -> None:
        """Registers an external source connector plugin."""
        with cls._lock:
            key = getattr(plugin, "source_id", getattr(plugin, "source_type", "")).lower()
            cls._connector_plugins[key] = plugin
            logger.debug("Registered connector plugin '%s'", key)

    @classmethod
    def register_domain(cls, plugin: DomainPlugin) -> None:
        """Registers a domain plugin for one or more domain identifiers."""
        with cls._lock:
            for d_id in plugin.supported_domain_ids():
                cls._domain_plugins[d_id.lower()] = plugin
            cls._domain_plugins[plugin.domain_id.lower()] = plugin
            logger.debug("Registered domain plugin '%s'", plugin.domain_id)

    @classmethod
    def get_domain_plugin(cls, domain_id: str) -> DomainPlugin | None:
        """Retrieves a registered domain plugin handling domain_id."""
        cls._ensure_initialized()
        with cls._lock:
            return cls._domain_plugins.get(domain_id.lower())

    @classmethod
    def get_primary_domain_plugin(cls) -> DomainPlugin | None:
        """Returns the first registered domain plugin if any."""
        cls._ensure_initialized()
        with cls._lock:
            for plugin in cls._domain_plugins.values():
                return plugin
            return None

    @classmethod
    def list_domain_plugins(cls) -> list[DomainPlugin]:
        """Returns unique registered domain plugin instances."""
        cls._ensure_initialized()
        with cls._lock:
            seen: dict[int, DomainPlugin] = {}
            for plugin in cls._domain_plugins.values():
                seen[id(plugin)] = plugin
            return list(seen.values())

    @classmethod
    def get_language_plugin(cls, language_id: str) -> LanguagePlugin | None:
        """Retrieves a language plugin by ID."""
        cls._ensure_initialized()
        with cls._lock:
            return cls._language_plugins.get(language_id.lower())

    @classmethod
    def get_connector(cls, source_id: str) -> SourceConnectorPlugin | None:
        """Retrieves a connector plugin by ID."""
        cls._ensure_initialized()
        with cls._lock:
            return cls._connector_plugins.get(source_id.lower())

    @classmethod
    def get_plugin_for_file(cls, file_path: str) -> LanguagePlugin | None:
        """Finds the best matching language plugin for a given file path.

        Evaluates specialized plugins in priority order first; then tests extension mapping.
        """
        cls._ensure_initialized()
        normalized_path = file_path.replace("\\", "/").lower()

        with cls._lock:
            # 1. Check specialized handlers in priority order
            for _, plugin in cls._specialized_plugins:
                try:
                    if plugin.can_handle(file_path):
                        return plugin
                except Exception as exc:
                    logger.warning(
                        "Plugin '%s'.can_handle check failed: %s",
                        plugin.language_id,
                        exc,
                    )

            # 2. Match longest file extension (confirming with can_handle if needed)
            for ext, plugin in sorted(
                cls._extension_map.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):
                if normalized_path.endswith(ext):
                    try:
                        if plugin.can_handle(file_path):
                            return plugin
                    except Exception as exc:
                        logger.warning(
                            "Plugin '%s'.can_handle check failed: %s",
                            plugin.language_id,
                            exc,
                        )

        return None

    @classmethod
    def list_supported_languages(cls) -> list[str]:
        """Returns list of registered language plugin identifiers."""
        cls._ensure_initialized()
        with cls._lock:
            return sorted(cls._language_plugins.keys())

    @classmethod
    def list_supported_connectors(cls) -> list[str]:
        """Returns list of registered connector plugin identifiers."""
        cls._ensure_initialized()
        with cls._lock:
            return sorted(cls._connector_plugins.keys())

    @classmethod
    def reset(cls) -> None:
        """Clears all registered plugins (for test teardown)."""
        with cls._lock:
            cls._language_plugins.clear()
            cls._specialized_plugins.clear()
            cls._extension_map.clear()
            cls._connector_plugins.clear()
            cls._domain_plugins.clear()
            cls._unregistered_languages.clear()
            cls._initialized = False

    @classmethod
    @contextlib.contextmanager
    def temporary_registry(cls, *, include_builtins: bool = False) -> Iterator[None]:
        """Context manager isolating test mutations and restoring registry state cleanly."""
        with cls._lock:
            if not cls._initialized and not cls._suppress_auto_init:
                cls._ensure_initialized()
            prev_langs = dict(cls._language_plugins)
            prev_spec = list(cls._specialized_plugins)
            prev_ext = dict(cls._extension_map)
            prev_conn = dict(cls._connector_plugins)
            prev_dom = dict(cls._domain_plugins)
            prev_unreg = set(cls._unregistered_languages)
            prev_init = cls._initialized
            prev_suppress = cls._suppress_auto_init
            if not include_builtins:
                cls._language_plugins.clear()
                cls._specialized_plugins.clear()
                cls._extension_map.clear()
                cls._connector_plugins.clear()
                cls._domain_plugins.clear()
                cls._unregistered_languages.clear()
                cls._initialized = True
                cls._suppress_auto_init = True
        try:
            yield
        finally:
            with cls._lock:
                cls._language_plugins = prev_langs
                cls._specialized_plugins = prev_spec
                cls._extension_map = prev_ext
                cls._connector_plugins = prev_conn
                cls._domain_plugins = prev_dom
                cls._unregistered_languages = prev_unreg
                cls._initialized = prev_init
                cls._suppress_auto_init = prev_suppress

    @classmethod
    def _ensure_initialized(cls) -> None:
        with cls._lock:
            needs_reload = not cls._initialized or (
                not cls._suppress_auto_init
                and not cls._language_plugins
                and not cls._unregistered_languages
            )
            if needs_reload:
                cls._initialized = True
                prev_suppress = cls._suppress_auto_init
                cls._suppress_auto_init = True
                try:
                    cls._load_builtin_plugins()
                finally:
                    cls._suppress_auto_init = prev_suppress

    @classmethod
    def _load_builtin_plugins(cls) -> None:
        """Initializes and registers core language and connector plugins."""
        try:
            from context_scout.plugins.languages.storybook import StorybookPlugin

            if "storybook" not in cls._unregistered_languages:
                cls.register_language(StorybookPlugin(), priority=10)
        except ImportError as err:
            logger.debug("StorybookPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.react_redux import ReactReduxPlugin

            if "react_redux" not in cls._unregistered_languages:
                cls.register_language(ReactReduxPlugin(), priority=20)
        except ImportError as err:
            logger.debug("ReactReduxPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.templates_i18n import TemplatesI18nPlugin

            if "templates_i18n" not in cls._unregistered_languages:
                cls.register_language(TemplatesI18nPlugin(), priority=25)
        except ImportError as err:
            logger.debug("TemplatesI18nPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.api_specs import ApiSpecsPlugin

            if "api_specs" not in cls._unregistered_languages:
                cls.register_language(ApiSpecsPlugin(), priority=28)
        except ImportError as err:
            logger.debug("ApiSpecsPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.terraform import TerraformPlugin

            if "terraform" not in cls._unregistered_languages:
                cls.register_language(TerraformPlugin(), priority=30)
        except ImportError as err:
            logger.debug("TerraformPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.csharp import CSharpPlugin

            if "csharp" not in cls._unregistered_languages:
                cls.register_language(CSharpPlugin(), priority=50)
        except ImportError as err:
            logger.debug("CSharpPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.java_spring import JavaSpringPlugin

            if "java" not in cls._unregistered_languages:
                cls.register_language(JavaSpringPlugin(), priority=50)
        except ImportError as err:
            logger.debug("JavaSpringPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.kotlin import KotlinPlugin

            if "kotlin" not in cls._unregistered_languages:
                cls.register_language(KotlinPlugin(), priority=50)
        except ImportError as err:
            logger.debug("KotlinPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.go import GoPlugin

            if "go" not in cls._unregistered_languages:
                cls.register_language(GoPlugin(), priority=50)
        except ImportError as err:
            logger.debug("GoPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.rust import RustPlugin

            if "rust" not in cls._unregistered_languages:
                cls.register_language(RustPlugin(), priority=50)
        except ImportError as err:
            logger.debug("RustPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.sql_dbt import SqlDbtPlugin

            if "sql" not in cls._unregistered_languages:
                cls.register_language(SqlDbtPlugin(), priority=50)
        except ImportError as err:
            logger.debug("SqlDbtPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.protobuf_grpc import ProtobufGrpcPlugin

            if "protobuf" not in cls._unregistered_languages:
                cls.register_language(ProtobufGrpcPlugin(), priority=50)
        except ImportError as err:
            logger.debug("ProtobufGrpcPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.python_lang import PythonPlugin

            if "python" not in cls._unregistered_languages:
                cls.register_language(PythonPlugin(), priority=50)
        except ImportError as err:
            logger.debug("PythonPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.scip_ingestor import SCIPIndexPlugin

            if "scip" not in cls._unregistered_languages:
                cls.register_language(SCIPIndexPlugin(), priority=60)
        except ImportError as err:
            logger.debug("SCIPIndexPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.languages.typescript_lang import TypeScriptPlugin

            if "typescript" not in cls._unregistered_languages:
                cls.register_language(TypeScriptPlugin(), priority=70)
        except ImportError as err:
            logger.debug("TypeScriptPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.connectors.confluence import ConfluenceConnectorPlugin

            cls.register_connector(
                ConfluenceConnectorPlugin(base_url="https://confluence.example.com")
            )
        except ImportError as err:
            logger.debug("ConfluenceConnectorPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.connectors.jira import JiraConnectorPlugin

            cls.register_connector(JiraConnectorPlugin(base_url="https://jira.example.com"))
        except ImportError as err:
            logger.debug("JiraConnectorPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin

            cls.register_connector(GitLabConnectorPlugin(base_url="https://gitlab.example.com"))
        except ImportError as err:
            logger.debug("GitLabConnectorPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.connectors.github import GitHubEnterpriseConnectorPlugin

            cls.register_connector(
                GitHubEnterpriseConnectorPlugin(base_url="https://api.github.com")
            )
        except ImportError as err:
            logger.debug("GitHubEnterpriseConnectorPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.domains.enterprise_core import EnterpriseCoreDomainPlugin

            cls.register_domain(EnterpriseCoreDomainPlugin())
        except ImportError as err:
            logger.debug("EnterpriseCoreDomainPlugin not loaded: %s", err)

        try:
            from context_scout.plugins.domains.data_engineering import DataEngineeringDomainPlugin

            cls.register_domain(DataEngineeringDomainPlugin())
        except ImportError as err:
            logger.debug("DataEngineeringDomainPlugin not loaded: %s", err)
