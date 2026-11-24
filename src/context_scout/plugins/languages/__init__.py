"""Polyglot Language Plugins (M2-02 / F11-F14)."""

from context_scout.plugins.languages.api_specs import ApiSpecsPlugin
from context_scout.plugins.languages.csharp import CSharpPlugin
from context_scout.plugins.languages.go import GoPlugin
from context_scout.plugins.languages.java_spring import JavaSpringPlugin
from context_scout.plugins.languages.kotlin import KotlinPlugin
from context_scout.plugins.languages.protobuf_grpc import ProtobufGrpcPlugin
from context_scout.plugins.languages.react_redux import ReactReduxPlugin
from context_scout.plugins.languages.rust import RustPlugin
from context_scout.plugins.languages.scip_ingestor import SCIPIndexPlugin
from context_scout.plugins.languages.sql_dbt import SqlDbtPlugin
from context_scout.plugins.languages.storybook import StorybookPlugin
from context_scout.plugins.languages.templates_i18n import TemplatesI18nPlugin
from context_scout.plugins.languages.terraform import TerraformPlugin

__all__ = [
    "ApiSpecsPlugin",
    "CSharpPlugin",
    "GoPlugin",
    "JavaSpringPlugin",
    "KotlinPlugin",
    "ProtobufGrpcPlugin",
    "ReactReduxPlugin",
    "RustPlugin",
    "SCIPIndexPlugin",
    "SqlDbtPlugin",
    "StorybookPlugin",
    "TemplatesI18nPlugin",
    "TerraformPlugin",
]
