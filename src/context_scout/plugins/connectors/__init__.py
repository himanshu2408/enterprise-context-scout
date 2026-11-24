"""Enterprise Source Connectors for context scout (F15-F17)."""

from context_scout.plugins.connectors.confluence import ConfluenceConnectorPlugin
from context_scout.plugins.connectors.github import GitHubEnterpriseConnectorPlugin
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.plugins.connectors.jira import JiraConnectorPlugin

__all__ = [
    "ConfluenceConnectorPlugin",
    "GitHubEnterpriseConnectorPlugin",
    "GitLabConnectorPlugin",
    "JiraConnectorPlugin",
]
