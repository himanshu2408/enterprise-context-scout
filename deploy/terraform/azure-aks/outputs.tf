output "aks_cluster_name" {
  description = "Private AKS cluster name."
  value       = azurerm_kubernetes_cluster.primary.name
}

output "aks_oidc_issuer_url" {
  description = "AKS OIDC issuer URL for Workload Identity."
  value       = azurerm_kubernetes_cluster.primary.oidc_issuer_url
}

output "scout_managed_identity_client_id" {
  description = "Workload Identity User-Assigned Managed Identity client ID."
  value       = azurerm_user_assigned_identity.scout_app.client_id
}

output "postgres_fqdn" {
  description = "Azure PostgreSQL Flexible Server PG17 FQDN."
  value       = azurerm_postgresql_flexible_server.postgres17.fqdn
}

output "redis_hostname" {
  description = "Azure Managed Redis hostname."
  value       = azurerm_managed_redis.cache.hostname
}

output "storage_account_name" {
  description = "CMK-encrypted Azure Storage Account name."
  value       = azurerm_storage_account.artifacts.name
}

output "acr_login_server" {
  description = "Private Azure Container Registry login server."
  value       = azurerm_container_registry.containers.login_server
}
