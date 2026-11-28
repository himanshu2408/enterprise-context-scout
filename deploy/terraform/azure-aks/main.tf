data "azurerm_client_config" "current" {}

locals {
  name_hash    = substr(md5("${var.resource_group_name}-${var.cluster_name}"), 0, 6)
  clean_prefix = substr(replace(lower(var.cluster_name), "/[^a-z0-9]/", ""), 0, 12)
  kv_name      = substr("${local.clean_prefix}kv${local.name_hash}", 0, 24)
  storage_name = substr("${local.clean_prefix}art${local.name_hash}", 0, 24)
  acr_name     = substr("${local.clean_prefix}acr${local.name_hash}", 0, 50)
  common_tags = {
    app        = "enterprise-context-scout"
    managed_by = "terraform"
  }
}

resource "azurerm_resource_group" "scout" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

# -----------------------------------------------------------------------------
# 1. User-Assigned Managed Identity & Key Vault CMK (Zero Static Keys)
# -----------------------------------------------------------------------------

resource "azurerm_user_assigned_identity" "scout_app" {
  name                = "${var.cluster_name}-app-identity"
  resource_group_name = azurerm_resource_group.scout.name
  location            = azurerm_resource_group.scout.location
  tags                = local.common_tags
}

resource "azurerm_key_vault" "cmk" {
  name                       = local.kv_name
  location                   = azurerm_resource_group.scout.location
  resource_group_name        = azurerm_resource_group.scout.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "premium"
  purge_protection_enabled   = true
  soft_delete_retention_days = 90
  rbac_authorization_enabled = true

  network_acls {
    default_action             = "Deny"
    bypass                     = "AzureServices"
    virtual_network_subnet_ids = [var.subnet_id]
  }

  tags = local.common_tags
}

resource "azurerm_role_assignment" "kv_crypto_officer" {
  scope                = azurerm_key_vault.cmk.id
  role_definition_name = "Key Vault Crypto Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "kv_storage_cmek" {
  scope                = azurerm_key_vault.cmk.id
  role_definition_name = "Key Vault Crypto Service Encryption User"
  principal_id         = azurerm_user_assigned_identity.scout_app.principal_id
}

resource "azurerm_key_vault_key" "cmek" {
  name         = "${var.cluster_name}-cmek"
  key_vault_id = azurerm_key_vault.cmk.id
  key_type     = "RSA-HSM"
  key_size     = 2048
  key_opts     = ["decrypt", "encrypt", "sign", "unwrapKey", "verify", "wrapKey"]

  depends_on = [azurerm_role_assignment.kv_crypto_officer]
}

# -----------------------------------------------------------------------------
# 2. Private AKS Cluster (OIDC + Workload Identity) & GPU Node Pool
# -----------------------------------------------------------------------------

resource "azurerm_kubernetes_cluster" "primary" {
  name                      = var.cluster_name
  location                  = azurerm_resource_group.scout.location
  resource_group_name       = azurerm_resource_group.scout.name
  dns_prefix                = var.cluster_name
  private_cluster_enabled   = true
  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  default_node_pool {
    name           = "cpupool"
    node_count     = 2
    vm_size        = "Standard_D8s_v5"
    vnet_subnet_id = var.subnet_id
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "azure"
    network_policy     = "cilium"
    network_data_plane = "cilium"
  }

  tags = local.common_tags
}

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.primary.id
  vm_size               = var.gpu_vm_size
  vnet_subnet_id        = var.subnet_id
  auto_scaling_enabled  = true
  min_count             = 0
  max_count             = 4
  node_taints           = ["nvidia.com/gpu=present:NoSchedule"]
  tags                  = local.common_tags
}

resource "azurerm_federated_identity_credential" "scout_api" {
  name                = "${var.cluster_name}-api-fed-id"
  resource_group_name = azurerm_resource_group.scout.name
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.primary.oidc_issuer_url
  parent_id           = azurerm_user_assigned_identity.scout_app.id
  subject             = "system:serviceaccount:${var.k8s_namespace}:context-scout-api"
}

resource "azurerm_federated_identity_credential" "scout_worker" {
  name                = "${var.cluster_name}-worker-fed-id"
  resource_group_name = azurerm_resource_group.scout.name
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.primary.oidc_issuer_url
  parent_id           = azurerm_user_assigned_identity.scout_app.id
  subject             = "system:serviceaccount:${var.k8s_namespace}:context-scout-worker"
}

resource "azurerm_federated_identity_credential" "scout_vllm" {
  name                = "${var.cluster_name}-vllm-fed-id"
  resource_group_name = azurerm_resource_group.scout.name
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.primary.oidc_issuer_url
  parent_id           = azurerm_user_assigned_identity.scout_app.id
  subject             = "system:serviceaccount:${var.k8s_namespace}:context-scout-vllm"
}

# -----------------------------------------------------------------------------
# 3. Azure Database for PostgreSQL Flexible Server (PG17, Entra ID, VECTOR)
# -----------------------------------------------------------------------------

resource "azurerm_postgresql_flexible_server" "postgres17" {
  name                          = "${var.cluster_name}-pg17"
  resource_group_name           = azurerm_resource_group.scout.name
  location                      = azurerm_resource_group.scout.location
  version                       = "17"
  delegated_subnet_id           = var.postgres_subnet_id
  public_network_access_enabled = false
  sku_name                      = "GP_Standard_D4s_v3"
  storage_mb                    = 262144
  backup_retention_days         = 30

  authentication {
    active_directory_auth_enabled = true
    password_auth_enabled         = false
    tenant_id                     = data.azurerm_client_config.current.tenant_id
  }

  tags = local.common_tags
}

resource "azurerm_postgresql_flexible_server_configuration" "pgvector" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.postgres17.id
  value     = "VECTOR"
}

resource "azurerm_postgresql_flexible_server_database" "scout_db" {
  name      = "context_scout"
  server_id = azurerm_postgresql_flexible_server.postgres17.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

resource "azurerm_postgresql_flexible_server_active_directory_administrator" "pg_entra_admin" {
  server_name         = azurerm_postgresql_flexible_server.postgres17.name
  resource_group_name = azurerm_resource_group.scout.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = var.entra_postgres_admin_object_id
  principal_name      = var.entra_postgres_admin_principal_name
  principal_type      = "Group"
}

# -----------------------------------------------------------------------------
# 4. Azure Managed Redis (TLS 1.2+)
# -----------------------------------------------------------------------------

resource "azurerm_managed_redis" "cache" {
  name                = "${var.cluster_name}-redis"
  resource_group_name = azurerm_resource_group.scout.name
  location            = azurerm_resource_group.scout.location
  sku_name            = "Balanced_B5"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.scout_app.id]
  }

  default_database {
    access_keys_authentication_enabled = false
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# 5. CMK-Encrypted Storage Account / Blob Container & Private ACR
# -----------------------------------------------------------------------------

resource "azurerm_storage_account" "artifacts" {
  name                          = local.storage_name
  resource_group_name           = azurerm_resource_group.scout.name
  location                      = azurerm_resource_group.scout.location
  account_tier                  = "Standard"
  account_replication_type      = "ZRS"
  shared_access_key_enabled     = false
  public_network_access_enabled = false
  min_tls_version               = "TLS1_2"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.scout_app.id]
  }

  customer_managed_key {
    key_vault_key_id          = azurerm_key_vault_key.cmek.id
    user_assigned_identity_id = azurerm_user_assigned_identity.scout_app.id
  }

  blob_properties {
    versioning_enabled = true
  }

  tags = local.common_tags

  depends_on = [azurerm_role_assignment.kv_storage_cmek]
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "context-scout-artifacts"
  storage_account_id    = azurerm_storage_account.artifacts.id
  container_access_type = "private"
}

resource "azurerm_role_assignment" "scout_blob_contributor" {
  scope                = azurerm_storage_account.artifacts.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.scout_app.principal_id
}

resource "azurerm_container_registry" "containers" {
  name                          = local.acr_name
  resource_group_name           = azurerm_resource_group.scout.name
  location                      = azurerm_resource_group.scout.location
  sku                           = "Premium"
  admin_enabled                 = false
  public_network_access_enabled = false
  tags                          = local.common_tags
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = azurerm_container_registry.containers.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.primary.kubelet_identity[0].object_id
}
