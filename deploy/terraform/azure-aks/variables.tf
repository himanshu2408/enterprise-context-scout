variable "location" {
  description = "Azure region (defaults to Germany West Central for EU data residency)."
  type        = string
  default     = "germanywestcentral"
}

variable "resource_group_name" {
  description = "Target Azure Resource Group name."
  type        = string
  default     = "rg-context-scout"
}

variable "cluster_name" {
  description = "Private AKS cluster name."
  type        = string
  default     = "context-scout-aks"
}

variable "subnet_id" {
  description = "Private VNet subnet ID for AKS nodes and private endpoints."
  type        = string
}

variable "postgres_subnet_id" {
  description = "Delegated private subnet ID for PostgreSQL Flexible Server."
  type        = string
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for Workload Identity federation."
  type        = string
  default     = "context-scout"
}

variable "gpu_vm_size" {
  description = "Azure GPU VM size for vLLM inference & LoRA specialization."
  type        = string
  default     = "Standard_NC24ads_A100_v4"
}

variable "entra_postgres_admin_object_id" {
  description = "Microsoft Entra ID group/principal object ID for PostgreSQL Flexible Server admin."
  type        = string
  default     = "00000000-0000-0000-0000-000000000001"
}

variable "entra_postgres_admin_principal_name" {
  description = "Microsoft Entra ID group/principal display name for PostgreSQL Flexible Server admin."
  type        = string
  default     = "context-scout-pg-admins"
}
