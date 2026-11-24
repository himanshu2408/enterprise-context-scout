variable "project_id" {
  description = "GCP project ID hosting Enterprise Context Scout."
  type        = string
}

variable "region" {
  description = "GCP region (defaults to EU Frankfurt for GDPR data residency)."
  type        = string
  default     = "europe-west3"
}

variable "cluster_name" {
  description = "Private GKE cluster name."
  type        = string
  default     = "context-scout-gke"
}

variable "environment" {
  description = "Deployment environment label (dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "Primary CIDR block for the private GKE subnet."
  type        = string
  default     = "10.20.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary CIDR block for GKE pods."
  type        = string
  default     = "10.24.0.0/14"
}

variable "services_cidr" {
  description = "Secondary CIDR block for GKE services."
  type        = string
  default     = "10.28.0.0/20"
}

variable "master_ipv4_cidr_block" {
  description = "Private /28 CIDR block for the GKE control plane."
  type        = string
  default     = "172.16.0.0/28"
}

variable "authorized_cidr_blocks" {
  description = "Authorized corporate/VPN CIDR blocks for private GKE control plane access."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = [
    {
      cidr_block   = "10.20.0.0/20"
      display_name = "vpc-internal"
    }
  ]
}

variable "kms_rotation_period" {
  description = "CMEK automatic rotation period (90 days)."
  type        = string
  default     = "7776000s"
}

variable "db_tier" {
  description = "Cloud SQL PostgreSQL 17 machine tier."
  type        = string
  default     = "db-custom-4-16384"
}

variable "redis_memory_size_gb" {
  description = "Memorystore for Redis HA capacity in GiB."
  type        = number
  default     = 5
}

variable "enable_l4_pool" {
  description = "Enable NVIDIA L4 GPU node pool for embedding & guardrail workloads."
  type        = bool
  default     = true
}

variable "enable_a100_pool" {
  description = "Enable NVIDIA A100 GPU node pool for medium open-weights inference & LoRA."
  type        = bool
  default     = false
}

variable "enable_h100_pool" {
  description = "Enable NVIDIA H100 GPU node pool for large MoE inference & specialization."
  type        = bool
  default     = false
}

variable "iap_enabled" {
  description = "Enable Cloud Identity-Aware Proxy (IAP) on the API backend service."
  type        = bool
  default     = true
}

variable "iap_oauth2_client_id" {
  description = "OAuth2 client ID for Cloud IAP (inject from Secret Manager or CI tfvars; never hardcode)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "iap_oauth2_client_secret" {
  description = "OAuth2 client secret for Cloud IAP (inject from Secret Manager or CI tfvars; never hardcode)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "iap_members" {
  description = "IAM principals permitted through Cloud IAP for the Workbench & API."
  type        = list(string)
  default     = []
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for Workload Identity binding."
  type        = string
  default     = "context-scout"
}

variable "enable_vertex_ai" {
  description = "Grant roles/aiplatform.user to API and worker GSAs for keyless Vertex AI execution."
  type        = bool
  default     = true
}
