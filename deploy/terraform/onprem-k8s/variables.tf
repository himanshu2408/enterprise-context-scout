variable "namespace" {
  description = "Target Kubernetes namespace for Enterprise Context Scout."
  type        = string
  default     = "context-scout"
}

variable "storage_provisioner" {
  description = "CSI provisioner for local NVMe storage (e.g. rancher.io/local-path or topolvm.io)."
  type        = string
  default     = "topolvm.io"
}

variable "postgres_instances" {
  description = "Number of CloudNativePG PostgreSQL 17 instances."
  type        = number
  default     = 3
}

variable "postgres_storage_size" {
  description = "PVC size for PostgreSQL 17 data volume."
  type        = string
  default     = "200Gi"
}

variable "s3_endpoint" {
  description = "On-prem MinIO / Ceph RGW S3 endpoint URL."
  type        = string
  default     = "https://minio.context-scout.svc.cluster.local:9000"
}

variable "s3_bucket_name" {
  description = "On-prem S3-compatible artifact & PITR WAL bucket name."
  type        = string
  default     = "context-scout-artifacts"
}

variable "redis_replicas" {
  description = "Number of Redis HA replicas."
  type        = number
  default     = 3
}
