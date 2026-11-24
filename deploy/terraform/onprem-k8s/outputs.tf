output "namespace" {
  description = "Deployed Kubernetes namespace."
  value       = kubernetes_namespace_v1.scout.metadata[0].name
}

output "storage_class_name" {
  description = "Local NVMe StorageClass name."
  value       = kubernetes_storage_class_v1.local_nvme.metadata[0].name
}

output "redis_service_name" {
  description = "Redis HA headless service name."
  value       = kubernetes_service_v1.redis_ha.metadata[0].name
}

output "s3_bucket_name" {
  description = "MinIO / Ceph S3-compatible artifact bucket."
  value       = var.s3_bucket_name
}

output "service_accounts" {
  description = "Runtime Kubernetes ServiceAccounts."
  value = {
    api    = kubernetes_service_account_v1.scout_api.metadata[0].name
    worker = kubernetes_service_account_v1.scout_worker.metadata[0].name
    vllm   = kubernetes_service_account_v1.scout_vllm.metadata[0].name
  }
}
