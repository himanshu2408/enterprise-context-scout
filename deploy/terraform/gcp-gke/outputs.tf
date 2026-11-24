output "gke_cluster_name" {
  description = "Private GKE cluster name."
  value       = google_container_cluster.primary.name
}

output "gke_endpoint" {
  description = "Private GKE control-plane endpoint."
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cmek_key_id" {
  description = "Cloud KMS CMEK key resource ID."
  value       = google_kms_crypto_key.cmek.id
}

output "cloud_sql_connection_name" {
  description = "Cloud SQL PostgreSQL 17 instance connection name."
  value       = google_sql_database_instance.postgres17.connection_name
}

output "cloud_sql_psc_service_attachment" {
  description = "Cloud SQL Private Service Connect attachment link."
  value       = google_sql_database_instance.postgres17.psc_service_attachment_link
}

output "redis_host" {
  description = "Memorystore Redis HA private host."
  value       = google_redis_instance.cache.host
}

output "artifact_bucket_name" {
  description = "CMEK-encrypted GCS artifact bucket name."
  value       = google_storage_bucket.artifacts.name
}

output "artifact_registry_repository" {
  description = "Google Artifact Registry repository ID."
  value       = google_artifact_registry_repository.containers.id
}

output "scout_api_service_account_email" {
  description = "Workload Identity Google Service Account email for context-scout-api."
  value       = google_service_account.scout_api.email
}

output "scout_worker_service_account_email" {
  description = "Workload Identity Google Service Account email for context-scout-worker."
  value       = google_service_account.scout_worker.email
}

output "scout_vllm_service_account_email" {
  description = "Workload Identity Google Service Account email for context-scout-vllm."
  value       = google_service_account.scout_vllm.email
}

output "cloud_armor_security_policy_name" {
  description = "Cloud Armor WAF security policy name."
  value       = google_compute_security_policy.edge_armor.name
}

output "scout_app_service_account_email" {
  description = "Backward-compatible alias for the API Workload Identity GSA email."
  value       = google_service_account.scout_api.email
}
