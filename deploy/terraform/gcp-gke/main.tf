data "google_project" "current" {
  project_id = var.project_id
}

locals {
  common_labels = {
    app         = "enterprise-context-scout"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# -----------------------------------------------------------------------------
# 1. Private VPC, Subnetwork, and Private Service Connect (PSC)
# -----------------------------------------------------------------------------

resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "gke_subnet" {
  name                     = "${var.cluster_name}-subnet"
  ip_cidr_range            = var.vpc_cidr
  region                   = var.region
  network                  = google_compute_network.vpc.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = var.services_cidr
  }

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_address" "psc_endpoint_ip" {
  name         = "${var.cluster_name}-psc-endpoint-ip"
  region       = var.region
  subnetwork   = google_compute_subnetwork.gke_subnet.id
  address_type = "INTERNAL"
  purpose      = "GCE_ENDPOINT"
}

resource "google_compute_global_address" "private_service_range" {
  name          = "${var.cluster_name}-psa-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_compute_forwarding_rule" "psc_google_apis" {
  name                  = "${var.cluster_name}-psc-apis"
  region                = var.region
  target                = "all-apis"
  network               = google_compute_network.vpc.id
  ip_address            = google_compute_address.psc_endpoint_ip.id
  load_balancing_scheme = ""
}

# -----------------------------------------------------------------------------
# 2. Cloud KMS (CMEK) for GKE, Cloud SQL, Memorystore, GCS, Artifact Registry
# -----------------------------------------------------------------------------

resource "google_kms_key_ring" "scout_ring" {
  name     = "${var.cluster_name}-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "cmek" {
  name            = "${var.cluster_name}-cmek"
  key_ring        = google_kms_key_ring.scout_ring.id
  purpose         = "ENCRYPT_DECRYPT"
  rotation_period = var.kms_rotation_period

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "HSM"
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key_iam_member" "gke_cmek" {
  crypto_key_id = google_kms_crypto_key.cmek.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@container-engine-robot.iam.gserviceaccount.com"
}

resource "google_kms_crypto_key_iam_member" "sql_cmek" {
  crypto_key_id = google_kms_crypto_key.cmek.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-cloud-sql.iam.gserviceaccount.com"
}

resource "google_kms_crypto_key_iam_member" "redis_cmek" {
  crypto_key_id = google_kms_crypto_key.cmek.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@cloud-redis.iam.gserviceaccount.com"
}

resource "google_kms_crypto_key_iam_member" "gcs_cmek" {
  crypto_key_id = google_kms_crypto_key.cmek.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@gs-project-accounts.iam.gserviceaccount.com"
}

resource "google_kms_crypto_key_iam_member" "ar_cmek" {
  crypto_key_id = google_kms_crypto_key.cmek.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-artifactregistry.iam.gserviceaccount.com"
}

# -----------------------------------------------------------------------------
# 3. Dedicated Google Service Accounts & Workload Identity (Zero Static Keys)
# -----------------------------------------------------------------------------

resource "google_service_account" "scout_api" {
  account_id   = "scout-api"
  display_name = "Enterprise Context Scout API Workload SA"
}

resource "google_service_account" "scout_worker" {
  account_id   = "scout-worker"
  display_name = "Enterprise Context Scout Worker Workload SA"
}

resource "google_service_account" "scout_vllm" {
  account_id   = "scout-vllm"
  display_name = "Enterprise Context Scout vLLM & Adapter Sidecar Workload SA"
}

resource "google_service_account" "scout_node" {
  account_id   = "scout-gke-node-sa"
  display_name = "Enterprise Context Scout GKE Node SA"
}

resource "google_service_account_iam_member" "workload_identity_api" {
  service_account_id = google_service_account.scout_api.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/context-scout-api]"
}

resource "google_service_account_iam_member" "workload_identity_worker" {
  service_account_id = google_service_account.scout_worker.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/context-scout-worker]"
}

resource "google_service_account_iam_member" "workload_identity_vllm" {
  service_account_id = google_service_account.scout_vllm.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/context-scout-vllm]"
}

resource "google_project_iam_member" "scout_api_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.scout_api.email}"
}

resource "google_project_iam_member" "scout_api_sql_user" {
  project = var.project_id
  role    = "roles/cloudsql.instanceUser"
  member  = "serviceAccount:${google_service_account.scout_api.email}"
}

resource "google_project_iam_member" "scout_worker_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.scout_worker.email}"
}

resource "google_project_iam_member" "scout_worker_sql_user" {
  project = var.project_id
  role    = "roles/cloudsql.instanceUser"
  member  = "serviceAccount:${google_service_account.scout_worker.email}"
}

resource "google_project_iam_member" "scout_vllm_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.scout_vllm.email}"
}

resource "google_project_iam_member" "scout_vllm_sql_user" {
  project = var.project_id
  role    = "roles/cloudsql.instanceUser"
  member  = "serviceAccount:${google_service_account.scout_vllm.email}"
}

resource "google_project_iam_member" "scout_api_vertex_ai" {
  count   = var.enable_vertex_ai ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.scout_api.email}"
}

resource "google_project_iam_member" "scout_worker_vertex_ai" {
  count   = var.enable_vertex_ai ? 1 : 0
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.scout_worker.email}"
}

# -----------------------------------------------------------------------------
# 4. Private GKE Cluster + CPU / L4 / A100 / H100 Node Pools
# -----------------------------------------------------------------------------

resource "google_container_cluster" "primary" {
  provider = google-beta
  name     = var.cluster_name
  location = var.region

  network                  = google_compute_network.vpc.id
  subnetwork               = google_compute_subnetwork.gke_subnet.id
  remove_default_node_pool = true
  initial_node_count       = 1
  datapath_provider        = "ADVANCED_DATAPATH"
  networking_mode          = "VPC_NATIVE"

  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = true
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.authorized_cidr_blocks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.cmek.id
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
    managed_prometheus {
      enabled = true
    }
  }

  release_channel {
    channel = "REGULAR"
  }

  resource_labels = local.common_labels

  depends_on = [google_kms_crypto_key_iam_member.gke_cmek]
}

resource "google_container_node_pool" "cpu_pool" {
  name     = "cpu-control-plane"
  location = var.region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 1
    max_node_count = 6
  }

  node_config {
    machine_type      = "n2-standard-8"
    service_account   = google_service_account.scout_node.email
    boot_disk_kms_key = google_kms_crypto_key.cmek.id
    oauth_scopes      = ["https://www.googleapis.com/auth/cloud-platform"]
    labels            = merge(local.common_labels, { workload = "control-plane" })

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

resource "google_container_node_pool" "gpu_l4_pool" {
  count    = var.enable_l4_pool ? 1 : 0
  name     = "gpu-l4-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0
    max_node_count = 4
  }

  node_config {
    machine_type      = "g2-standard-24"
    service_account   = google_service_account.scout_node.email
    boot_disk_kms_key = google_kms_crypto_key.cmek.id
    oauth_scopes      = ["https://www.googleapis.com/auth/cloud-platform"]
    labels            = merge(local.common_labels, { gpu_tier = "l4" })

    guest_accelerator {
      type  = "nvidia-l4"
      count = 2
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu_a100_pool" {
  count    = var.enable_a100_pool ? 1 : 0
  name     = "gpu-a100-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0
    max_node_count = 4
  }

  node_config {
    machine_type      = "a2-highgpu-1g"
    service_account   = google_service_account.scout_node.email
    boot_disk_kms_key = google_kms_crypto_key.cmek.id
    oauth_scopes      = ["https://www.googleapis.com/auth/cloud-platform"]
    labels            = merge(local.common_labels, { gpu_tier = "a100" })

    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu_h100_pool" {
  count    = var.enable_h100_pool ? 1 : 0
  name     = "gpu-h100-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = 0
    max_node_count = 2
  }

  node_config {
    machine_type      = "a3-highgpu-8g"
    service_account   = google_service_account.scout_node.email
    boot_disk_kms_key = google_kms_crypto_key.cmek.id
    oauth_scopes      = ["https://www.googleapis.com/auth/cloud-platform"]
    labels            = merge(local.common_labels, { gpu_tier = "h100" })

    guest_accelerator {
      type  = "nvidia-h100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# -----------------------------------------------------------------------------
# 5. Cloud SQL PostgreSQL 17 (PSC, IAM Auth, PITR, CMEK, Passwordless User)
# -----------------------------------------------------------------------------

resource "google_sql_database_instance" "postgres17" {
  name                = "${var.cluster_name}-pg17"
  database_version    = "POSTGRES_17"
  region              = var.region
  encryption_key_name = google_kms_crypto_key.cmek.id
  deletion_protection = true

  settings {
    tier              = var.db_tier
    availability_type = "REGIONAL"
    disk_autoresize   = true
    disk_type         = "PD_SSD"

    ip_configuration {
      ipv4_enabled = false
      ssl_mode     = "ENCRYPTED_ONLY"

      psc_config {
        psc_enabled               = true
        allowed_consumer_projects = [var.project_id]
      }
    }

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    user_labels = local.common_labels
  }

  depends_on = [google_kms_crypto_key_iam_member.sql_cmek]
}

resource "google_sql_database" "scout_db" {
  name     = "context_scout"
  instance = google_sql_database_instance.postgres17.name
}

resource "google_sql_user" "scout_api_db_user" {
  name     = trimsuffix(google_service_account.scout_api.email, ".gserviceaccount.com")
  instance = google_sql_database_instance.postgres17.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

resource "google_sql_user" "scout_worker_db_user" {
  name     = trimsuffix(google_service_account.scout_worker.email, ".gserviceaccount.com")
  instance = google_sql_database_instance.postgres17.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

resource "google_sql_user" "scout_vllm_db_user" {
  name     = trimsuffix(google_service_account.scout_vllm.email, ".gserviceaccount.com")
  instance = google_sql_database_instance.postgres17.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

# -----------------------------------------------------------------------------
# 6. Memorystore for Redis HA (TLS, AUTH, CMEK)
# -----------------------------------------------------------------------------

resource "google_redis_instance" "cache" {
  name                    = "${var.cluster_name}-redis"
  tier                    = "STANDARD_HA"
  memory_size_gb          = var.redis_memory_size_gb
  region                  = var.region
  authorized_network      = google_compute_network.vpc.id
  connect_mode            = "PRIVATE_SERVICE_ACCESS"
  redis_version           = "REDIS_7_2"
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  auth_enabled            = true
  customer_managed_key    = google_kms_crypto_key.cmek.id
  labels                  = local.common_labels

  depends_on = [google_kms_crypto_key_iam_member.redis_cmek]
}

# -----------------------------------------------------------------------------
# 7. CMEK-Encrypted GCS Artifact Bucket & Artifact Registry
# -----------------------------------------------------------------------------

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-${var.cluster_name}-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.cmek.id
  }

  labels = local.common_labels

  depends_on = [google_kms_crypto_key_iam_member.gcs_cmek]
}

resource "google_storage_bucket_iam_member" "scout_api_bucket_rw" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.scout_api.email}"
}

resource "google_storage_bucket_iam_member" "scout_worker_bucket_rw" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.scout_worker.email}"
}

resource "google_storage_bucket_iam_member" "scout_vllm_bucket_ro" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.scout_vllm.email}"
}

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = "${var.cluster_name}-containers"
  description   = "CMEK-encrypted immutable container registry for Enterprise Context Scout."
  format        = "DOCKER"
  kms_key_name  = google_kms_crypto_key.cmek.id

  docker_config {
    immutable_tags = true
  }

  labels = local.common_labels

  depends_on = [google_kms_crypto_key_iam_member.ar_cmek]
}

# -----------------------------------------------------------------------------
# 8. Cloud Armor & Cloud IAP
# -----------------------------------------------------------------------------

resource "google_compute_security_policy" "edge_armor" {
  name        = "${var.cluster_name}-cloud-armor"
  description = "Cloud Armor WAF & rate-limiting policy for Context Scout Ingress."

  rule {
    action   = "deny(403)"
    priority = 1000
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('xss-v33-stable') || evaluatePreconfiguredWaf('sqli-v33-stable')"
      }
    }
    description = "Block OWASP XSS and SQLi probes at edge."
  }

  rule {
    action   = "allow"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow behind Cloud IAP."
  }
}

resource "google_compute_health_check" "api_health" {
  name = "${var.cluster_name}-api-hc"
  http_health_check {
    port         = 8080
    request_path = "/api/v1/health"
  }
}

resource "google_compute_backend_service" "api_backend" {
  name            = "${var.cluster_name}-api-backend"
  protocol        = "HTTP"
  security_policy = google_compute_security_policy.edge_armor.id
  health_checks   = [google_compute_health_check.api_health.id]

  dynamic "iap" {
    for_each = var.iap_enabled && var.iap_oauth2_client_id != "" && var.iap_oauth2_client_secret != "" ? [1] : []
    content {
      enabled              = true
      oauth2_client_id     = var.iap_oauth2_client_id
      oauth2_client_secret = var.iap_oauth2_client_secret
    }
  }
}

resource "google_iap_web_backend_service_iam_binding" "iap_access" {
  count               = length(var.iap_members) > 0 ? 1 : 0
  project             = var.project_id
  web_backend_service = google_compute_backend_service.api_backend.name
  role                = "roles/iap.httpsResourceAccessor"
  members             = var.iap_members
}
