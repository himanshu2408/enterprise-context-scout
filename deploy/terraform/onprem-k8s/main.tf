locals {
  common_labels = {
    "app.kubernetes.io/part-of"    = "enterprise-context-scout"
    "app.kubernetes.io/managed-by" = "terraform"
  }
}

# -----------------------------------------------------------------------------
# 1. Isolated Namespace & Zero-Trust NetworkPolicy
# -----------------------------------------------------------------------------

resource "kubernetes_namespace_v1" "scout" {
  metadata {
    name = var.namespace
    labels = merge(local.common_labels, {
      "pod-security.kubernetes.io/enforce" = "restricted"
    })
  }
}

resource "kubernetes_network_policy_v1" "airgap_egress_guard" {
  metadata {
    name      = "scout-airgap-network-policy"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
  }

  spec {
    pod_selector {}

    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        pod_selector {}
      }
    }

    egress {
      to {
        pod_selector {}
      }
    }

    egress {
      ports {
        port     = "53"
        protocol = "UDP"
      }
      ports {
        port     = "53"
        protocol = "TCP"
      }
    }
  }
}

# -----------------------------------------------------------------------------
# 2. Local NVMe StorageClass & ServiceAccounts
# -----------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "local_nvme" {
  metadata {
    name   = "local-nvme"
    labels = local.common_labels
  }

  storage_provisioner = var.storage_provisioner
  reclaim_policy      = "Retain"
  volume_binding_mode = "WaitForFirstConsumer"
}

resource "kubernetes_service_account_v1" "scout_api" {
  metadata {
    name      = "context-scout-api"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }
  automount_service_account_token = false
}

resource "kubernetes_service_account_v1" "scout_worker" {
  metadata {
    name      = "context-scout-worker"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }
  automount_service_account_token = false
}

resource "kubernetes_service_account_v1" "scout_vllm" {
  metadata {
    name      = "context-scout-vllm"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }
  automount_service_account_token = false
}

# -----------------------------------------------------------------------------
# 3. CloudNativePG PostgreSQL 17 Cluster (pgvector + PITR WAL Archiving)
# -----------------------------------------------------------------------------

resource "kubernetes_manifest" "cnpg_postgres17" {
  manifest = {
    apiVersion = "postgresql.cnpg.io/v1"
    kind       = "Cluster"
    metadata = {
      name      = "context-scout-pg17"
      namespace = kubernetes_namespace_v1.scout.metadata[0].name
      labels    = local.common_labels
    }
    spec = {
      instances = var.postgres_instances
      imageName = "ghcr.io/cloudnative-pg/postgresql:17.2"
      storage = {
        size         = var.postgres_storage_size
        storageClass = kubernetes_storage_class_v1.local_nvme.metadata[0].name
      }
      postgresql = {
        shared_preload_libraries = ["vector"]
        parameters = {
          ssl             = "on"
          max_connections = "200"
          "pgaudit.log"   = "ddl,write"
          "row_security"  = "on"
        }
      }
      bootstrap = {
        initdb = {
          database = "context_scout"
          owner    = "scout_app_user"
          postInitSQL = [
            "CREATE EXTENSION IF NOT EXISTS vector;"
          ]
        }
      }
      backup = {
        retentionPolicy = "30d"
        barmanObjectStore = {
          destinationPath = "s3://${var.s3_bucket_name}/cnpg-wal"
          endpointURL     = var.s3_endpoint
          wal = {
            compression = "gzip"
          }
        }
      }
    }
  }
}

# -----------------------------------------------------------------------------
# 4. Redis HA StatefulSet & Service
# -----------------------------------------------------------------------------

resource "kubernetes_service_v1" "redis_ha" {
  metadata {
    name      = "context-scout-redis"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    cluster_ip = "None"
    selector = {
      app = "context-scout-redis"
    }
    port {
      name = "redis"
      port = 6379
    }
  }
}

resource "kubernetes_stateful_set_v1" "redis_ha" {
  metadata {
    name      = "context-scout-redis"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    service_name = kubernetes_service_v1.redis_ha.metadata[0].name
    replicas     = var.redis_replicas

    selector {
      match_labels = {
        app = "context-scout-redis"
      }
    }

    template {
      metadata {
        labels = {
          app = "context-scout-redis"
        }
      }

      spec {
        container {
          name  = "redis"
          image = "redis:7.4-alpine"
          port {
            container_port = 6379
          }
        }
      }
    }
  }
}

# -----------------------------------------------------------------------------
# 5. MinIO / Ceph S3-Compatible Artifact Bucket Configuration
# -----------------------------------------------------------------------------

resource "kubernetes_service_v1" "minio_s3" {
  metadata {
    name      = "minio"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    selector = {
      app = "context-scout-minio"
    }
    port {
      name = "s3-tls"
      port = 9000
    }
  }
}

resource "kubernetes_config_map_v1" "s3_bucket_config" {
  metadata {
    name      = "scout-s3-bucket-config"
    namespace = kubernetes_namespace_v1.scout.metadata[0].name
    labels    = local.common_labels
  }

  data = {
    S3_ENDPOINT    = var.s3_endpoint
    S3_BUCKET_NAME = var.s3_bucket_name
    S3_PROVIDER    = "minio-ceph-rgw"
  }
}
