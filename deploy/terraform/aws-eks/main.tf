data "aws_caller_identity" "current" {}

locals {
  common_tags = {
    App       = "enterprise-context-scout"
    ManagedBy = "terraform"
  }
}

# -----------------------------------------------------------------------------
# 1. Customer-Managed KMS Key
# -----------------------------------------------------------------------------

resource "aws_kms_key" "cmek" {
  description             = "CMEK for Enterprise Context Scout EKS, RDS PG17, ElastiCache, S3, and ECR."
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = local.common_tags
}

# -----------------------------------------------------------------------------
# 2. Private EKS Cluster, OIDC Provider & IRSA (Zero IAM Access Keys)
# -----------------------------------------------------------------------------

resource "aws_iam_role" "eks_cluster" {
  name = "${var.cluster_name}-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_eks_cluster" "primary" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.31"

  vpc_config {
    subnet_ids              = var.private_subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = false
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.cmek.arn
    }
    resources = ["secrets"]
  }

  tags       = local.common_tags
  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

resource "aws_iam_openid_connect_provider" "eks_oidc" {
  url             = aws_eks_cluster.primary.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["9e99a48a9960b14926bb7f3b02e22da2b0ab7280"]
  tags            = local.common_tags
}

resource "aws_iam_role" "scout_irsa" {
  name = "${var.cluster_name}-irsa-api-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks_oidc.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:${var.k8s_namespace}:context-scout-api"
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role" "scout_worker_irsa" {
  name = "${var.cluster_name}-irsa-worker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks_oidc.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:${var.k8s_namespace}:context-scout-worker"
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role" "scout_vllm_irsa" {
  name = "${var.cluster_name}-irsa-vllm-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks_oidc.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:${var.k8s_namespace}:context-scout-vllm"
          "${replace(aws_eks_cluster.primary.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# 3. EKS CPU & GPU Node Groups
# -----------------------------------------------------------------------------

resource "aws_iam_role" "eks_nodes" {
  name = "${var.cluster_name}-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "node_worker" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "node_cni" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "node_ecr" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_eks_node_group" "cpu_pool" {
  cluster_name    = aws_eks_cluster.primary.name
  node_group_name = "cpu-control-plane"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["m6i.2xlarge"]

  scaling_config {
    desired_size = 2
    min_size     = 1
    max_size     = 6
  }

  tags = local.common_tags
}

resource "aws_eks_node_group" "gpu_pool" {
  cluster_name    = aws_eks_cluster.primary.name
  node_group_name = "gpu-vllm-pool"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = var.private_subnet_ids
  ami_type        = "AL2023_x86_64_NVIDIA"
  instance_types  = var.gpu_instance_types

  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = 4
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# 4. RDS PostgreSQL 17 (rds.force_ssl, rds.iam_authentication, KMS)
# -----------------------------------------------------------------------------

resource "aws_db_subnet_group" "pg17" {
  name       = "${var.cluster_name}-pg17-subnets"
  subnet_ids = var.private_subnet_ids
  tags       = local.common_tags
}

resource "aws_db_parameter_group" "pg17_hardened" {
  name   = "${var.cluster_name}-pg17-params"
  family = "postgres17"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name  = "rds.iam_authentication"
    value = "1"
  }

  tags = local.common_tags
}

resource "aws_db_instance" "postgres17" {
  identifier                          = "${var.cluster_name}-pg17"
  engine                              = "postgres"
  engine_version                      = "17.2"
  instance_class                      = var.db_instance_class
  allocated_storage                   = 200
  storage_type                        = "gp3"
  storage_encrypted                   = true
  kms_key_id                          = aws_kms_key.cmek.arn
  db_name                             = "context_scout"
  username                            = "scout_rds_admin"
  manage_master_user_password         = true
  iam_database_authentication_enabled = true
  db_subnet_group_name                = aws_db_subnet_group.pg17.name
  parameter_group_name                = aws_db_parameter_group.pg17_hardened.name
  publicly_accessible                 = false
  multi_az                            = true
  backup_retention_period             = 30
  deletion_protection                 = true
  tags                                = local.common_tags
}

# -----------------------------------------------------------------------------
# 5. ElastiCache Redis (TLS + At-Rest KMS Encryption)
# -----------------------------------------------------------------------------

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.cluster_name}-redis-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "TLS and KMS-encrypted Redis HA for Context Scout."
  node_type                  = var.redis_node_type
  num_cache_clusters         = 2
  engine                     = "redis"
  engine_version             = "7.1"
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
  kms_key_id                 = aws_kms_key.cmek.arn
  automatic_failover_enabled = true
  tags                       = local.common_tags
}

# -----------------------------------------------------------------------------
# 6. KMS-Encrypted S3 Bucket + IRSA Policy + Immutable ECR Repository
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.cluster_name}-${data.aws_caller_identity.current.account_id}-artifacts"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.cmek.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_role_policy" "irsa_s3_rds" {
  name = "${var.cluster_name}-irsa-s3-rds"
  role = aws_iam_role.scout_irsa.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["rds-db:connect"]
        Resource = "arn:aws:rds-db:${var.region}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_db_instance.postgres17.resource_id}/scout_app_user"
      }
    ]
  })
}

resource "aws_iam_role_policy" "irsa_worker_s3_rds" {
  name = "${var.cluster_name}-irsa-worker-s3-rds"
  role = aws_iam_role.scout_worker_irsa.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["rds-db:connect"]
        Resource = "arn:aws:rds-db:${var.region}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_db_instance.postgres17.resource_id}/scout_app_user"
      }
    ]
  })
}

resource "aws_iam_role_policy" "irsa_vllm_s3_rds" {
  name = "${var.cluster_name}-irsa-vllm-s3-rds"
  role = aws_iam_role.scout_vllm_irsa.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["rds-db:connect"]
        Resource = "arn:aws:rds-db:${var.region}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_db_instance.postgres17.resource_id}/scout_app_user"
      }
    ]
  })
}

resource "aws_ecr_repository" "containers" {
  name                 = "${var.cluster_name}-containers"
  image_tag_mutability = "IMMUTABLE"

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.cmek.arn
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}
