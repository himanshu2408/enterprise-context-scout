output "eks_cluster_name" {
  description = "Private EKS cluster name."
  value       = aws_eks_cluster.primary.name
}

output "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN for IRSA."
  value       = aws_iam_openid_connect_provider.eks_oidc.arn
}

output "scout_irsa_role_arn" {
  description = "Backward-compatible IRSA IAM Role ARN for Context Scout API pods."
  value       = aws_iam_role.scout_irsa.arn
}

output "scout_api_irsa_role_arn" {
  description = "IRSA IAM Role ARN for context-scout-api."
  value       = aws_iam_role.scout_irsa.arn
}

output "scout_worker_irsa_role_arn" {
  description = "IRSA IAM Role ARN for context-scout-worker."
  value       = aws_iam_role.scout_worker_irsa.arn
}

output "scout_vllm_irsa_role_arn" {
  description = "IRSA IAM Role ARN for context-scout-vllm."
  value       = aws_iam_role.scout_vllm_irsa.arn
}

output "rds_endpoint" {
  description = "RDS PostgreSQL 17 endpoint."
  value       = aws_db_instance.postgres17.endpoint
}

output "redis_primary_endpoint" {
  description = "ElastiCache Redis primary TLS endpoint."
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "s3_artifact_bucket" {
  description = "KMS-encrypted S3 artifact bucket."
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  description = "Immutable KMS-encrypted ECR repository URL."
  value       = aws_ecr_repository.containers.repository_url
}
