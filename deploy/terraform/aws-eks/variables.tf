variable "region" {
  description = "AWS region (defaults to EU Frankfurt for GDPR data residency)."
  type        = string
  default     = "eu-central-1"
}

variable "cluster_name" {
  description = "Private EKS cluster name."
  type        = string
  default     = "context-scout-eks"
}

variable "vpc_id" {
  description = "Target private VPC ID."
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs across >= 2 Availability Zones."
  type        = list(string)
}

variable "k8s_namespace" {
  description = "Kubernetes namespace for IRSA workload binding."
  type        = string
  default     = "context-scout"
}

variable "db_instance_class" {
  description = "RDS PostgreSQL 17 instance class."
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type."
  type        = string
  default     = "cache.r6g.large"
}

variable "gpu_instance_types" {
  description = "EC2 GPU instance types for vLLM inference & LoRA specialization."
  type        = list(string)
  default     = ["g5.2xlarge"]
}
