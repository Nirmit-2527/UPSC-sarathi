# =============================================================================
# upsc-sarathi-agentic :: infrastructure/main.tf
#
# Provisions:
#   - S3 bucket for PDF storage
#   - ECR repository for the Docker image
#   - OpenSearch Serverless collection (vector engine) for embeddings
#   - DynamoDB table for spaced-revision progress
#   - IAM role for ECS tasks (Bedrock, S3, OpenSearch, Textract, DDB access)
#   - ALB + 2 target groups (Streamlit UI on default route, FastAPI on /api/*)
#   - ECS Fargate cluster + task + service (single container, both ports)
#   - CloudWatch log group
#
# Structurally identical to the non-agentic reference project's
# infrastructure (same AWS resource shapes work fine for CrewAI agents -
# only the application code changed, not what it needs from AWS), with one
# simplification: this project has a single DynamoDB table (progress) since
# the agentic project doesn't track separate "tests" state server-side.
# =============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws",    version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" {
  region = var.region
}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
variable "project"   { type = string  default = "upsc-sarathi-agentic" }
variable "region"    { type = string  default = "us-east-1" }
variable "image_tag" { type = string  default = "latest" }
variable "fargate_cpu"    { type = number  default = 1024 }   # 1 vCPU
variable "fargate_memory" { type = number  default = 2048 }   # 2 GB

data "aws_caller_identity" "me" {}
data "aws_vpc"             "default" { default = true }
data "aws_subnets"         "default" {
  filter { name = "vpc-id" values = [data.aws_vpc.default.id] }
}

resource "random_id" "suffix" { byte_length = 3 }
locals {
  name = "${var.project}-${random_id.suffix.hex}"
  tags = { Project = var.project, ManagedBy = "Terraform" }
}

# -----------------------------------------------------------------------------
# 1. S3 bucket (raw + processed PDFs)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "docs" {
  bucket        = "${local.name}-docs"
  force_destroy = true
  tags          = local.tags
}
resource "aws_s3_bucket_versioning" "docs" {
  bucket = aws_s3_bucket.docs.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}

# -----------------------------------------------------------------------------
# 2. ECR
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

# -----------------------------------------------------------------------------
# 3. OpenSearch Serverless (vector engine)
# -----------------------------------------------------------------------------
resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${local.name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules        = [{ Resource = ["collection/${local.name}-vdb"], ResourceType = "collection" }],
    AWSOwnedKey  = true
  })
}
resource "aws_opensearchserverless_security_policy" "network" {
  name = "${local.name}-net"
  type = "network"
  policy = jsonencode([{
    Rules = [
      { Resource = ["collection/${local.name}-vdb"], ResourceType = "collection" },
      { Resource = ["collection/${local.name}-vdb"], ResourceType = "dashboard" }
    ]
    AllowFromPublic = true
  }])
}
resource "aws_opensearchserverless_access_policy" "data" {
  name = "${local.name}-access"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        Resource     = ["collection/${local.name}-vdb"]
        Permission   = ["aoss:CreateCollectionItems", "aoss:DeleteCollectionItems",
                        "aoss:UpdateCollectionItems", "aoss:DescribeCollectionItems"]
        ResourceType = "collection"
      },
      {
        Resource     = ["index/${local.name}-vdb/*"]
        Permission   = ["aoss:*"]
        ResourceType = "index"
      }
    ]
    Principal = [
      data.aws_caller_identity.me.arn,
      aws_iam_role.task.arn
    ]
  }])
}
resource "aws_opensearchserverless_collection" "vdb" {
  name = "${local.name}-vdb"
  type = "VECTORSEARCH"
  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network
  ]
  tags = local.tags
}

# -----------------------------------------------------------------------------
# 4. DynamoDB (spaced-revision progress only)
# -----------------------------------------------------------------------------
resource "aws_dynamodb_table" "progress" {
  name         = "${local.name}-progress"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "card_id"
  attribute { name = "card_id" type = "S" }
  tags = local.tags
}

# -----------------------------------------------------------------------------
# 5. IAM role for ECS tasks
# -----------------------------------------------------------------------------
data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals { type = "Service" identifiers = ["ecs-tasks.amazonaws.com"] }
  }
}
resource "aws_iam_role" "task" {
  name               = "${local.name}-task-role"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
  tags               = local.tags
}
resource "aws_iam_role" "execution" {
  name               = "${local.name}-exec-role"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}
resource "aws_iam_role_policy_attachment" "exec_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
data "aws_iam_policy_document" "task_policy" {
  statement {
    sid     = "Bedrock"
    # Wildcard resource so both the "smart" tier (Claude 3.5 Sonnet) and the
    # "fast" tier (Claude 3.5 Haiku) work without separately enumerating
    # model ARNs - just remember to enable model access for BOTH in the
    # Bedrock console (see DEPLOY_AWS.md step 1).
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = ["*"]
  }
  statement {
    sid       = "S3"
    actions   = ["s3:*Object", "s3:ListBucket"]
    resources = [aws_s3_bucket.docs.arn, "${aws_s3_bucket.docs.arn}/*"]
  }
  statement {
    sid     = "OpenSearch"
    actions = ["aoss:APIAccessAll", "aoss:DashboardsAccessAll"]
    resources = ["*"]
  }
  statement {
    sid     = "Textract"
    actions = [
      "textract:StartDocumentTextDetection",
      "textract:GetDocumentTextDetection",
      "textract:DetectDocumentText"
    ]
    resources = ["*"]
  }
  statement {
    sid     = "DynamoDB"
    actions = ["dynamodb:*"]
    resources = [aws_dynamodb_table.progress.arn]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
}
resource "aws_iam_role_policy" "task_inline" {
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_policy.json
}

# -----------------------------------------------------------------------------
# 6. CloudWatch logs
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = 14
}

# -----------------------------------------------------------------------------
# 7. ALB + Security group
# -----------------------------------------------------------------------------
resource "aws_security_group" "alb" {
  name   = "${local.name}-alb-sg"
  vpc_id = data.aws_vpc.default.id
  ingress { from_port = 80  to_port = 80  protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  egress  { from_port = 0   to_port = 0   protocol = "-1"  cidr_blocks = ["0.0.0.0/0"] }
}
resource "aws_security_group" "svc" {
  name   = "${local.name}-svc-sg"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port       = 8000  to_port = 8000  protocol = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    from_port       = 8501  to_port = 8501  protocol = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress { from_port = 0 to_port = 0 protocol = "-1" cidr_blocks = ["0.0.0.0/0"] }
}
resource "aws_lb" "this" {
  name               = "${local.name}-alb"
  load_balancer_type = "application"
  subnets            = data.aws_subnets.default.ids
  security_groups    = [aws_security_group.alb.id]
}
resource "aws_lb_target_group" "ui" {
  name        = "${local.name}-ui-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  health_check { path = "/" matcher = "200-399" interval = 30 timeout = 10 }
}
resource "aws_lb_target_group" "api" {
  name        = "${local.name}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  health_check { path = "/api/v1/health" matcher = "200" interval = 30 timeout = 10 }
}
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10
  action { type = "forward" target_group_arn = aws_lb_target_group.api.arn }
  condition { path_pattern { values = ["/api/*", "/docs", "/openapi.json", "/redoc"] } }
}

# -----------------------------------------------------------------------------
# 8. ECS cluster + task + service
# -----------------------------------------------------------------------------
resource "aws_ecs_cluster" "this" {
  name = "${local.name}-cluster"
  tags = local.tags
}

resource "aws_ecs_task_definition" "this" {
  family                   = local.name
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true
      portMappings = [
        { containerPort = 8000 hostPort = 8000 protocol = "tcp" },
        { containerPort = 8501 hostPort = 8501 protocol = "tcp" }
      ]
      environment = [
        { name = "APP_MODE",                      value = "aws" },
        { name = "AWS_REGION",                    value = var.region },
        { name = "S3_BUCKET_NAME",                value = aws_s3_bucket.docs.bucket },
        { name = "VECTOR_STORE",                  value = "opensearch" },
        { name = "OPENSEARCH_ENDPOINT",           value = aws_opensearchserverless_collection.vdb.collection_endpoint },
        { name = "OPENSEARCH_INDEX",              value = "${local.name}-index" },
        { name = "LLM_PROVIDER",                  value = "bedrock" },
        { name = "BEDROCK_LLM_MODEL",              value = "anthropic.claude-3-5-sonnet-20241022-v2:0" },
        { name = "BEDROCK_LLM_MODEL_FAST",         value = "anthropic.claude-3-5-haiku-20241022-v1:0" },
        { name = "EMBEDDINGS_PROVIDER",           value = "bedrock" },
        { name = "BEDROCK_EMBED_MODEL",            value = "amazon.titan-embed-text-v2:0" },
        { name = "EMBEDDINGS_DIM",                 value = "1024" },
        { name = "USE_TEXTRACT",                  value = "true" },
        { name = "ENABLE_PROGRESS_TRACKING",      value = "true" },
        { name = "DDB_TABLE_PROGRESS",            value = aws_dynamodb_table.progress.name },
        { name = "AGENTS_VERBOSE",                value = "false" },
        { name = "GUARDRAIL_MAX_RETRIES",         value = "4" },
        { name = "ENABLE_PYQ_MATCHING_IN_QA",     value = "true" },
        { name = "ENABLE_CURRENT_AFFAIRS_DIGEST", value = "true" },
        { name = "ENABLE_GROUNDEDNESS_CHECK",     value = "true" },
        { name = "API_BASE",                      value = "http://localhost:8000/api/v1" }
      ]
      command = ["bash", "-lc",
        "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "app"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "this" {
  name            = "${local.name}-svc"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.svc.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "app"
    container_port   = 8501
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "app"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.http]
}

# -----------------------------------------------------------------------------
# 9. Outputs
# -----------------------------------------------------------------------------
output "ecr_repo_url"          { value = aws_ecr_repository.app.repository_url }
output "s3_bucket"             { value = aws_s3_bucket.docs.bucket }
output "opensearch_endpoint"   { value = aws_opensearchserverless_collection.vdb.collection_endpoint }
output "ecs_cluster_name"      { value = aws_ecs_cluster.this.name }
output "ecs_service_name"      { value = aws_ecs_service.this.name }
output "alb_dns_name"          { value = aws_lb.this.dns_name }
output "log_group"             { value = aws_cloudwatch_log_group.app.name }
output "ddb_progress_table"    { value = aws_dynamodb_table.progress.name }
