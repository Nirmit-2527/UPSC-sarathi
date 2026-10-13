#!/usr/bin/env bash
# =============================================================================
# upsc-sarathi-agentic :: scripts/deploy_aws.sh
#
# Provisions and deploys UPSC Sarathi Agentic on AWS:
#   1. Validates prereqs (aws, docker, terraform, jq)
#   2. Creates S3 doc bucket, ECR repo, OpenSearch Serverless collection,
#      IAM roles, ECS Fargate service, ALB - all via Terraform
#   3. Builds and pushes the Docker image to ECR
#   4. Forces a new ECS deployment
#   5. Prints the ALB URL to open in a browser (feature: "must be able to
#      open a browser URL for the Streamlit frontend")
#
# Usage:
#   bash scripts/deploy_aws.sh init       # one-time terraform init
#   bash scripts/deploy_aws.sh plan       # preview infra changes
#   bash scripts/deploy_aws.sh apply      # provision infra
#   bash scripts/deploy_aws.sh build      # build & push image only
#   bash scripts/deploy_aws.sh deploy     # apply + build + push + redeploy (default)
#   bash scripts/deploy_aws.sh destroy    # tear everything down
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLU}[deploy]${NC} $*"; }
ok()   { echo -e "${GRN}[ok]${NC} $*"; }
warn() { echo -e "${YLW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[err]${NC} $*"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- configuration ----------
PROJECT="${PROJECT:-upsc-sarathi-agentic}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TF_DIR="$ROOT/infrastructure"

# ---------- prereqs ----------
prereqs() {
  for cmd in aws docker terraform jq; do
    command -v $cmd >/dev/null 2>&1 || die "$cmd not installed"
  done
  aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials invalid"
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
  ok "AWS account: $ACCOUNT_ID, region: $AWS_REGION"
  export TF_VAR_project="$PROJECT"
  export TF_VAR_region="$AWS_REGION"
  export TF_VAR_image_tag="$IMAGE_TAG"
  ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT"
}

# ---------- enable Bedrock model access (informational) ----------
bedrock_note() {
  warn "Reminder: Bedrock model access is gated. In the AWS console go to"
  warn "  Bedrock -> Model access -> Manage model access"
  warn "and enable BOTH (this project uses a fast + smart tier):"
  warn "  Anthropic Claude 3.5 Sonnet  +  Anthropic Claude 3.5 Haiku  +  Amazon Titan Text Embeddings v2"
}

# ---------- terraform ----------
tf_init()    { (cd "$TF_DIR" && terraform init -upgrade); }
tf_plan()    { (cd "$TF_DIR" && terraform plan -out=tfplan); }
tf_apply()   { (cd "$TF_DIR" && terraform apply -auto-approve); }
tf_destroy() { (cd "$TF_DIR" && terraform destroy -auto-approve); }

# ---------- docker build & push ----------
build_push() {
  log "Logging in to ECR..."
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_URI"

  log "Building docker image (linux/amd64)..."
  docker build --platform linux/amd64 -t "$PROJECT:$IMAGE_TAG" .
  docker tag "$PROJECT:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"

  log "Pushing $ECR_URI:$IMAGE_TAG ..."
  docker push "$ECR_URI:$IMAGE_TAG"
  ok "Image pushed."
}

# ---------- ECS force redeploy ----------
redeploy_ecs() {
  CLUSTER=$(cd "$TF_DIR" && terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
  SERVICE=$(cd "$TF_DIR" && terraform output -raw ecs_service_name 2>/dev/null || echo "")
  [[ -z "$CLUSTER" || -z "$SERVICE" ]] && warn "ECS not yet provisioned; skipping redeploy." && return
  log "Forcing ECS new deployment ($CLUSTER / $SERVICE)..."
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
      --force-new-deployment --region "$AWS_REGION" >/dev/null
  ok "ECS deployment triggered."
}

# ---------- main ----------
prereqs
ACTION="${1:-deploy}"

case "$ACTION" in
  init)    tf_init ;;
  plan)    tf_init && tf_plan ;;
  apply)   tf_init && tf_apply && bedrock_note ;;
  build)   build_push ;;
  deploy)
    tf_init && tf_apply
    build_push
    redeploy_ecs
    bedrock_note
    LB=$(cd "$TF_DIR" && terraform output -raw alb_dns_name 2>/dev/null || echo "")
    if [[ -n "$LB" ]]; then
      ok "App URL: http://$LB"
      ok "API docs: http://$LB/docs"
      log "ECS Fargate tasks take ~1-2 min to become healthy after a fresh deploy - retry the URL if it doesn't load immediately."
      if command -v open >/dev/null 2>&1; then open "http://$LB"; fi          # macOS
      if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://$LB"; fi  # Linux desktop
    fi
    ;;
  destroy) tf_destroy ;;
  *) die "Unknown action: $ACTION (use init|plan|apply|build|deploy|destroy)" ;;
esac
