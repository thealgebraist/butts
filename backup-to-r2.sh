#!/bin/bash

# Cloudflare R2 Backup Script
# This script backs up the waste image dataset to Cloudflare R2

set -e

# Configuration
R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-}"
R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-}"
R2_BUCKET_NAME="${R2_BUCKET_NAME:-butts-backup}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Check credentials
if [ -z "$R2_ACCOUNT_ID" ] || [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "Error: Missing Cloudflare R2 credentials"
    echo "Please set the following environment variables:"
    echo "  - R2_ACCOUNT_ID"
    echo "  - R2_ACCESS_KEY_ID"
    echo "  - R2_SECRET_ACCESS_KEY"
    echo "  - R2_BUCKET_NAME (optional, defaults to 'butts-backup')"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Cloudflare R2 backup...${NC}"
echo "Endpoint: $R2_ENDPOINT"
echo "Bucket: $R2_BUCKET_NAME"

# Configure AWS CLI for R2
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

# Backup datasets
echo -e "\n${YELLOW}Backing up datasets...${NC}"
aws s3 sync datasets/ \
    "s3://${R2_BUCKET_NAME}/datasets/" \
    --endpoint-url "$R2_ENDPOINT" \
    --no-progress \
    --exclude ".DS_Store" \
    --exclude "*.git*"

# Backup analysis
echo -e "\n${YELLOW}Backing up analysis...${NC}"
aws s3 sync analysis/ \
    "s3://${R2_BUCKET_NAME}/analysis/" \
    --endpoint-url "$R2_ENDPOINT" \
    --no-progress \
    --exclude ".DS_Store"

# Backup reference materials
echo -e "\n${YELLOW}Backing up reference materials...${NC}"
aws s3 sync reference/ \
    "s3://${R2_BUCKET_NAME}/reference/" \
    --endpoint-url "$R2_ENDPOINT" \
    --no-progress \
    --exclude ".DS_Store"

# Backup tools
echo -e "\n${YELLOW}Backing up tools...${NC}"
aws s3 sync tools/ \
    "s3://${R2_BUCKET_NAME}/tools/" \
    --endpoint-url "$R2_ENDPOINT" \
    --no-progress \
    --exclude ".DS_Store" \
    --exclude ".git*"

# Backup configuration files
echo -e "\n${YELLOW}Backing up configuration files...${NC}"
aws s3 cp README.md "s3://${R2_BUCKET_NAME}/README.md" --endpoint-url "$R2_ENDPOINT"
aws s3 cp .gitignore "s3://${R2_BUCKET_NAME}/.gitignore" --endpoint-url "$R2_ENDPOINT"
aws s3 cp @instructions.md "s3://${R2_BUCKET_NAME}/@instructions.md" --endpoint-url "$R2_ENDPOINT" 2>/dev/null || true

echo -e "\n${GREEN}✓ Backup to Cloudflare R2 completed successfully!${NC}"
echo "Backup location: s3://${R2_BUCKET_NAME}/"
