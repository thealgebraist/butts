# Cloudflare R2 Backup Setup

This guide will help you set up automated backups of your waste image dataset to Cloudflare R2.

## Prerequisites

1. **Cloudflare Account** with R2 enabled
2. **AWS CLI** installed (`brew install awscli` on macOS)
3. **R2 Credentials** (API token with S3 access)

## Step 1: Create R2 Bucket

1. Log in to your Cloudflare dashboard
2. Go to **R2** → **Buckets**
3. Click **Create bucket**
4. Name: `butts-backup` (or your preferred name)
5. Click **Create bucket**

## Step 2: Generate R2 API Token

1. In Cloudflare dashboard, go to **R2** → **Settings**
2. Under **API Tokens**, click **Create API token**
3. Give it a name like "Dataset Backup"
4. Select **S3 API** as the permission type
5. Choose **Edit** for permissions
6. Select the `butts-backup` bucket (or "All buckets")
7. Click **Create API token**
8. Save these credentials:
   - **Access Key ID**
   - **Secret Access Key**
   - Your **Account ID** (visible in R2 settings)

## Step 3: Configure Environment Variables

### Option A: Temporary (Current Session Only)

```bash
export R2_ACCOUNT_ID="your-account-id"
export R2_ACCESS_KEY_ID="your-access-key"
export R2_SECRET_ACCESS_KEY="your-secret-key"
export R2_BUCKET_NAME="butts-backup"
```

### Option B: Permanent (Recommended)

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Cloudflare R2 Backup Credentials
export R2_ACCOUNT_ID="your-account-id"
export R2_ACCESS_KEY_ID="your-access-key"
export R2_SECRET_ACCESS_KEY="your-secret-key"
export R2_BUCKET_NAME="butts-backup"
```

Then reload your shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

## Step 4: Install AWS CLI (if not already installed)

```bash
brew install awscli
```

## Step 5: Run Backup

```bash
cd /Users/anders/projects/thrash
chmod +x backup-to-r2.sh
./backup-to-r2.sh
```

## Step 6: Automate Backups (Optional)

### Daily Backup Using Cron

1. Open cron editor:
```bash
crontab -e
```

2. Add a daily backup at 2 AM:
```cron
0 2 * * * cd /Users/anders/projects/thrash && ./backup-to-r2.sh >> /tmp/r2-backup.log 2>&1
```

3. Save and exit

### Monitor Backup Logs

```bash
tail -f /tmp/r2-backup.log
```

## Step 7: Restore from R2 (If Needed)

### List Contents in R2

```bash
aws s3 ls s3://butts-backup/ \
  --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  --recursive
```

### Download Specific Dataset

```bash
aws s3 sync \
  "s3://butts-backup/datasets/waste_items/packaging/" \
  "./restored-packaging/" \
  --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
```

### Download Everything

```bash
aws s3 sync \
  "s3://butts-backup/" \
  "./r2-restore/" \
  --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
```

## Pricing & Storage Estimates

**Cloudflare R2 Pricing:**
- **Storage**: $0.015 per GB/month
- **API Operations**: $0.0000036 per 10,000 requests
- **No egress fees** (unlike S3)

**Your Dataset Size:**
- ~3,000+ HEIC images
- ~2,800+ annotation JSON files
- Estimated: 50-100 GB
- **Monthly Cost**: ~$0.75 - $1.50

## Troubleshooting

### "Command not found: aws"
```bash
brew install awscli
```

### "Unable to locate credentials"
Verify environment variables are set:
```bash
echo $R2_ACCOUNT_ID
echo $R2_ACCESS_KEY_ID
```

### "Access Denied"
- Check credentials are correct
- Verify API token has S3 permissions
- Ensure bucket name is correct

### "Slow Upload Speed"
- Normal for large datasets
- R2 has no egress fees, so bandwidth is reasonable
- Consider running during off-peak hours

## Security Best Practices

1. **Never commit credentials** to git
2. Use **environment variables** instead of hardcoding
3. **Rotate API tokens** regularly
4. Consider using **bucket versioning** for data recovery
5. Enable **Cloudflare DDoS protection** for the bucket

## Additional Resources

- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3.html)
- [R2 Pricing Calculator](https://www.cloudflare.com/en-gb/products/r2/pricing/)
