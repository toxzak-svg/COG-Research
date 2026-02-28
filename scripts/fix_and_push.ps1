# Script to remove large files from git history and push to remote
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git History Cleanup and Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a git repo
if (-not (Test-Path .git)) {
    Write-Host "ERROR: Not in a git repository!" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Removing large files from git history..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
Write-Host ""

# Remove large files from history
$filterCommand = @"
git rm --cached --ignore-unmatch data/Traffic/traffic.txt data/Electricity/ECL.csv results/image_vae_aesthetics/vae_best_epoch_1.pt results/image_vae_aesthetics/vae_final.pt
"@

git filter-branch --force --index-filter $filterCommand --prune-empty --tag-name-filter cat -- --all

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: filter-branch failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Cleaning up backup refs..." -ForegroundColor Yellow
git for-each-ref --format="%(refname)" refs/original/ | ForEach-Object { 
    git update-ref -d $_
}

Write-Host ""
Write-Host "Step 3: Running aggressive garbage collection..." -ForegroundColor Yellow
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host ""
Write-Host "Step 4: Verifying large files are removed..." -ForegroundColor Yellow
$largeFiles = git rev-list --objects --all | 
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    Where-Object { $_ -match '^blob' } |
    ForEach-Object { 
        $parts = $_ -split '\s+', 4
        [PSCustomObject]@{
            Hash = $parts[1]
            Size = [long]$parts[2]
            Path = if ($parts.Length -gt 3) { $parts[3] } else { "" }
        }
    } |
    Where-Object { $_.Size -gt 100MB } |
    Select-Object -First 10

if ($largeFiles) {
    Write-Host ""
    Write-Host "WARNING: Large files still found in history:" -ForegroundColor Red
    $largeFiles | Format-Table -AutoSize
    Write-Host ""
    Write-Host "You may need to identify and remove these files as well." -ForegroundColor Yellow
} else {
    Write-Host "SUCCESS: No files over 100MB found in history!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 5: Force pushing to origin master..." -ForegroundColor Yellow
Write-Host "This will overwrite the remote history!" -ForegroundColor Red
Write-Host ""

$response = Read-Host "Do you want to proceed with force push? (yes/no)"
if ($response -ne "yes") {
    Write-Host "Push cancelled. You can manually push later with: git push origin master --force" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Pushing to remote..." -ForegroundColor Yellow
git push origin master --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! Repository pushed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ERROR: Push failed! Check the error messages above." -ForegroundColor Red
    exit 1
}
