# Script to remove large files from git history

Write-Host "Removing large files from git history..."

# Remove large files from all commits
git filter-branch --force --index-filter `
"git rm --cached --ignore-unmatch data/Traffic/traffic.txt data/Electricity/ECL.csv results/image_vae_aesthetics/vae_best_epoch_1.pt results/image_vae_aesthetics/vae_final.pt" `
--prune-empty --tag-name-filter cat -- --all

Write-Host "Cleaning up..."
# Clean up the backup refs
git for-each-ref --format="%(refname)" refs/original/ | ForEach-Object { git update-ref -d $_ }

# Garbage collect to remove the old objects
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "Done! You can now push with: git push origin master --force"
