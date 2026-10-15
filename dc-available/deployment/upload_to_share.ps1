# upload_to_share.ps1 — finish ONLY the Extracted_resources/ subtree into the
# available-data share (Artifacts/ and _snapshots/ are already fully uploaded).
# Stages to a SHORT root (C:\au) first so long image paths clear the 260 limit.
$ErrorActionPreference = "Stop"

# --- target share ----------------------------------------------------------
$rg    = "MC_AVAILABLE2.0_available_germanywestcentral"
$acct  = "fbe89e4305c0f46d18dddb1"
$share = "pvc-90f4a392-35ef-476c-a094-11cd3d73cad1"

# --- source: just Extracted_resources -------------------------------------
$repo    = "C:\Users\VAS1UL\Master Thesis\AVAILABLE_2.0\dc-available"
$extSrc  = Join-Path $repo "dc_available\data\Extracted_resources"
$short   = "C:\au"        # short staging root

Write-Host "Fetching storage key..." -ForegroundColor Cyan
$key = az storage account keys list -g $rg -n $acct --query "[0].value" -o tsv
if (-not $key) { throw "could not get storage key" }

# --- 1) stage Extracted_resources at a short path (robocopy = long-path ok) --
Write-Host "Staging Extracted_resources -> $short ..." -ForegroundColor Cyan
robocopy $extSrc $short /E /R:1 /W:1 /MT:16 /NFL /NDL /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }
if (-not (Test-Path "$short\resources")) { throw "guard: $short\resources missing -> abort" }

try {
    # --- 2) upload only into the share's Extracted_resources/ folder --------
    Write-Host "`nUploading Extracted_resources ..." -ForegroundColor Cyan
    az storage file upload-batch --account-name $acct --account-key $key `
      --destination "$share/Extracted_resources" --source $short

    Write-Host "`nExtracted_resources subfolders on the share:" -ForegroundColor Green
    az storage file list --account-name $acct --account-key $key --share-name $share `
      --path "Extracted_resources" --output table
}
finally {
    Write-Host "Cleaning up $short ..."
    Remove-Item $short -Recurse -Force -ErrorAction SilentlyContinue
}
