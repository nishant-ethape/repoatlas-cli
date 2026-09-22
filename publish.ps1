# PowerShell Publish Script for RepoAtlas
param (
    [string]$Token = $env:TWINE_PASSWORD
)

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Installing build tools ===" -ForegroundColor Cyan
python -m pip install build twine -q

Write-Host "`n=== 2. Cleaning old dist files ===" -ForegroundColor Cyan
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

Write-Host "`n=== 3. Building package ===" -ForegroundColor Cyan
python -m build

Write-Host "`n=== 4. Uploading to PyPI ===" -ForegroundColor Cyan
if ($Token) {
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = $Token
    python -m twine upload dist/* --non-interactive
} else {
    python -m twine upload dist/*
}

Write-Host "`n=== Done! Successfully published repoatlas-cli! ===" -ForegroundColor Green
