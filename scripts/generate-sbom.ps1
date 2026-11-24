[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDirectory = "release"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoRoot $OutputDirectory
$syft = Get-Command syft -ErrorAction SilentlyContinue

if ($null -eq $syft) {
    throw "Syft is required to generate an SBOM. Install it from https://github.com/anchore/syft."
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$sbomPath = Join-Path $outputPath "sypra-$Version.spdx.json"
& $syft.Source "dir:$repoRoot" --output "spdx-json=$sbomPath"
if ($LASTEXITCODE -ne 0) {
    throw "SBOM generation failed."
}

Write-Host "Created $sbomPath"
