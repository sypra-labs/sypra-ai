[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$')]
    [string]$Version,
    [string]$OutputDirectory = "release"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoRoot $OutputDirectory
$goCache = Join-Path $repoRoot ".cache\go-build"
$env:GOCACHE = $goCache
$env:CGO_ENABLED = "0"

if (Test-Path $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

Push-Location (Join-Path $repoRoot "web")
try {
    corepack pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "UI dependency installation failed." }
    corepack pnpm run build
    if ($LASTEXITCODE -ne 0) { throw "UI build failed." }
}
finally {
    Pop-Location
}

$targets = @(
    @{ GOOS = "windows"; GOARCH = "amd64"; Extension = ".exe" },
    @{ GOOS = "linux"; GOARCH = "amd64"; Extension = "" },
    @{ GOOS = "darwin"; GOARCH = "arm64"; Extension = "" }
)

foreach ($target in $targets) {
    $name = "sypra-$Version-$($target.GOOS)-$($target.GOARCH)"
    $stage = Join-Path $outputPath $name
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    $binary = Join-Path $stage ("sypra" + $target.Extension)

    $env:GOOS = $target.GOOS
    $env:GOARCH = $target.GOARCH
    go build -trimpath -buildvcs=false -ldflags "-s -w -X main.version=$Version" -o $binary .\cmd\sypra
    if ($LASTEXITCODE -ne 0) { throw "Go build failed for $name." }
    $mockBinary = Join-Path $stage ("sypra-mock" + $target.Extension)
    go build -trimpath -buildvcs=false -ldflags "-s -w" -o $mockBinary .\cmd\sypra-mock
    if ($LASTEXITCODE -ne 0) { throw "Sypra Mock build failed for $name." }

    Copy-Item (Join-Path $repoRoot "README.md") $stage
    Copy-Item (Join-Path $repoRoot "LICENSE") $stage
    Copy-Item (Join-Path $repoRoot ".env.example") $stage
    Copy-Item (Join-Path $repoRoot "web\dist") (Join-Path $stage "web") -Recurse
}

Remove-Item Env:GOOS -ErrorAction SilentlyContinue
Remove-Item Env:GOARCH -ErrorAction SilentlyContinue
Get-ChildItem $outputPath -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -Recurse -File | Get-FileHash -Algorithm SHA256
} | Format-Table -AutoSize

Write-Host "Release staging created at $outputPath. Generate an SBOM before publishing."
