# Setup script for project dependencies
Write-Host "🔧 Setting up project dependencies..." -ForegroundColor Cyan

# Create directories
$ExternalDir = Join-Path $PSScriptRoot ".." "external"
$ThirdPartyDir = Join-Path $PSScriptRoot ".." "third_party"

# Create directories if they don't exist
@($ExternalDir, $ThirdPartyDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
        Write-Host "✅ Created directory: $_" -ForegroundColor Green
    }
}

# Clone GLM if not present
$GlmPath = Join-Path $ThirdPartyDir "glm"
if (-not (Test-Path (Join-Path $GlmPath "glm/glm.hpp"))) {
    Write-Host "📦 Downloading GLM..." -ForegroundColor Yellow
    Push-Location $ThirdPartyDir
    try {
        & git clone --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git glm
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ GLM downloaded successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to download GLM" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "✅ GLM already present" -ForegroundColor Green
}

# Create symbolic links for convenience
$GlmLink = Join-Path $ExternalDir "glm"
if (-not (Test-Path $GlmLink)) {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        New-Item -ItemType SymbolicLink -Path $GlmLink -Target $GlmPath
    } else {
        # Fallback for older PowerShell versions
        cmd /c mklink /D $GlmLink $GlmPath
    }
    Write-Host "✅ Created GLM symbolic link" -ForegroundColor Green
}

Write-Host "`n✨ Dependencies setup complete!" -ForegroundColor Cyan