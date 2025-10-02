# CI Setup Verification Script
Write-Host "🔍 Verifying CI/CD Setup..." -ForegroundColor Cyan

# Function to check command availability
function Test-Command {
    param ($Command)
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "✅ $Command is available" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $Command not found" -ForegroundColor Red
        return $false
    }
}

# Function to check file existence
function Test-ConfigFile {
    param ($Path, $Description)
    if (Test-Path $Path) {
        Write-Host "✅ $Description found" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $Description missing" -ForegroundColor Red
        return $false
    }
}

# Function to verify GitHub secrets
function Test-GitHubSecrets {
    $repoRoot = git rev-parse --show-toplevel
    $githubDir = Join-Path $repoRoot ".github"
    $workflowsDir = Join-Path $githubDir "workflows"
    
    if (Test-Path $workflowsDir) {
        $workflows = Get-ChildItem $workflowsDir -Filter "*.yml"
        foreach ($workflow in $workflows) {
            $content = Get-Content $workflow.FullName -Raw
            if ($content -match '\$\{\{\s*secrets\.SONAR_TOKEN\s*\}\}' -and 
                $content -match '\$\{\{\s*secrets\.INTEGRATION_PAT\s*\}\}') {
                Write-Host "✅ GitHub Secrets properly referenced in $($workflow.Name)" -ForegroundColor Green
                return $true
            }
        }
    }
    Write-Host "❌ GitHub Secrets not properly configured" -ForegroundColor Red
    return $false
}

Write-Host "`n1️⃣ Checking Required Tools..." -ForegroundColor Yellow

$tools = @(
    "git",
    "cmake",
    "nvcc",
    "cl",
    "java"
)

$toolsAvailable = $true
foreach ($tool in $tools) {
    if (-not (Test-Command $tool)) {
        $toolsAvailable = $false
    }
}

Write-Host "`n2️⃣ Verifying Configuration Files..." -ForegroundColor Yellow

$configFiles = @{
    "CI Bridge Config" = Join-Path $PWD "ci-bridge.yml"
    "SonarCloud Config" = Join-Path $PWD "sonar-project.properties"
    "CUDA Rules" = Join-Path $PWD ".sonarcloud\cuda-rules.json"
    "Main CI Workflow" = Join-Path $PWD ".github\workflows\ci.yml"
    "Rendering Tests Workflow" = Join-Path $PWD ".github\workflows\rendering-tests.yml"
}

$configsPresent = $true
foreach ($config in $configFiles.GetEnumerator()) {
    if (-not (Test-ConfigFile $config.Value $config.Key)) {
        $configsPresent = $false
    }
}

Write-Host "`n3️⃣ Checking GitHub Configuration..." -ForegroundColor Yellow

# Verify git repository
if (Test-Path ".git") {
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
    $gitConfigured = $true
} else {
    Write-Host "❌ Git repository not initialized" -ForegroundColor Red
    $gitConfigured = $false
}

# Check GitHub secrets configuration
$secretsConfigured = Test-GitHubSecrets

Write-Host "`n4️⃣ Verifying Build System..." -ForegroundColor Yellow

# Check CMake configuration
if (Test-Path "CMakeLists.txt") {
    try {
        $cmakeOutput = cmake --version 2>&1
        Write-Host "✅ CMake configuration present" -ForegroundColor Green
        $cmakeConfigured = $true
    } catch {
        Write-Host "❌ CMake configuration issue" -ForegroundColor Red
        $cmakeConfigured = $false
    }
} else {
    Write-Host "❌ CMakeLists.txt not found" -ForegroundColor Red
    $cmakeConfigured = $false
}

Write-Host "`n5️⃣ Testing SonarCloud Integration..." -ForegroundColor Yellow

# Verify SonarCloud configuration
if (Test-Path "sonar-project.properties") {
    $sonarConfig = Get-Content "sonar-project.properties" -Raw
    if ($sonarConfig -match "sonar.projectKey" -and 
        $sonarConfig -match "sonar.organization") {
        Write-Host "✅ SonarCloud configuration valid" -ForegroundColor Green
        $sonarConfigured = $true
    } else {
        Write-Host "❌ SonarCloud configuration incomplete" -ForegroundColor Red
        $sonarConfigured = $false
    }
} else {
    Write-Host "❌ SonarCloud configuration missing" -ForegroundColor Red
    $sonarConfigured = $false
}

# Summary
Write-Host "`n📋 Setup Verification Summary:" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host "Required Tools: $(if($toolsAvailable){'✅'}else{'❌'})"
Write-Host "Configuration Files: $(if($configsPresent){'✅'}else{'❌'})"
Write-Host "Git Configuration: $(if($gitConfigured){'✅'}else{'❌'})"
Write-Host "GitHub Secrets: $(if($secretsConfigured){'✅'}else{'❌'})"
Write-Host "CMake Setup: $(if($cmakeConfigured){'✅'}else{'❌'})"
Write-Host "SonarCloud Integration: $(if($sonarConfigured){'✅'}else{'❌'})"
Write-Host "----------------------------------------"

# Overall status
$success = $toolsAvailable -and $configsPresent -and $gitConfigured -and 
          $secretsConfigured -and $cmakeConfigured -and $sonarConfigured

if ($success) {
    Write-Host "`n✅ CI/CD Setup Verification PASSED" -ForegroundColor Green
    Write-Host "You can now push changes to trigger the CI pipeline"
} else {
    Write-Host "`n❌ CI/CD Setup Verification FAILED" -ForegroundColor Red
    Write-Host "Please fix the issues marked with ❌ above"
}