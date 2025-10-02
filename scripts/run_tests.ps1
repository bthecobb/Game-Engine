# Advanced Test Runner Script for CudaGame Engine
param(
    [switch]$SkipBuild,
    [switch]$GenerateReport,
    [string]$TestFilter = "*",
    [string]$OutputDir = "test_results"
)

# Configuration
$BuildType = "Release"
$BuildDir = "../build"
$TestRunner = "$BuildDir/bin/tests/TestRunner"
$ReportDir = "$OutputDir/reports"
$PerformanceDataDir = "$OutputDir/performance"

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path $ReportDir
New-Item -ItemType Directory -Force -Path $PerformanceDataDir

# Build tests if not skipped
if (-not $SkipBuild) {
    Write-Host "Building tests in $BuildType configuration..."
    
    Push-Location $BuildDir
    cmake --build . --config $BuildType --target TestRunner
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Pop-Location
}

# Function to parse performance data from test output
function Parse-PerformanceData {
    param($OutputFile)
    
    $performanceData = @{}
    $content = Get-Content $OutputFile
    
    foreach ($line in $content) {
        if ($line -match "\[PERFORMANCE\].*") {
            $parts = $line -split ": "
            $testName = $parts[1]
            $metrics = $parts[2]
            $performanceData[$testName] = $metrics
        }
    }
    
    return $performanceData
}

# Run tests with detailed output
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$testOutput = "$ReportDir/test_output_$timestamp.txt"
$xmlReport = "$ReportDir/test_results_$timestamp.xml"
$jsonReport = "$ReportDir/test_results_$timestamp.json"

Write-Host "Running tests..."
& $TestRunner --gtest_filter=$TestFilter `
              --gtest_output="xml:$xmlReport" `
              | Tee-Object -FilePath $testOutput

# Parse test results
$testResults = [xml](Get-Content $xmlReport)
$totalTests = $testResults.testsuites.tests
$failures = $testResults.testsuites.failures
$errors = $testResults.testsuites.errors
$time = $testResults.testsuites.time

# Generate performance report
if ($GenerateReport) {
    Write-Host "Generating performance report..."
    
    $performanceData = Parse-PerformanceData $testOutput
    
    # Create JSON report
    $report = @{
        timestamp = (Get-Date).ToString("o")
        summary = @{
            totalTests = $totalTests
            failures = $failures
            errors = $errors
            time = $time
        }
        performance = $performanceData
    }
    
    $report | ConvertTo-Json -Depth 10 | Set-Content $jsonReport
    
    # Generate HTML report
    $htmlReport = "$ReportDir/test_report_$timestamp.html"
    $htmlContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>CudaGame Engine Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .success { color: green; }
        .failure { color: red; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>CudaGame Engine Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: $totalTests</p>
        <p>Failures: <span class="$(if($failures -eq 0){'success'}else{'failure'})">$failures</span></p>
        <p>Errors: <span class="$(if($errors -eq 0){'success'}else{'failure'})">$errors</span></p>
        <p>Total Time: $time seconds</p>
    </div>
    
    <h2>Performance Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Metrics</th>
        </tr>
"@

    foreach ($kvp in $performanceData.GetEnumerator()) {
        $htmlContent += @"
        <tr>
            <td>$($kvp.Key)</td>
            <td>$($kvp.Value)</td>
        </tr>
"@
    }

    $htmlContent += @"
    </table>
</body>
</html>
"@

    $htmlContent | Set-Content $htmlReport
    
    Write-Host "Reports generated:"
    Write-Host "  XML Report: $xmlReport"
    Write-Host "  JSON Report: $jsonReport"
    Write-Host "  HTML Report: $htmlReport"
}

# Final status
if ($failures -eq 0 -and $errors -eq 0) {
    Write-Host "All tests passed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Tests completed with $failures failures and $errors errors." -ForegroundColor Red
    exit 1
}