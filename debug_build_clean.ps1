# Advanced Debug Build Script for CudaGame Engine
# Uses clang++ with comprehensive debugging and analysis features

$ErrorActionPreference = "Stop"

# Ensure clang++ is in PATH
$env:PATH += ";C:\Program Files\LLVM\bin"

Write-Host "Advanced Debug Build for CudaGame Engine" -ForegroundColor Green

# Define paths
$SrcDir = "src"
$IncludeDir = "include_refactored" 
$BuildDir = "debug_build"
$GlmDir = "glm"

# Create debug build directory
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path $BuildDir | Out-Null

# Clang++ debug flags
$ClangFlags = @(
    "-std=c++17",
    "-g",                          # Debug symbols
    "-O0",                         # No optimization
    "-fno-omit-frame-pointer",     # Keep frame pointers for better stack traces
    "-fstandalone-debug",          # Full debug info
    "-Wall",                       # All warnings
    "-Wextra",                     # Extra warnings  
    "-Wpedantic",                  # Pedantic warnings
    "-Wunused-variable",           # Unused variables
    "-Wunused-parameter",          # Unused parameters
    "-Wshadow",                    # Variable shadowing
    "-Wnull-dereference",          # Null pointer dereference
    "-Wdouble-promotion",          # Float to double promotion
    "-I$IncludeDir",
    "-I$GlmDir",
    "-DDEBUG_BUILD=1",
    "-DDIAGNOSTICS_ENABLED=1"
)

Write-Host "Compiling DiagnosticsSystem with advanced debugging..." -ForegroundColor Yellow

$DiagnosticsCpp = "src\Debug\DiagnosticsSystem.cpp"
$DiagnosticsObj = "$BuildDir\DiagnosticsSystem.o"

try {
    $result = & clang++ @ClangFlags -c $DiagnosticsCpp -o $DiagnosticsObj 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "DiagnosticsSystem compiled successfully!" -ForegroundColor Green
        Write-Host "   Object file: $DiagnosticsObj" -ForegroundColor Cyan
    } else {
        Write-Host "Compilation failed:" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        exit 1
    }
    
    # Show any warnings
    if ($result) {
        Write-Host "Compiler warnings:" -ForegroundColor Yellow
        $result | ForEach-Object { Write-Host "   $_" -ForegroundColor Yellow }
    }
    
} catch {
    Write-Host "Error during compilation: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Debug Analysis Summary:" -ForegroundColor Green
Write-Host "   - Debug symbols: Enabled (-g)" -ForegroundColor Cyan
Write-Host "   - Frame pointers: Preserved (-fno-omit-frame-pointer)" -ForegroundColor Cyan  
Write-Host "   - Full debug info: Enabled (-fstandalone-debug)" -ForegroundColor Cyan
Write-Host "   - Comprehensive warnings: Enabled (-Wall -Wextra -Wpedantic)" -ForegroundColor Cyan
Write-Host "   - No optimizations: Disabled (-O0)" -ForegroundColor Cyan

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "   1. Use 'lldb' or 'gdb' to debug the object file" -ForegroundColor White
Write-Host "   2. Set breakpoints in DiagnosticsSystem functions" -ForegroundColor White  
Write-Host "   3. Inspect variables and memory at runtime" -ForegroundColor White

Write-Host ""
Write-Host "Debug build complete!" -ForegroundColor Green
