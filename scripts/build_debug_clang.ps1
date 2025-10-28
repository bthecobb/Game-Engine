# Build DiagnosticsSystem with clang++ for debugging
# This script creates a debug object file with full debug symbols

Write-Host "Building DiagnosticsSystem with clang++ for debugging..." -ForegroundColor Green

# Ensure debug_build directory exists
if (-not (Test-Path "debug_build")) {
    New-Item -ItemType Directory -Path "debug_build" | Out-Null
    Write-Host "Created debug_build directory" -ForegroundColor Yellow
}

# Clean previous debug build
$objectFile = "debug_build/DiagnosticsSystem.o"
if (Test-Path $objectFile) {
    Remove-Item $objectFile
    Write-Host "Removed previous object file" -ForegroundColor Yellow
}

# Clang++ debug compile command
$clangCmd = @(
    "clang++",
    "-std=c++17",
    "-g",                      # Generate debug symbols
    "-O0",                     # No optimization for debugging
    "-fno-omit-frame-pointer", # Keep frame pointer for better stack traces
    "-fstandalone-debug",      # Full debug info in object files
    "-Wall -Wextra -Wpedantic", # Enable comprehensive warnings
    "-Wunused-variable",
    "-Wunused-parameter", 
    "-Wshadow",
    "-Wnull-dereference",
    "-Wdouble-promotion",
    "-I include_refactored",   # Include paths
    "-I glm",
    "-DDEBUG_BUILD=1",         # Define debug macros
    "-DDIAGNOSTICS_ENABLED=1",
    "-c",                      # Compile only, don't link
    "src/Debug/DiagnosticsSystem.cpp",
    "-o", $objectFile
) -join " "

Write-Host "Executing: $clangCmd" -ForegroundColor Cyan

try {
    # Execute clang++ compile
    $result = Invoke-Expression $clangCmd 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: DiagnosticsSystem compiled with clang++" -ForegroundColor Green
        
        # Show file info
        if (Test-Path $objectFile) {
            $fileInfo = Get-Item $objectFile
            $sizeKB = [math]::Round($fileInfo.Length / 1024, 1)
            Write-Host "Object file: $($fileInfo.Name) (${sizeKB} KB)" -ForegroundColor Cyan
            Write-Host "Debug symbols included: YES" -ForegroundColor Green
            Write-Host "Optimization level: O0 (none)" -ForegroundColor Green
        }
        
        Write-Host ""
        Write-Host "Debug build complete! To use with debuggers:" -ForegroundColor Yellow
        Write-Host "  - GDB: gdb your_executable" -ForegroundColor White
        Write-Host "  - LLDB: lldb your_executable" -ForegroundColor White
        Write-Host "  - Visual Studio: Load project and attach debugger" -ForegroundColor White
        
    } else {
        Write-Host "ERROR: Compilation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        if ($result) {
            Write-Host "Output: $result" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "ERROR: Failed to execute clang++ command" -ForegroundColor Red
    Write-Host "Exception: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Note: This creates only the DiagnosticsSystem object file." -ForegroundColor Yellow
Write-Host "For full debugging, you would need to build the entire project with clang++." -ForegroundColor Yellow
