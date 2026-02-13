---
description: Run the Total Testing System (Unit Tests + Visual Verification)
---

# Total Testing System

This workflow executes the complete validation suite for the CudaGame engine, covering both logic verification (Unit Tests) and visual regression (Photo Mode/Test Suite).

## 1. Build Verification
Ensure all test targets are built.
```powershell
cmake --build . --config Release --target DX12UnitTests
cmake --build . --config Release --target Full3DGame_DX12_Debug3
```
// turbo-all

## 2. Unit Tests
Run the logic verification suite.
```powershell
.\Release\Release\DX12UnitTests.exe
```

## 3. Visual Verification (Photo Mode)
Run the engine's built-in test suite to capture image sequences.
```powershell
.\Release\Release\Full3DGame_DX12_Debug3.exe --test-suite all
```

## 4. Result Analysis
- **Unit Output**: Check console for "FAILED" count.
- **Visual Output**: Inspect `Testing/TestResults/` for generated BMP sequences.
