#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib> // _wsystem

#pragma comment(lib, "d3dcompiler.lib")

namespace CudaGame {
namespace Rendering {

// Shader compilation utility for D3D12
class ShaderCompiler {
public:
    enum class ShaderType {
        Vertex,
        Pixel,
        Compute,
        Amplification,  // DX12 Ultimate - AS 6.5
        Mesh,           // DX12 Ultimate - MS 6.5
        Pixel_6_5       // DX12 Ultimate - PS 6.5 (Required for Mesh Shader pipeline)
    };
    
    // Helper to find and run DXC
    static Microsoft::WRL::ComPtr<ID3DBlob> CompileWithDXC(
        const std::wstring& filePath,
        const std::string& entryPoint,
        const char* target)
    {
        using Microsoft::WRL::ComPtr;
        namespace fs = std::filesystem;

        // 1. Find DXC
        std::wstring dxcPath;
        const wchar_t* candidates[] = {
            L"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\x64\\dxc.exe",
            L"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.20348.0\\x64\\dxc.exe",
            L"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.19041.0\\x64\\dxc.exe",
            L"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\x64\\dxc.exe"
        };
        
        for (const auto& p : candidates) {
            if (fs::exists(p)) {
                dxcPath = p;
                break;
            }
        }
        
        if (dxcPath.empty()) {
            std::cerr << "[ShaderCompiler] DXC compiler not found in Windows SDK paths" << std::endl;
            return nullptr;
        }
        
        // 2. Prepare command
        // Use PowerShell to avoid cmd.exe quoting hell
        std::wstring outFile = filePath + L".cso";
        fs::path p(filePath);
        std::wstring dir = p.parent_path().wstring();
        
        std::wstring cmd = L"powershell -Command \"& '" + dxcPath + L"' -T " + 
            std::wstring(target, target + strlen(target)) + 
            L" -E " + std::wstring(entryPoint.begin(), entryPoint.end()) +
            L" -I '" + dir + L"' " +
            L" '" + filePath + L"' -Fo '" + outFile + L"'\"";
            
        // 3. Run command
        // Redirect stderr to stdout to capture it
        cmd += L" > dxc_out.txt 2>&1";
        
        std::wcerr << L"[ShaderCompiler] Executing with include path: " << dir << std::endl;  
        std::wcout << L"[ShaderCompiler] Executing: " << cmd << std::endl;
        
        int ret = _wsystem(cmd.c_str());
        
        // Print output
        std::ifstream logFile("dxc_out.txt");
        if (logFile) {
            std::cout << "[ShaderCompiler] DXC Output:\n" << logFile.rdbuf() << std::endl;
            logFile.close();
        }
        
        if (ret != 0) {
            std::cerr << "[ShaderCompiler] DXC compilation failed with code " << ret << std::endl;
            return nullptr;
        }
        
        // 4. Read result
        std::ifstream binFile(outFile, std::ios::binary | std::ios::ate);
        if (!binFile) {
            std::cerr << "[ShaderCompiler] Failed to open output CSO file" << std::endl;
            return nullptr;
        }
        
        size_t size = binFile.tellg();
        binFile.seekg(0, std::ios::beg);
        
        std::cout << "[ShaderCompiler] CSO size: " << size << " bytes" << std::endl;
        
        if (size == 0) {
            std::cerr << "[ShaderCompiler] Error: Compiled shader is empty" << std::endl;
            return nullptr;
        }

        ComPtr<ID3DBlob> blob;
        HRESULT hr = D3DCreateBlob(size, &blob);
        if (FAILED(hr)) return nullptr;
        
        binFile.read((char*)blob->GetBufferPointer(), size);
        
        return blob;
    }

    // Compile HLSL shader from file
    static Microsoft::WRL::ComPtr<ID3DBlob> CompileFromFile(
        const std::wstring& filePath,
        const std::string& entryPoint,
        ShaderType type,
        bool enableDebug = false)
    {
        using Microsoft::WRL::ComPtr;
        
        // Determine shader model based on type
        const char* target = nullptr;
        bool useDxc = false;
        
        switch (type) {
            case ShaderType::Vertex:
                target = "vs_5_1";
                break;
            case ShaderType::Pixel:
                target = "ps_5_1";
                break;
            case ShaderType::Compute:
                target = "cs_5_1";
                break;
            case ShaderType::Amplification:
                target = "as_6_5";  // Shader Model 6.5
                useDxc = true;
                break;
            case ShaderType::Mesh:
                target = "ms_6_5";  // Shader Model 6.5
                useDxc = true;
                break;
            case ShaderType::Pixel_6_5:
                target = "ps_6_5";  // Shader Model 6.5
                useDxc = true;
                break;
        }
        
        if (useDxc) {
            return CompileWithDXC(filePath, entryPoint, target);
        }
        
        // Compilation flags (D3DCompiler)
        UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
        if (enableDebug) {
            compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
        } else {
            compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
        }
        
        // Compile shader
        ComPtr<ID3DBlob> shaderBlob;
        ComPtr<ID3DBlob> errorBlob;
        
        HRESULT hr = D3DCompileFromFile(
            filePath.c_str(),
            nullptr,                    // Defines
            D3D_COMPILE_STANDARD_FILE_INCLUDE,  // Include handler
            entryPoint.c_str(),
            target,
            compileFlags,
            0,
            &shaderBlob,
            &errorBlob
        );
        
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "[ShaderCompiler] Compilation failed for " << entryPoint << ":\n";
                std::cerr << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
            } else {
                std::cerr << "[ShaderCompiler] Failed to compile " << entryPoint 
                         << " (HRESULT: 0x" << std::hex << hr << ")" << std::endl;
            }
            return nullptr;
        }
        
        std::wcout << L"[ShaderCompiler] Compiled: " << filePath 
                   << L" (" << entryPoint.c_str() << L") - " 
                   << shaderBlob->GetBufferSize() << L" bytes" << std::endl;
        
        return shaderBlob;
    }
};

} // namespace Rendering
} // namespace CudaGame

#endif // _WIN32
