#pragma once
#ifdef _WIN32

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <string>
#include <iostream>

#pragma comment(lib, "d3dcompiler.lib")

namespace CudaGame {
namespace Rendering {

// Shader compilation utility for D3D12
class ShaderCompiler {
public:
    enum class ShaderType {
        Vertex,
        Pixel,
        Compute
    };
    
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
        }
        
        // Compilation flags
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
