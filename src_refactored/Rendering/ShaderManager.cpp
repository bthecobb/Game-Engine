#ifdef _WIN32
#include "Rendering/ShaderManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <d3d12shader.h>

namespace CudaGame {
namespace Rendering {

ShaderManager::ShaderManager() = default;
ShaderManager::~ShaderManager() { Shutdown(); }

bool ShaderManager::Initialize() {
    if (m_initialized) return true;

    // Create DXC utils
    HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils));
    if (FAILED(hr)) {
        std::cerr << "[ShaderManager] Failed to create DxcUtils" << std::endl;
        return false;
    }

    // Create DXC compiler (DXC is the modern HLSL compiler, supports SM 6.0+)
    hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler));
    if (FAILED(hr)) {
        std::cerr << "[ShaderManager] Failed to create DxcCompiler" << std::endl;
        return false;
    }

    // Create default include handler
    hr = m_dxcUtils->CreateDefaultIncludeHandler(&m_includeHandler);
    if (FAILED(hr)) {
        std::cerr << "[ShaderManager] Failed to create include handler" << std::endl;
        return false;
    }

    m_initialized = true;
    std::cout << "[ShaderManager] Initialized with DXC compiler" << std::endl;
    return true;
}

void ShaderManager::Shutdown() {
    if (!m_initialized) return;

    m_shaderCache.clear();
    m_sourceTracking.clear();
    m_includeDirectories.clear();

    m_includeHandler.Reset();
    m_dxcCompiler.Reset();
    m_dxcUtils.Reset();

    m_initialized = false;
}

bool ShaderManager::CompileFromFile(
    const std::wstring& filePath,
    const std::wstring& entryPoint,
    ShaderStage stage,
    ShaderCompileFlags flags,
    ShaderBytecode& outBytecode,
    std::string& outErrorMsg
) {
    if (!m_initialized) {
        outErrorMsg = "ShaderManager not initialized";
        return false;
    }

    // Check cache first
    std::wstring cacheKey = GenerateCacheKey(filePath, entryPoint, stage, flags);
    const ShaderBytecode* cached = GetCachedShader(cacheKey);
    if (cached) {
        outBytecode = *cached;
        return true;
    }

    // Load file
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        outErrorMsg = "Failed to open shader file: " + std::string(filePath.begin(), filePath.end());
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string sourceStr = buffer.str();
    std::wstring source(sourceStr.begin(), sourceStr.end());

    // Compile
    bool success = CompileFromSource(source, filePath, entryPoint, stage, flags, outBytecode, outErrorMsg);

    // Cache if successful
    if (success) {
        CacheShader(cacheKey, outBytecode);

        // Track for hot-reload if enabled
        if (m_hotReloadEnabled) {
            RegisterShaderForHotReload(cacheKey, filePath);
        }
    }

    return success;
}

bool ShaderManager::CompileFromSource(
    const std::wstring& source,
    const std::wstring& sourceName,
    const std::wstring& entryPoint,
    ShaderStage stage,
    ShaderCompileFlags flags,
    ShaderBytecode& outBytecode,
    std::string& outErrorMsg
) {
    if (!m_initialized) {
        outErrorMsg = "ShaderManager not initialized";
        return false;
    }

    // Get shader profile (vs_6_0, ps_6_0, etc.)
    std::wstring profile = GetShaderProfile(stage);

    // Build compiler arguments
    std::vector<LPCWSTR> arguments = BuildCompilerArguments(entryPoint, profile, flags);

    // Create source blob
    DxcBuffer sourceBuffer = {};
    sourceBuffer.Ptr = source.c_str();
    sourceBuffer.Size = source.size() * sizeof(wchar_t);
    sourceBuffer.Encoding = DXC_CP_UTF16;

    // Compile
    ComPtr<IDxcResult> result;
    HRESULT hr = m_dxcCompiler->Compile(
        &sourceBuffer,
        arguments.data(),
        static_cast<UINT32>(arguments.size()),
        m_includeHandler.Get(),
        IID_PPV_ARGS(&result)
    );

    if (FAILED(hr)) {
        outErrorMsg = "DXC compilation failed with HRESULT: " + std::to_string(hr);
        return false;
    }

    // Check compilation status
    HRESULT compileStatus;
    result->GetStatus(&compileStatus);

    if (FAILED(compileStatus)) {
        // Get error messages
        ComPtr<IDxcBlobUtf8> errors;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        if (errors && errors->GetStringLength() > 0) {
            outErrorMsg = std::string(errors->GetStringPointer());
        } else {
            outErrorMsg = "Shader compilation failed with no error message";
        }
        return false;
    }

    // Get compiled shader object
    ComPtr<IDxcBlob> shaderBlob;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);
    if (!shaderBlob) {
        outErrorMsg = "Failed to retrieve compiled shader blob";
        return false;
    }

    // Copy to output
    outBytecode.data.resize(shaderBlob->GetBufferSize());
    memcpy(outBytecode.data.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());
    outBytecode.stage = stage;
    outBytecode.entryPoint = entryPoint;
    outBytecode.profile = profile;

    return true;
}

const ShaderBytecode* ShaderManager::GetCachedShader(const std::wstring& key) const {
    auto it = m_shaderCache.find(key);
    if (it != m_shaderCache.end()) {
        return &it->second;
    }
    return nullptr;
}

void ShaderManager::CacheShader(const std::wstring& key, const ShaderBytecode& bytecode) {
    m_shaderCache[key] = bytecode;
}

void ShaderManager::ClearCache() {
    m_shaderCache.clear();
}

void ShaderManager::CheckForModifiedShaders() {
    if (!m_hotReloadEnabled) return;

    for (auto& [key, sourceInfo] : m_sourceTracking) {
        if (!std::filesystem::exists(sourceInfo.filePath)) continue;

        auto currentWriteTime = std::filesystem::last_write_time(sourceInfo.filePath);
        if (currentWriteTime != sourceInfo.lastWriteTime) {
            std::wcout << L"[ShaderManager] Hot-reload detected: " << sourceInfo.filePath << std::endl;
            
            // Invalidate cache entry
            m_shaderCache.erase(key);
            
            // Update timestamp
            sourceInfo.lastWriteTime = currentWriteTime;
            
            // Note: Actual recompilation happens when shader is next requested
        }
    }
}

void ShaderManager::RegisterShaderForHotReload(const std::wstring& key, const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) return;

    ShaderSourceInfo info;
    info.filePath = filePath;
    info.lastWriteTime = std::filesystem::last_write_time(filePath);
    
    m_sourceTracking[key] = info;
}

void ShaderManager::AddIncludeDirectory(const std::filesystem::path& dir) {
    m_includeDirectories.push_back(dir);
}

void ShaderManager::ClearIncludeDirectories() {
    m_includeDirectories.clear();
}

std::wstring ShaderManager::GenerateCacheKey(
    const std::wstring& filePath,
    const std::wstring& entryPoint,
    ShaderStage stage,
    ShaderCompileFlags flags
) {
    std::wstringstream ss;
    ss << filePath << L"|" << entryPoint << L"|" << static_cast<int>(stage) << L"|" << static_cast<uint32_t>(flags);
    return ss.str();
}

std::wstring ShaderManager::GetShaderProfile(ShaderStage stage, const wchar_t* shaderModel) const {
    std::wstring prefix;
    switch (stage) {
        case ShaderStage::Vertex:   prefix = L"vs_"; break;
        case ShaderStage::Pixel:    prefix = L"ps_"; break;
        case ShaderStage::Geometry: prefix = L"gs_"; break;
        case ShaderStage::Hull:     prefix = L"hs_"; break;
        case ShaderStage::Domain:   prefix = L"ds_"; break;
        case ShaderStage::Compute:  prefix = L"cs_"; break;
        default:                    prefix = L"vs_"; break;
    }
    return prefix + shaderModel;
}

std::vector<LPCWSTR> ShaderManager::BuildCompilerArguments(
    const std::wstring& entryPoint,
    const std::wstring& profile,
    ShaderCompileFlags flags
) {
    std::vector<LPCWSTR> args;

    // Entry point
    args.push_back(L"-E");
    args.push_back(entryPoint.c_str());

    // Target profile
    args.push_back(L"-T");
    args.push_back(profile.c_str());

    // Compilation flags
    if (HasFlag(flags, ShaderCompileFlags::Debug)) {
        args.push_back(L"-Zi"); // Debug info
    }
    if (HasFlag(flags, ShaderCompileFlags::SkipOptimization)) {
        args.push_back(L"-Od"); // Skip optimization
    } else {
        args.push_back(L"-O3"); // Max optimization (release)
    }
    if (HasFlag(flags, ShaderCompileFlags::WarningsAsErrors)) {
        args.push_back(L"-WX"); // Warnings as errors
    }
    if (HasFlag(flags, ShaderCompileFlags::IEEE)) {
        args.push_back(L"-Gis"); // IEEE strictness
    }

    // Enable SPIR-V codegen if needed (for Vulkan interop)
    // args.push_back(L"-spirv");

    return args;
}

} // namespace Rendering
} // namespace CudaGame
#endif // _WIN32
