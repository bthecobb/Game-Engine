#include "Testing/AdvancedTestFramework.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <gtest/gtest.h>
#include <windows.h>
#include <psapi.h>
#include <memory>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

namespace CudaGame {
namespace Testing {

// Static member initialization
cudaEvent_t CUDAPerformanceMonitor::m_startEvent = nullptr;
cudaEvent_t CUDAPerformanceMonitor::m_stopEvent = nullptr;
bool CUDAPerformanceMonitor::s_initialized = false;
size_t MemoryLeakDetector::m_initialMemoryGPU = 0;
size_t MemoryLeakDetector::m_initialMemoryCPU = 0;

#ifdef NVML_AVAILABLE
nvmlDevice_t CUDAPerformanceMonitor::m_device = nullptr;
#endif

void CUDAPerformanceMonitor::Initialize() {
    if (s_initialized) return;
    
    cudaDeviceSynchronize();
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
    
#ifdef NVML_AVAILABLE
    // Initialize NVML for detailed GPU metrics
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &m_device);
#endif
    
    s_initialized = true;
}

void CUDAPerformanceMonitor::StartRecording() {
    if (!s_initialized) return;
    cudaEventRecord(m_startEvent);
}

void CUDAPerformanceMonitor::StopRecording() {
    if (!s_initialized) return;
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
}

GPUMetrics CUDAPerformanceMonitor::GetGPUMetrics() {
    GPUMetrics metrics{};
    if (!s_initialized) return metrics;
    
#ifdef NVML_AVAILABLE
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(m_device, &utilization);
    metrics.utilization = static_cast<float>(utilization.gpu);
    
    nvmlMemory_t memInfo;
    nvmlDeviceGetMemoryInfo(m_device, &memInfo);
    metrics.memoryUsage = static_cast<float>(memInfo.used);
#else
    // Fallback: approximate utilization and memory via CUDA if available
    metrics.utilization = 0.0f; // Unknown without NVML
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    metrics.memoryUsage = static_cast<float>(total - free);
#endif
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    metrics.computeTime = milliseconds;
    metrics.kernelTime = milliseconds;
    
    return metrics;
}

void CUDAPerformanceMonitor::Cleanup() {
    if (!s_initialized) return;
    
    if (m_startEvent) cudaEventDestroy(m_startEvent);
    if (m_stopEvent) cudaEventDestroy(m_stopEvent);
#ifdef NVML_AVAILABLE
    nvmlShutdown();
    m_device = nullptr;
#endif
    
    m_startEvent = nullptr;
    m_stopEvent = nullptr;
    s_initialized = false;
}


void MemoryLeakDetector::StartTracking() {
    m_initialMemoryGPU = GetGPUMemoryUsage();
    m_initialMemoryCPU = GetCPUMemoryUsage();
}

MemoryDelta MemoryLeakDetector::GetLeaks() {
    MemoryDelta delta;
    delta.gpuBytes = GetGPUMemoryUsage() - m_initialMemoryGPU;
    delta.cpuBytes = GetCPUMemoryUsage() - m_initialMemoryCPU;
    return delta;
}

size_t MemoryLeakDetector::GetGPUMemoryUsage() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

size_t MemoryLeakDetector::GetCPUMemoryUsage() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
}

} // namespace Testing
} // namespace CudaGame
