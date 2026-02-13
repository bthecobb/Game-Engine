#pragma once
#ifdef _WIN32
#include <d3d12.h>

namespace CudaGame {
namespace Rendering {

// Helper for aligning PSO subobjects
// Must be used to layout the stream struct correctly
template <typename T, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE TType>
struct alignas(void*) StreamSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type = TType;
    T Desc;
};

}
}
#endif
