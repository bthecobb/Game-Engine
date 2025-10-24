#pragma once

#include "Core/System.h"
#include <memory>
#include <vector>

namespace CudaGame {
namespace Rendering {

class RenderSystem : public Core::System {
public:
    RenderSystem() = default;
    ~RenderSystem() override = default;

    void Configure() {}
    void LoadShaders() {}
    void Update(float deltaTime) override {}
    void Render() {}
};

} // namespace Rendering
} // namespace CudaGame
