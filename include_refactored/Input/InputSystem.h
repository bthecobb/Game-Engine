#pragma once

#include "Core/System.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <functional>

namespace CudaGame {
namespace Input {

enum class InputAction {
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    JUMP,
    ATTACK,
    BLOCK,
    SPECIAL,
    PAUSE
};

class InputSystem : public Core::System {
public:
    InputSystem() = default;
    
    bool Initialize() override { return true; }
    void Update(float deltaTime) override;
    
    // Input state queries
    bool IsKeyPressed(int key) const;
    bool IsKeyHeld(int key) const;
    bool IsMouseButtonPressed(int button) const;
    glm::vec2 GetMousePosition() const;
    glm::vec2 GetMouseDelta() const;
    
    // Action mapping
    void MapKeyToAction(int key, InputAction action);
    bool IsActionPressed(InputAction action) const;
    bool IsActionHeld(InputAction action) const;
    
    // Callbacks
    void SetKeyCallback(GLFWwindow* window);
    void SetMouseCallback(GLFWwindow* window);
    
private:
    std::unordered_map<int, bool> m_keyStates;
    std::unordered_map<int, bool> m_previousKeyStates;
    std::unordered_map<int, InputAction> m_keyToAction;
    
    glm::vec2 m_mousePosition{0.0f};
    glm::vec2 m_previousMousePosition{0.0f};
    glm::vec2 m_mouseDelta{0.0f};
    
    std::unordered_map<int, bool> m_mouseButtonStates;
};

} // namespace Input
} // namespace CudaGame
