# Third-Person Orbit Camera Proposal

## Why I Agree with Gemini

Gemini correctly identified that the flickering artifacts and erratic camera movement in `EnhancedGame` were not caused by depth-buffer or V-Sync settings, but by two conflicting camera-control systems fighting over the camera’s transform:

1. **Mouse-based rotation** in the `mouse_callback`, which updates camera orientation each frame.
2. **Frame-by-frame “LookAt”** logic in the main game loop, which recalculates and overrides orientation to point at the player.

These two systems form a feedback loop:
- `Rotate()` adjusts the camera’s forward vector based on mouse input.
- `LookAt()` immediately resets orientation to target the player, negating the mouse input.
- Smoothing attempts (`glm::mix`) then fight both instantaneous changes.

By unifying control under a single “orbit” camera model—where the mouse only controls rotation, and the main loop only calculates position based on that rotation—we eliminate the conflict and produce stable, intuitive third-person movement.

## Proposed Implementation

Below is the code snippet to replace the old follow logic in `EnhancedGameMain_Full3D.cpp`:

```cpp
// --- Before (conflicting systems) ---
// Update camera to follow player
auto& playerT = coordinator.GetComponent<Rendering::TransformComponent>(player);
glm::vec3 cameraOffset(0.0f, 8.0f, 15.0f);
glm::vec3 desiredCameraPos = playerT.position + cameraOffset;

glm::vec3 curPos = mainCamera->GetPosition();
glm::vec3 nextPos = glm::mix(curPos, desiredCameraPos, 5.0f * deltaTime);
mainCamera->SetPosition(nextPos);

mainCamera->LookAt(playerT.position + glm::vec3(0.0f, 2.0f, 0.0f));

// --- After (stable orbit camera) ---
// Mouse callback now only handles yaw/pitch:
//    mainCamera->Rotate(xOffset, yOffset, 0.0f);

// Get player transform
auto& playerT = coordinator.GetComponent<Rendering::TransformComponent>(player);

// Compute orbit offset from camera orientation:
glm::vec3 back = mainCamera->GetForward() * -1.0f;  // inverse forward
float distance = 20.0f;                              // radius
glm::vec3 cameraOffset = back * distance;
cameraOffset.y += 10.0f;                             // vertical lift

// Desired position
glm::vec3 desiredPos = playerT.position + cameraOffset;

// Smooth transition
glm::vec3 curPos2 = mainCamera->GetPosition();
glm::vec3 nextPos2 = glm::mix(curPos2, desiredPos, 10.0f * deltaTime);
mainCamera->SetPosition(nextPos2);

// Keep focus just above the player’s head
mainCamera->LookAt(playerT.position + glm::vec3(0.0f, 2.0f, 0.0f));
```

### Explanation of Key Steps

- **Rotation vs. Position Separation**: `Rotate()` (in the mouse callback) only changes orientation. The main loop never directly modifies orientation.
- **Orbit Vector Calculation**: We derive a position offset (`back * distance`) from the camera’s current forward vector, ensuring the camera orbits around the player based on mouse input.
- **Smoothing**: A higher interpolation factor (`10.0f * deltaTime`) produces snappy yet smooth movement, preventing sudden jumps.
- **Single LookAt**: After positioning, we call `LookAt` once to target the player’s head. No other orientation changes occur.

## Expected Outcome

- The flickering and artifacts will disappear, since the camera never clips inside geometry or jumps unexpectedly.
- The camera will smoothly orbit around the player based solely on mouse input.
- Positioning will follow the player at a stable radius, preventing feedback-loop instability.

Once applied, this change yields a standard, industry-accepted third-person camera behavior, matching best practices found in engines like Unity and Unreal.
