# Rhythm Arena Combat Enhancement Guide

## Current State
The RhythmArenaDemo currently has:
- Basic combat with Q (attack) and E (kick)
- Enemy health and damage systems
- Rhythm-based damage multipliers
- Weapon switching (1, 2)
- Visual rhythm indicators (Tab to toggle)

## Recommended Enhancements for 3D Combat Depth

### 1. **Dash Mechanic (F key)**
```cpp
// Add to player state
bool isDashing = false;
float dashTimer = 0.0f;
const float DASH_DURATION = 0.25f;
const float DASH_SPEED = 30.0f;
glm::vec3 dashDirection;

// In processInput()
if (keysPressed[GLFW_KEY_F] && !isDashing) {
    // Dash toward nearest enemy or movement direction
    isDashing = true;
    dashTimer = DASH_DURATION;
    dashDirection = /* calculate direction */;
}
```

### 2. **Enemy Targeting System**
```cpp
int targetedEnemyIndex = -1;

void updateTargeting() {
    // Find closest enemy in front of player
    float closestDist = FLT_MAX;
    for (int i = 0; i < enemies.size(); ++i) {
        if (!enemies[i].isDead) {
            glm::vec3 toEnemy = enemies[i].position - player.position;
            float dist = glm::length(toEnemy);
            if (dist < 15.0f) { // Max targeting range
                // Check if in front of player
                float dot = glm::dot(playerForward, normalize(toEnemy));
                if (dot > 0.5f && dist < closestDist) {
                    closestDist = dist;
                    targetedEnemyIndex = i;
                }
            }
        }
    }
}
```

### 3. **Combo System**
```cpp
enum ComboState {
    COMBO_NONE,
    COMBO_DASH,     // F
    COMBO_LIGHT_1,  // F → Q
    COMBO_LIGHT_2,  // F → Q → Q
    COMBO_LIGHT_3,  // F → Q → Q → Q
    COMBO_LAUNCHER, // F → Q → Q → Q → E
    COMBO_AIR       // After launcher, Q in air
};

ComboState currentCombo = COMBO_NONE;
float comboTimer = 0.0f;
int comboCount = 0;
```

### 4. **3D Arena Layout**
- Add elevated platforms at different heights
- Create multi-level combat zones
- Add walls that can be used for wall-running into air combos

### 5. **Visual Enhancements**
- Enemy health bars above their heads
- Targeting reticle on selected enemy
- Combo counter display
- Damage numbers floating up from hit enemies
- Particle effects for hits

### 6. **Controller Support (Future)**
For Xbox controller support, add:
```cpp
#include <Xinput.h>
#pragma comment(lib, "Xinput.lib")

XINPUT_STATE controllerState;
XInputGetState(0, &controllerState);

// Left stick for movement
float leftX = controllerState.Gamepad.sThumbLX / 32767.0f;
float leftY = controllerState.Gamepad.sThumbLY / 32767.0f;

// Buttons
bool attackPressed = (controllerState.Gamepad.wButtons & XINPUT_GAMEPAD_X);
bool dashPressed = (controllerState.Gamepad.wButtons & XINPUT_GAMEPAD_B);
```

## Implementation Priority
1. **Dash mechanic** - Core to combo system
2. **Enemy targeting** - Makes combat feel precise
3. **Basic combo chains** - Adds depth
4. **Visual feedback** - Health bars, damage numbers
5. **3D level design** - Vertical combat spaces
6. **Controller support** - Better feel for action combat

## Testing Approach
1. Test dash distance and timing
2. Verify targeting switches correctly
3. Ensure combos chain properly on rhythm
4. Check damage scaling with combo multipliers
5. Validate enemy AI responds to 3D movement

The key is to make combat feel fluid and rhythmic, where players can dash between enemies, chain attacks on beat, and use the 3D space for dynamic combat encounters.
