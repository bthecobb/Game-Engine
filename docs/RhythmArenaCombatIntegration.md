# RhythmArena Combat Integration Guide

## Current Working Features
- ✅ Basic combat (Q = attack, E = kick)
- ✅ Enemy spawning and AI
- ✅ Health and damage systems
- ✅ Rhythm-based damage multipliers
- ✅ Weapon switching (1, 2)
- ✅ Visual rhythm indicators

## Features to Add

### 1. Dash Mechanics (F key)
Add to RhythmArenaDemo.cpp after line 130 in Player struct:
```cpp
// Dash mechanics
bool isDashing = false;
float dashTimer = 0.0f;
const float DASH_DURATION = 0.25f;
const float DASH_SPEED = 30.0f;
glm::vec3 dashDirection;
```

In `processInput()` function, add:
```cpp
// Dash (F key)
if (keysPressed[GLFW_KEY_F] && !player.isDashing) {
    // Find nearest enemy or use movement direction
    glm::vec3 targetDir;
    if (targetedEnemyIndex >= 0) {
        targetDir = enemies[targetedEnemyIndex].position - player.position;
        targetDir.y = 0;
        targetDir = glm::normalize(targetDir);
    } else {
        targetDir = glm::vec3(sin(player.rotation), 0, -cos(player.rotation));
    }
    
    player.isDashing = true;
    player.dashTimer = player.DASH_DURATION;
    player.dashDirection = targetDir;
}
```

### 2. Enemy Targeting System
Add after global variables (around line 220):
```cpp
int targetedEnemyIndex = -1;

void updateTargeting() {
    targetedEnemyIndex = -1;
    if (enemies.empty()) return;
    
    float closestDist = FLT_MAX;
    glm::vec3 playerForward(sin(player.rotation), 0, -cos(player.rotation));
    
    for (int i = 0; i < enemies.size(); ++i) {
        if (enemies[i].isDead) continue;
        
        glm::vec3 toEnemy = enemies[i].position - player.position;
        float dist = glm::length(toEnemy);
        
        if (dist < 15.0f) {
            toEnemy = glm::normalize(toEnemy);
            float dot = glm::dot(playerForward, toEnemy);
            
            if (dot > 0.5f && dist < closestDist) {
                closestDist = dist;
                targetedEnemyIndex = i;
            }
        }
    }
}
```

### 3. Combo System
Add combo states to Player struct:
```cpp
enum ComboState {
    COMBO_NONE,
    COMBO_DASH,
    COMBO_LIGHT_1,
    COMBO_LIGHT_2,
    COMBO_LIGHT_3,
    COMBO_LAUNCHER,
    COMBO_AIR
};
ComboState comboState = COMBO_NONE;
float comboWindow = 0.0f;
```

### 4. Arena Bounds
Add after global variables:
```cpp
struct ArenaBounds {
    glm::vec3 center = glm::vec3(0, 0, 0);
    float radius = 30.0f;
    float wallHeight = 0.6f;
} arena;
```

In `updatePlayer()`, add bounds checking:
```cpp
// Check arena bounds
glm::vec3 toCenter = player.position - arena.center;
toCenter.y = 0;
float dist = glm::length(toCenter);

if (dist > arena.radius) {
    toCenter = glm::normalize(toCenter);
    player.position = arena.center + toCenter * arena.radius;
    
    // Stop outward velocity
    float dot = glm::dot(player.velocity, toCenter);
    if (dot > 0) {
        player.velocity -= toCenter * dot;
    }
}
```

### 5. Weapon Throwing (T key)
Add to Player struct:
```cpp
bool weaponThrown = false;
struct {
    glm::vec3 position;
    glm::vec3 velocity;
    float lifetime;
    bool returning;
} thrownWeapon;
```

In `processInput()`:
```cpp
// Throw weapon (T)
if (keysPressed[GLFW_KEY_T] && !player.weaponThrown) {
    player.weaponThrown = true;
    player.thrownWeapon.position = player.position + glm::vec3(0, 1.5f, 0);
    player.thrownWeapon.velocity = glm::vec3(sin(player.rotation), 0, -cos(player.rotation)) * 25.0f;
    player.thrownWeapon.lifetime = 3.0f;
    player.thrownWeapon.returning = false;
}

// Recall weapon (hold T)
if (keys[GLFW_KEY_T] && player.weaponThrown && !player.thrownWeapon.returning) {
    player.thrownWeapon.returning = true;
}
```

### 6. Visual Enhancements
Add damage numbers:
```cpp
struct DamageNumber {
    glm::vec3 position;
    float value;
    float lifetime;
    glm::vec3 color;
};
std::vector<DamageNumber> damageNumbers;

void spawnDamageNumber(glm::vec3 pos, float damage, glm::vec3 color) {
    DamageNumber dmg;
    dmg.position = pos + glm::vec3(0, 2, 0);
    dmg.value = damage;
    dmg.lifetime = 1.0f;
    dmg.color = color;
    damageNumbers.push_back(dmg);
}
```

### 7. 3D Combat Depth
Add elevated platforms in `setupWorld()`:
```cpp
// Elevated platforms
for (int i = 0; i < 4; ++i) {
    Platform platform;
    float angle = i * 90.0f * M_PI / 180.0f;
    platform.position = glm::vec3(
        cos(angle) * 15.0f,
        3.0f + i * 2.0f,
        sin(angle) * 15.0f
    );
    platform.size = glm::vec3(8.0f, 1.0f, 8.0f);
    platform.color = glm::vec3(0.4f + i * 0.1f, 0.6f, 0.6f);
    platforms.push_back(platform);
}
```

## Implementation Order
1. **Targeting system** - Essential for precise combat
2. **Dash mechanics** - Core movement enhancement
3. **Arena bounds** - Keeps combat contained
4. **Combo states** - Adds depth to combat
5. **Weapon throwing** - Advanced mechanic
6. **Visual feedback** - Polish and game feel
7. **3D platforms** - Vertical gameplay

## Testing Checklist
- [ ] Dash moves player quickly toward target
- [ ] Targeting switches to nearest enemy in front
- [ ] Combos chain properly: Dash → Light×3 → Launcher
- [ ] Arena bounds prevent leaving the arena
- [ ] Thrown weapon damages enemies and returns
- [ ] Damage numbers appear and fade
- [ ] Players can jump between platforms

## Key Bindings Summary
- **F** - Dash attack (starts combos)
- **Q** - Light attack (continues combos)
- **E** - Heavy/Launcher (at 3rd hit)
- **T** - Throw weapon (tap) / Recall (hold)
- **Tab** - Toggle rhythm visualization
- **R** - Reset game

The enhanced combat system will make the game feel more like a character action game (Devil May Cry, Bayonetta) with rhythm elements!
