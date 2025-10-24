# Rhythm Arena Aerial Combat Integration Guide

## Overview
This system adds advanced aerial movement and combat mechanics inspired by Devil May Cry and Bayonetta:

1. **Dive after first jump** - Press Space again to dive downward
2. **Dive Bounce** - Landing after dive launches you higher than double jump
3. **Weapon Dive** - Press X while airborne with weapon out for devastating crash attack
4. **Massive particle effects** - Explosions, trails, and impact waves
5. **Air combos** - Juggle enemies in the air with extended combos

## Key Mechanics

### 1. Basic Dive (Space → Space)
```
Jump → Dive → Ground Impact → Bounce (higher jump)
```
- First jump activates aerial state
- Second press initiates dive
- Landing with dive creates bounce effect
- Bounce launches player 25 units high (vs 18 for normal jump)

### 2. Weapon Dive (Space → X with weapon)
```
Jump → Switch Weapon → X → Crash Landing
```
- Devastating head-first dive attack
- Creates massive explosion on impact
- Damages all enemies in 5-unit radius
- Screen shake and particle explosion
- Can be initiated from dive bounce for extra height

### 3. Control Flow
```cpp
// In Player struct, add:
RhythmArenaAerial::AerialCombatSystem aerialCombat;
RhythmArenaAerial::AirComboSystem airCombo;

// In processInput():
// Dive input (second space press while airborne)
if (keysPressed[GLFW_KEY_SPACE] && player.aerialCombat.canDive()) {
    player.aerialCombat.startDive(player.velocity);
}

// Weapon dive (X while airborne)
if (keysPressed[GLFW_KEY_X] && player.aerialCombat.canWeaponDive()) {
    glm::vec3 targetDir = targetedEnemyIndex >= 0 ? 
        enemies[targetedEnemyIndex].position - player.position :
        glm::vec3(0, -1, 0);
    player.aerialCombat.startWeaponDive(player.position, targetDir);
}

// Weapon switch mid-air
if (keys[GLFW_KEY_1] || keys[GLFW_KEY_2]) {
    // Allow weapon switching during any aerial state
}
```

### 4. Physics Integration
```cpp
// In updatePlayer():
// Apply dive velocity
if (player.aerialCombat.isDiving || player.aerialCombat.isWeaponDiving) {
    player.velocity = player.aerialCombat.getDiveVelocity();
}

// Check ground impact
if (player.onGround && player.aerialCombat.aerialState != AERIAL_NONE) {
    bool wasWeaponDiving = player.aerialCombat.isWeaponDiving;
    player.aerialCombat.onGroundImpact(player.position, wasWeaponDiving);
    
    // Apply bounce if from regular dive
    if (player.aerialCombat.aerialState == AERIAL_DIVE_BOUNCE) {
        player.velocity.y = player.aerialCombat.getDiveBounceVelocity();
        player.onGround = false;
    }
    
    // Check impact damage for weapon dive
    if (wasWeaponDiving) {
        for (auto& enemy : enemies) {
            if (player.aerialCombat.checkImpactDamage(enemy.position)) {
                enemy.health -= player.aerialCombat.impactDamage;
                // Launch enemy upward
                enemy.velocity.y = 20.0f;
                spawnDamageNumber(enemy.position, player.aerialCombat.impactDamage, 
                                 glm::vec3(1, 0.5f, 0));
            }
        }
    }
}

// Track aerial state
if (!player.onGround && player.jumpCount == 1) {
    player.aerialCombat.aerialState = AERIAL_FIRST_JUMP;
}

if (player.onGround) {
    player.aerialCombat.resetAerialState();
}
```

### 5. Rendering Integration
```cpp
// In renderScene(), add:
// Apply camera shake
if (player.aerialCombat.screenShake > 0) {
    float shake = player.aerialCombat.screenShake;
    cameraPos += glm::vec3(
        (rand() % 100 - 50) * 0.01f * shake,
        (rand() % 100 - 50) * 0.01f * shake,
        0
    );
}

// Render aerial effects
player.aerialCombat.render(player.position, currentTime);

// Render player with dive rotation
if (player.aerialCombat.isDiving || player.aerialCombat.isWeaponDiving) {
    glPushMatrix();
    glTranslatef(player.position.x, player.position.y, player.position.z);
    glRotatef(player.aerialCombat.diveRotation, 1, 0, 0); // Head-first rotation
    // Render player model
    glPopMatrix();
}
```

## Visual Effects Details

### Dive Trail
- Golden spiral trail for regular dive
- Purple energy trail for weapon dive
- Particles follow player during descent

### Impact Effects
**Regular Dive Landing:**
- Blue energy burst
- 30 particles shooting upward
- Small screen shake (0.3)

**Weapon Dive Impact:**
- Massive explosion with 100+ particles
- Radial shockwave effect
- Ground crack particles
- Full screen shake (1.0)
- White to orange gradient particles

## Combo Integration

### Air Juggle System
```cpp
// When launching enemy with E
if (enemy.velocity.y > 10.0f) {
    player.airCombo.startAirCombo();
}

// During air attacks
if (player.aerialCombat.aerialState != AERIAL_NONE && attackHit) {
    player.airCombo.continueAirCombo();
    damage *= player.airCombo.getAirComboDamageMultiplier();
}
```

## Key Bindings
- **Space** (first) - Jump
- **Space** (second) - Dive
- **X** (airborne) - Weapon dive attack
- **1/2** (anytime) - Switch weapons (even mid-air)
- **Q** (airborne) - Air combo attacks
- **E** (ground) - Launcher to start air combos

## Testing Checklist
- [ ] Jump → Dive → Bounce gives higher jump than double jump
- [ ] Weapon dive creates massive particle explosion
- [ ] Screen shakes on impact
- [ ] Enemies take damage in radius
- [ ] Can switch weapons mid-air before committing to dive
- [ ] Dive rotation looks correct (head-first)
- [ ] Air combos chain properly
- [ ] Particles and effects clean up properly

## Performance Notes
- Limit particle count to ~200 active
- Use simple shapes for particles
- Fade particles with alpha over lifetime
- Remove dead particles each frame

This system creates spectacular aerial combat with risk/reward mechanics - dive for mobility but commit to the landing, or weapon dive for massive damage but leave yourself vulnerable!
