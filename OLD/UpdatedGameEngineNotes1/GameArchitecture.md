# CudaGame Engine Architecture

## Core Systems Overview

### Game Flow
- Main game loop implemented in RhythmArenaDemo and RhythmArenaEnhanced3D
- State machine-based game flow (GAME_PLAYING, GAME_DEATH, GAME_GAME_OVER, GAME_RESTART)
- Enhanced state system in 3D version adds GAME_MENU, GAME_PAUSED, GAME_LEVEL_COMPLETE states

### Combat System
1. Combat States
   - Basic moves: PUNCH, KICK
   - Combo system: COMBO_1, COMBO_2, COMBO_3
   - Weapon-specific moves for SWORD, STAFF, HAMMER
   - Advanced moves: GRAB, PARRY, COUNTER_ATTACK, SLIDE

2. Combo System
   - Progressive combo states (LIGHT_1 -> LIGHT_2 -> LIGHT_3)
   - Special moves: LAUNCHER, AIR, WEAPON_DIVE
   - Combo window timing system
   - Rhythm-based accuracy bonuses

### Character Systems

1. Player
   - Position and physics-based movement
   - Enhanced movement abilities:
     - Wall running
     - Double jumping
     - Dashing
     - Weapon switching
   - Combat state management
   - Health and status effects

2. Enemy AI
   - State-based AI system
   - Detection and aggression mechanics
   - Advanced pathing and targeting
   - Mini-boss system with unique behaviors

### Rendering Pipeline
- OpenGL 3.3 based rendering
- Deferred rendering system
- Support for multiple camera modes
- Debug visualization system

### Physics Integration
- PhysX 5.6.0 integration
- Custom character controller
- Collision detection and response
- Environmental physics

### Reward System
- Zone-based powerups
- Multiple reward types:
  - SLOW_MOTION
  - COMBO_BONUS
  - STAMINA_REFUND
  - DAMAGE_BOOST
  - INVINCIBILITY
  - Others

## Testing Framework

### Automated Testing
- Comprehensive test scenarios
- Input simulation
- State validation
- Performance monitoring

### Debug Systems
- OpenGL debug output
- State tracking
- Performance metrics
- Visual debug overlays

## Asset Management
- Shader loading and compilation
- Mesh and texture management
- Animation system integration
- Resource loading and caching

## Next Development Areas

### Priority Fixes
1. Test suite compilation errors
2. PhysX integration refinement
3. OpenGL debugging improvements
4. Asset loading optimization

### Planned Enhancements
1. Extended combat system
2. Advanced AI behaviors
3. Enhanced visual effects
4. Performance optimizations