# Rhythm Arena TODO & Bug Tracker

## TODO List

### Combat System
- [x] Basic punch/kick mechanics
- [x] Combo system
- [ ] Enemy health and damage system
- [ ] Player health system
- [ ] Weapon equipping system
- [ ] Weapon inventory
- [ ] Different weapon types (sword, staff, etc.)
- [ ] Rhythm-based damage multipliers
- [ ] Perfect timing indicators
- [ ] Combo catch-up to rhythm system

### Rhythm Integration
- [ ] Toggle for rhythm flow visualization
- [ ] Beat indicators on screen
- [ ] Movement snap-to-beat option
- [ ] Combat moves align with beat
- [ ] Dynamic BPM per level
- [ ] Rhythm accuracy scoring
- [ ] Visual rhythm feedback (screen pulse, etc.)
- [ ] Audio feedback for on-beat actions

### Enemy System
- [ ] Enemy health bars
- [ ] Enemy destruction animations
- [ ] Different enemy types
- [ ] Enemy attack patterns
- [ ] Enemy rhythm-based behavior
- [ ] Boss enemies

### Level System
- [ ] Level loader
- [ ] Custom BPM per level
- [ ] Level progression
- [ ] Checkpoint system
- [ ] Level-specific rhythm patterns

### UI/UX
- [ ] HUD for health/score/combo
- [ ] Rhythm indicator UI
- [ ] Weapon selection UI
- [ ] Pause menu
- [ ] Settings menu

### Visual Effects
- [ ] Particle effects for hits
- [ ] Screen shake on impact
- [ ] Rhythm-based visual effects
- [ ] Weapon trail effects

## Known Bugs

### Current Issues
1. **Enemy Collision** - Enemies can overlap with each other
2. **Wall Running** - Sometimes doesn't activate at high speeds
3. **Double Jump** - Can occasionally be triggered multiple times
4. **Collectible Respawn** - Collectibles don't respawn properly after reset

### Fixed Issues
- [x] Character rotation when moving
- [x] Combat animation states
- [x] Score system implementation

## Feature Requests
1. Multiplayer support
2. Level editor
3. Custom music import
4. Leaderboards
5. Replay system

## Performance Notes
- Need to optimize enemy AI for large numbers
- Particle system could use GPU acceleration
- Consider LOD for distant objects
