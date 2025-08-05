#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>

namespace RhythmArenaAerial {

// Aerial states for complex air combat
enum AerialState {
    AERIAL_NONE,
    AERIAL_FIRST_JUMP,
    AERIAL_DIVE,
    AERIAL_DIVE_BOUNCE,     // Higher jump after dive
    AERIAL_WEAPON_DIVE,     // X + Dive with weapon out
    AERIAL_CRASH_LANDING,   // Impact state
    AERIAL_RECOVERY
};

// Particle system for dramatic effects
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 color;
    float lifetime;
    float size;
    float rotation;
    bool isGlowing;
};

class AerialCombatSystem {
public:
    // Aerial state
    AerialState aerialState = AERIAL_NONE;
    float diveSpeed = 40.0f;
    float diveBounceHeight = 25.0f;  // Higher than double jump
    float weaponDiveSpeed = 60.0f;   // Faster with weapon
    
    // Dive mechanics
    bool isDiving = false;
    bool canDiveAgain = false;
    float diveTimer = 0.0f;
    glm::vec3 diveDirection;
    float diveRotation = 0.0f;
    
    // Weapon dive
    bool isWeaponDiving = false;
    float weaponDiveCharge = 0.0f;
    float impactRadius = 5.0f;
    float impactDamage = 100.0f;
    
    // Visual effects
    std::vector<Particle> particles;
    std::vector<Particle> impactParticles;
    float screenShake = 0.0f;
    glm::vec3 impactPosition;
    float impactWaveRadius = 0.0f;
    
    // Jump tracking
    int airJumpCount = 0;
    float airTime = 0.0f;
    float maxAirTime = 3.0f;
    
    void update(float deltaTime) {
        // Update air time
        if (aerialState != AERIAL_NONE) {
            airTime += deltaTime;
        }
        
        // Update dive rotation
        if (isDiving || isWeaponDiving) {
            diveRotation += deltaTime * 720.0f; // 2 spins per second
        }
        
        // Update particles
        updateParticles(deltaTime);
        
        // Update impact effects
        if (impactWaveRadius > 0) {
            impactWaveRadius += deltaTime * 20.0f;
            if (impactWaveRadius > impactRadius * 3) {
                impactWaveRadius = 0;
            }
        }
        
        // Update screen shake
        if (screenShake > 0) {
            screenShake -= deltaTime * 5.0f;
            if (screenShake < 0) screenShake = 0;
        }
    }
    
    bool canDive() {
        return aerialState == AERIAL_FIRST_JUMP && !isDiving;
    }
    
    bool canWeaponDive() {
        return (aerialState == AERIAL_DIVE_BOUNCE || 
                aerialState == AERIAL_FIRST_JUMP) && 
                !isWeaponDiving && airJumpCount > 0;
    }
    
    void startDive(glm::vec3 playerVelocity) {
        if (!canDive()) return;
        
        isDiving = true;
        aerialState = AERIAL_DIVE;
        diveDirection = glm::normalize(glm::vec3(playerVelocity.x, -1.0f, playerVelocity.z));
        diveTimer = 0.5f;
        canDiveAgain = true;
        
        // Create dive trail particles
        for (int i = 0; i < 10; ++i) {
            createDiveParticle(glm::vec3(0), glm::vec3(1.0f, 0.8f, 0.3f));
        }
    }
    
    void startWeaponDive(glm::vec3 position, glm::vec3 targetDirection) {
        if (!canWeaponDive()) return;
        
        isWeaponDiving = true;
        aerialState = AERIAL_WEAPON_DIVE;
        diveDirection = glm::normalize(targetDirection + glm::vec3(0, -2, 0));
        weaponDiveCharge = 0.0f;
        
        // Create charged particles
        for (int i = 0; i < 30; ++i) {
            Particle p;
            p.position = position;
            p.velocity = glm::vec3(
                (rand() % 20 - 10) * 0.1f,
                (rand() % 20 - 10) * 0.1f,
                (rand() % 20 - 10) * 0.1f
            );
            p.color = glm::vec3(0.8f, 0.4f, 1.0f); // Purple energy
            p.lifetime = 1.0f;
            p.size = 0.3f;
            p.rotation = rand() % 360;
            p.isGlowing = true;
            particles.push_back(p);
        }
    }
    
    glm::vec3 getDiveVelocity() {
        if (isWeaponDiving) {
            return diveDirection * weaponDiveSpeed;
        } else if (isDiving) {
            return diveDirection * diveSpeed;
        }
        return glm::vec3(0);
    }
    
    void onGroundImpact(glm::vec3 position, bool fromWeaponDive) {
        impactPosition = position;
        
        if (fromWeaponDive) {
            // Massive impact
            screenShake = 1.0f;
            impactWaveRadius = 0.1f;
            aerialState = AERIAL_CRASH_LANDING;
            
            // Create explosion of particles
            for (int i = 0; i < 100; ++i) {
                Particle p;
                p.position = position;
                
                // Radial explosion
                float angle = (i / 100.0f) * 2.0f * 3.14159f;
                float speed = 10.0f + (rand() % 100) * 0.1f;
                p.velocity = glm::vec3(
                    cos(angle) * speed,
                    5.0f + (rand() % 100) * 0.1f,
                    sin(angle) * speed
                );
                
                // Color gradient from white to orange
                float t = (rand() % 100) / 100.0f;
                p.color = glm::mix(
                    glm::vec3(1.0f, 1.0f, 1.0f),
                    glm::vec3(1.0f, 0.5f, 0.0f),
                    t
                );
                
                p.lifetime = 2.0f;
                p.size = 0.5f + (rand() % 100) * 0.01f;
                p.rotation = rand() % 360;
                p.isGlowing = true;
                impactParticles.push_back(p);
            }
            
            // Ground crack particles
            for (int i = 0; i < 50; ++i) {
                Particle p;
                float angle = (i / 50.0f) * 2.0f * 3.14159f;
                float dist = impactRadius * (0.5f + (rand() % 100) * 0.005f);
                p.position = position + glm::vec3(cos(angle) * dist, 0.1f, sin(angle) * dist);
                p.velocity = glm::vec3(0, 0.5f, 0);
                p.color = glm::vec3(0.3f, 0.3f, 0.3f);
                p.lifetime = 3.0f;
                p.size = 0.3f;
                p.rotation = angle * 180.0f / 3.14159f;
                p.isGlowing = false;
                impactParticles.push_back(p);
            }
            
        } else if (isDiving && canDiveAgain) {
            // Dive bounce - launch higher
            aerialState = AERIAL_DIVE_BOUNCE;
            screenShake = 0.3f;
            
            // Bounce particles
            for (int i = 0; i < 30; ++i) {
                Particle p;
                p.position = position;
                p.velocity = glm::vec3(
                    (rand() % 20 - 10) * 0.3f,
                    5.0f + (rand() % 10) * 0.5f,
                    (rand() % 20 - 10) * 0.3f
                );
                p.color = glm::vec3(0.3f, 0.8f, 1.0f); // Blue energy
                p.lifetime = 1.0f;
                p.size = 0.2f;
                p.rotation = 0;
                p.isGlowing = true;
                particles.push_back(p);
            }
        }
        
        isDiving = false;
        isWeaponDiving = false;
        diveRotation = 0;
    }
    
    float getDiveBounceVelocity() {
        return diveBounceHeight;
    }
    
    bool checkImpactDamage(glm::vec3 enemyPos) {
        if (aerialState == AERIAL_CRASH_LANDING && impactWaveRadius > 0) {
            float dist = glm::length(enemyPos - impactPosition);
            if (dist < impactRadius) {
                return true;
            }
        }
        return false;
    }
    
    void resetAerialState() {
        aerialState = AERIAL_NONE;
        isDiving = false;
        isWeaponDiving = false;
        canDiveAgain = false;
        airJumpCount = 0;
        airTime = 0;
        diveRotation = 0;
    }
    
    void render(glm::vec3 playerPos, float currentTime) {
        // Render dive trail
        if (isDiving || isWeaponDiving) {
            glPushMatrix();
            glTranslatef(playerPos.x, playerPos.y, playerPos.z);
            glRotatef(diveRotation, diveDirection.x, diveDirection.y, diveDirection.z);
            
            // Spiral trail effect
            glBegin(GL_LINES);
            for (int i = 0; i < 10; ++i) {
                float angle = i * 0.628f + currentTime * 10;
                float radius = 0.5f + i * 0.1f;
                
                glm::vec3 color = isWeaponDiving ? 
                    glm::vec3(0.8f, 0.3f, 1.0f) : glm::vec3(1.0f, 0.8f, 0.3f);
                
                glColor4f(color.r, color.g, color.b, 1.0f - i * 0.1f);
                glVertex3f(cos(angle) * radius, i * 0.2f, sin(angle) * radius);
                glVertex3f(cos(angle + 0.1f) * radius, (i + 0.1f) * 0.2f, sin(angle + 0.1f) * radius);
            }
            glEnd();
            
            glPopMatrix();
        }
        
        // Render particles
        glBegin(GL_POINTS);
        glPointSize(3.0f);
        
        for (const auto& p : particles) {
            float alpha = p.lifetime;
            if (p.isGlowing) {
                glColor4f(p.color.r, p.color.g, p.color.b, alpha);
            } else {
                glColor4f(p.color.r, p.color.g, p.color.b, alpha * 0.5f);
            }
            glVertex3f(p.position.x, p.position.y, p.position.z);
        }
        
        for (const auto& p : impactParticles) {
            float alpha = p.lifetime / 2.0f;
            glColor4f(p.color.r, p.color.g, p.color.b, alpha);
            glVertex3f(p.position.x, p.position.y, p.position.z);
        }
        glEnd();
        
        // Render impact shockwave
        if (impactWaveRadius > 0) {
            glPushMatrix();
            glTranslatef(impactPosition.x, impactPosition.y + 0.1f, impactPosition.z);
            
            glColor4f(1.0f, 0.5f, 0.0f, 1.0f - impactWaveRadius / (impactRadius * 3));
            
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < 32; ++i) {
                float angle = i * 2.0f * 3.14159f / 32.0f;
                glVertex3f(cos(angle) * impactWaveRadius, 0, sin(angle) * impactWaveRadius);
            }
            glEnd();
            
            glPopMatrix();
        }
    }
    
private:
    void createDiveParticle(glm::vec3 offset, glm::vec3 color) {
        Particle p;
        p.position = offset;
        p.velocity = glm::vec3(
            (rand() % 10 - 5) * 0.1f,
            (rand() % 10) * 0.1f,
            (rand() % 10 - 5) * 0.1f
        );
        p.color = color;
        p.lifetime = 0.5f;
        p.size = 0.2f;
        p.rotation = rand() % 360;
        p.isGlowing = true;
        particles.push_back(p);
    }
    
    void updateParticles(float deltaTime) {
        // Update regular particles
        for (auto& p : particles) {
            p.position += p.velocity * deltaTime;
            p.velocity.y -= 9.8f * deltaTime; // Gravity
            p.lifetime -= deltaTime;
            p.rotation += deltaTime * 180.0f;
        }
        
        // Update impact particles
        for (auto& p : impactParticles) {
            p.position += p.velocity * deltaTime;
            p.velocity *= 0.95f; // Friction
            p.velocity.y -= 15.0f * deltaTime; // Heavier gravity
            p.lifetime -= deltaTime;
        }
        
        // Remove dead particles
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return p.lifetime <= 0; }),
            particles.end()
        );
        
        impactParticles.erase(
            std::remove_if(impactParticles.begin(), impactParticles.end(),
                [](const Particle& p) { return p.lifetime <= 0; }),
            impactParticles.end()
        );
    }
};

// Air combo extensions
class AirComboSystem {
public:
    enum AirComboState {
        AIR_COMBO_NONE,
        AIR_COMBO_LAUNCHER,
        AIR_COMBO_JUGGLE_1,
        AIR_COMBO_JUGGLE_2,
        AIR_COMBO_JUGGLE_3,
        AIR_COMBO_FINISHER
    };
    
    AirComboState currentAirCombo = AIR_COMBO_NONE;
    float airComboTimer = 0.0f;
    int airHitCount = 0;
    
    void startAirCombo() {
        currentAirCombo = AIR_COMBO_LAUNCHER;
        airComboTimer = 1.0f;
        airHitCount = 0;
    }
    
    void continueAirCombo() {
        airHitCount++;
        airComboTimer = 0.8f;
        
        switch (currentAirCombo) {
            case AIR_COMBO_LAUNCHER:
                currentAirCombo = AIR_COMBO_JUGGLE_1;
                break;
            case AIR_COMBO_JUGGLE_1:
                currentAirCombo = AIR_COMBO_JUGGLE_2;
                break;
            case AIR_COMBO_JUGGLE_2:
                currentAirCombo = AIR_COMBO_JUGGLE_3;
                break;
            case AIR_COMBO_JUGGLE_3:
                currentAirCombo = AIR_COMBO_FINISHER;
                break;
        }
    }
    
    float getAirComboDamageMultiplier() {
        return 1.0f + (airHitCount * 0.3f);
    }
    
    void update(float deltaTime) {
        if (airComboTimer > 0) {
            airComboTimer -= deltaTime;
            if (airComboTimer <= 0) {
                currentAirCombo = AIR_COMBO_NONE;
                airHitCount = 0;
            }
        }
    }
};

} // namespace RhythmArenaAerial
