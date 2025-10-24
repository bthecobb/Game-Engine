#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>

namespace RhythmArenaCombat {

// Enhanced combo states
enum ComboState {
    COMBO_NONE,
    COMBO_DASH,
    COMBO_LIGHT_1,
    COMBO_LIGHT_2,
    COMBO_LIGHT_3,
    COMBO_HEAVY,
    COMBO_LAUNCHER,
    COMBO_AIR_1,
    COMBO_AIR_2,
    COMBO_AIR_FINISH,
    COMBO_THROW,
    COMBO_RECALL
};

// Thrown weapon state
struct ThrownWeapon {
    glm::vec3 position;
    glm::vec3 velocity;
    float lifetime = 3.0f;
    bool returning = false;
    int weaponType = 0;
    float spinRotation = 0.0f;
};

// Visual feedback
struct DamageNumber {
    glm::vec3 position;
    float value;
    float lifetime = 1.0f;
    glm::vec3 color;
    float scale = 1.0f;
};

// Arena bounds
struct ArenaBounds {
    glm::vec3 center = glm::vec3(0, 0, 0);
    float radius = 30.0f;
    float wallHeight = 0.6f; // Low walls at leg height
    
    bool checkBounds(glm::vec3& position, glm::vec3& velocity) {
        glm::vec3 toCenter = position - center;
        toCenter.y = 0;
        float dist = glm::length(toCenter);
        
        if (dist > radius) {
            // Push back inside
            toCenter = glm::normalize(toCenter);
            position = center + toCenter * radius;
            
            // Stop outward velocity
            float dot = glm::dot(velocity, toCenter);
            if (dot > 0) {
                velocity -= toCenter * dot;
            }
            return true;
        }
        return false;
    }
    
    void render(float currentTime) {
        glm::vec3 boundColor(0.3f, 0.3f, 0.4f);
        
        // Draw circular wall segments
        const int segments = 32;
        for (int i = 0; i < segments; ++i) {
            float angle1 = i * 2.0f * 3.14159f / segments;
            float angle2 = (i + 1) * 2.0f * 3.14159f / segments;
            
            glBegin(GL_QUADS);
            glColor3f(boundColor.r, boundColor.g, boundColor.b);
            
            // Wall face
            glVertex3f(cos(angle1) * radius, 0, sin(angle1) * radius);
            glVertex3f(cos(angle2) * radius, 0, sin(angle2) * radius);
            glVertex3f(cos(angle2) * radius, wallHeight, sin(angle2) * radius);
            glVertex3f(cos(angle1) * radius, wallHeight, sin(angle1) * radius);
            glEnd();
            
            // Top of wall
            glBegin(GL_QUADS);
            glColor3f(boundColor.r * 0.8f, boundColor.g * 0.8f, boundColor.b * 0.8f);
            float innerRadius = radius - 0.2f;
            glVertex3f(cos(angle1) * innerRadius, wallHeight, sin(angle1) * innerRadius);
            glVertex3f(cos(angle2) * innerRadius, wallHeight, sin(angle2) * innerRadius);
            glVertex3f(cos(angle2) * radius, wallHeight, sin(angle2) * radius);
            glVertex3f(cos(angle1) * radius, wallHeight, sin(angle1) * radius);
            glEnd();
        }
    }
};

// Enhanced combat system
class CombatSystem {
public:
    // Combat state
    ComboState currentCombo = COMBO_NONE;
    int comboCount = 0;
    float comboTimer = 0.0f;
    const float COMBO_WINDOW = 0.8f;
    
    // Dash state
    bool isDashing = false;
    float dashTimer = 0.0f;
    const float DASH_DURATION = 0.25f;
    const float DASH_SPEED = 35.0f;
    glm::vec3 dashDirection;
    
    // Weapon throwing
    bool weaponThrown = false;
    ThrownWeapon thrownWeapon;
    
    // Targeting
    int targetedEnemyIndex = -1;
    
    // Visual effects
    std::vector<DamageNumber> damageNumbers;
    float cameraShake = 0.0f;
    
    void update(float deltaTime) {
        // Update combo timer
        if (comboTimer > 0) {
            comboTimer -= deltaTime;
            if (comboTimer <= 0) {
                currentCombo = COMBO_NONE;
                comboCount = 0;
            }
        }
        
        // Update dash
        if (isDashing) {
            dashTimer -= deltaTime;
            if (dashTimer <= 0) {
                isDashing = false;
                currentCombo = COMBO_DASH;
                comboTimer = COMBO_WINDOW;
            }
        }
        
        // Update thrown weapon
        if (weaponThrown) {
            thrownWeapon.spinRotation += deltaTime * 720.0f; // 2 rotations per second
            
            if (!thrownWeapon.returning) {
                thrownWeapon.lifetime -= deltaTime;
                if (thrownWeapon.lifetime <= 0) {
                    thrownWeapon.returning = true;
                }
            }
        }
        
        // Update damage numbers
        for (auto& dmg : damageNumbers) {
            dmg.lifetime -= deltaTime;
            dmg.position.y += deltaTime * 2.0f; // Float upward
            dmg.scale = 1.0f + (1.0f - dmg.lifetime) * 0.5f;
        }
        
        // Remove expired damage numbers
        damageNumbers.erase(
            std::remove_if(damageNumbers.begin(), damageNumbers.end(),
                [](const DamageNumber& d) { return d.lifetime <= 0; }),
            damageNumbers.end()
        );
        
        // Update camera shake
        if (cameraShake > 0) {
            cameraShake -= deltaTime * 3.0f;
            if (cameraShake < 0) cameraShake = 0;
        }
    }
    
    void startDash(const glm::vec3& direction) {
        if (!isDashing && !weaponThrown) {
            dashDirection = glm::normalize(direction);
            isDashing = true;
            dashTimer = DASH_DURATION;
            comboTimer = COMBO_WINDOW;
            cameraShake = 0.1f;
        }
    }
    
    void throwWeapon(const glm::vec3& position, const glm::vec3& direction, int weaponType) {
        if (!weaponThrown) {
            weaponThrown = true;
            thrownWeapon.position = position + glm::vec3(0, 1.5f, 0);
            thrownWeapon.velocity = direction * 25.0f;
            thrownWeapon.lifetime = 3.0f;
            thrownWeapon.returning = false;
            thrownWeapon.weaponType = weaponType;
            thrownWeapon.spinRotation = 0.0f;
            currentCombo = COMBO_THROW;
            comboTimer = COMBO_WINDOW;
        }
    }
    
    void recallWeapon() {
        if (weaponThrown && !thrownWeapon.returning) {
            thrownWeapon.returning = true;
            currentCombo = COMBO_RECALL;
            comboTimer = COMBO_WINDOW;
        }
    }
    
    bool updateWeaponPosition(const glm::vec3& playerPos, float deltaTime) {
        if (!weaponThrown) return false;
        
        if (thrownWeapon.returning) {
            glm::vec3 toPlayer = playerPos - thrownWeapon.position;
            float dist = glm::length(toPlayer);
            
            if (dist < 2.0f) {
                weaponThrown = false;
                comboTimer = COMBO_WINDOW; // Allow combo continuation
                return true; // Weapon retrieved
            }
            
            thrownWeapon.velocity = glm::normalize(toPlayer) * 40.0f;
        }
        
        thrownWeapon.position += thrownWeapon.velocity * deltaTime;
        return false;
    }
    
    void spawnDamageNumber(const glm::vec3& position, float damage, const glm::vec3& color) {
        DamageNumber dmg;
        dmg.position = position + glm::vec3(0, 2.0f, 0);
        dmg.value = damage;
        dmg.lifetime = 1.0f;
        dmg.color = color;
        dmg.scale = 1.0f;
        damageNumbers.push_back(dmg);
    }
    
    void renderThrownWeapon() {
        if (!weaponThrown) return;
        
        glPushMatrix();
        glTranslatef(thrownWeapon.position.x, thrownWeapon.position.y, thrownWeapon.position.z);
        glRotatef(thrownWeapon.spinRotation, 1, 1, 0);
        
        glm::vec3 color = thrownWeapon.returning ? 
            glm::vec3(0.5f, 0.5f, 1.0f) : glm::vec3(0.8f, 0.8f, 0.2f);
        
        glColor3f(color.r, color.g, color.b);
        
        // Weapon shape
        glBegin(GL_QUADS);
        // Blade
        glVertex3f(-0.1f, -1.0f, 0.0f);
        glVertex3f(0.1f, -1.0f, 0.0f);
        glVertex3f(0.1f, 1.0f, 0.0f);
        glVertex3f(-0.1f, 1.0f, 0.0f);
        
        // Cross guard
        glVertex3f(-0.5f, -0.3f, 0.0f);
        glVertex3f(0.5f, -0.3f, 0.0f);
        glVertex3f(0.5f, -0.1f, 0.0f);
        glVertex3f(-0.5f, -0.1f, 0.0f);
        glEnd();
        
        glPopMatrix();
    }
    
    void renderDamageNumbers() {
        glDisable(GL_DEPTH_TEST);
        
        for (const auto& dmg : damageNumbers) {
            glPushMatrix();
            glTranslatef(dmg.position.x, dmg.position.y, dmg.position.z);
            
            glColor4f(dmg.color.r, dmg.color.g, dmg.color.b, dmg.lifetime);
            
            // Scale effect
            float scale = 0.02f * dmg.scale;
            glScalef(scale, scale, scale);
            
            // Draw damage text (simplified as boxes)
            char buffer[16];
            sprintf_s(buffer, "%.0f", dmg.value);
            
            float xOffset = 0;
            for (char* c = buffer; *c; ++c) {
                glBegin(GL_QUADS);
                glVertex3f(xOffset, 0, 0);
                glVertex3f(xOffset + 20, 0, 0);
                glVertex3f(xOffset + 20, 30, 0);
                glVertex3f(xOffset, 30, 0);
                glEnd();
                xOffset += 25;
            }
            
            glPopMatrix();
        }
        
        glEnable(GL_DEPTH_TEST);
    }
    
    void renderComboUI() {
        if (comboCount > 0 && comboTimer > 0) {
            glDisable(GL_DEPTH_TEST);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0, 800, 600, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            
            // Combo counter
            glColor3f(1.0f, 1.0f, 0.0f);
            
            // Background
            glColor4f(0, 0, 0, 0.5f);
            glBegin(GL_QUADS);
            glVertex2f(300, 40);
            glVertex2f(500, 40);
            glVertex2f(500, 80);
            glVertex2f(300, 80);
            glEnd();
            
            // Combo hits
            glColor3f(1.0f, 1.0f, 0.0f);
            for (int i = 0; i < comboCount; ++i) {
                glBegin(GL_QUADS);
                glVertex2f(310 + i * 15, 50);
                glVertex2f(320 + i * 15, 50);
                glVertex2f(320 + i * 15, 70);
                glVertex2f(310 + i * 15, 70);
                glEnd();
            }
            
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glEnable(GL_DEPTH_TEST);
        }
    }
};

// 3D Platform system for vertical combat
struct Platform3D {
    glm::vec3 position;
    glm::vec3 size;
    glm::vec3 color;
    float moveSpeed = 0.0f;
    float moveRange = 0.0f;
    float moveTimer = 0.0f;
    bool isRhythmic = false;
    
    void update(float deltaTime, bool onBeat) {
        if (moveSpeed > 0) {
            moveTimer += deltaTime;
            position.y += sin(moveTimer * moveSpeed) * moveRange * deltaTime;
        }
        
        if (isRhythmic && onBeat) {
            // Pulse on beat
            size *= 1.05f;
        } else if (isRhythmic) {
            // Return to normal size
            size = glm::mix(size, glm::vec3(8.0f, 1.0f, 8.0f), deltaTime * 5.0f);
        }
    }
    
    void render() {
        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        
        glColor3f(color.r, color.g, color.b);
        
        glBegin(GL_QUADS);
        // Top
        glVertex3f(-size.x/2, size.y/2, -size.z/2);
        glVertex3f(size.x/2, size.y/2, -size.z/2);
        glVertex3f(size.x/2, size.y/2, size.z/2);
        glVertex3f(-size.x/2, size.y/2, size.z/2);
        
        // Sides...
        glEnd();
        
        glPopMatrix();
    }
};

} // namespace RhythmArenaCombat
