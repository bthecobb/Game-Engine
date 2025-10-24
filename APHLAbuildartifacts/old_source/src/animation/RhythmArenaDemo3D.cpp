#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "window.h"
#include "RhythmArenaDemo.h"

// Combat system structures
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
    COMBO_AIR_FINISH
};

struct CombatState {
    ComboState currentCombo = COMBO_NONE;
    int comboCount = 0;
    float comboTimer = 0.0f;
    const float COMBO_WINDOW = 0.8f;
    bool isDashing = false;
    float dashTimer = 0.0f;
    const float DASH_DURATION = 0.25f;
    const float DASH_SPEED = 35.0f;
    glm::vec3 dashDirection;
    int targetedEnemyIndex = -1;
    float lastHitTime = 0.0f;
};

// Enhanced Player structure
struct Player3D {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float rotation;
    float health;
    float maxHealth;
    bool isGrounded;
    bool isAirborne;
    float airTime;
    int jumpCount;
    CombatState combat;
    
    // Animation states
    std::string currentAnimation;
    float animationTime;
    
    Player3D() : position(0, 1, 0), velocity(0), size(0.8f, 1.8f, 0.8f),
                rotation(0), health(100), maxHealth(100), isGrounded(true),
                isAirborne(false), airTime(0), jumpCount(0),
                currentAnimation("idle"), animationTime(0) {}
};

// Enhanced Enemy structure with better AI
struct Enemy3D {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
    float rotation;
    float health;
    float maxHealth;
    bool isDead;
    float deathTimer;
    
    // Combat AI
    float aggroRange;
    float attackRange;
    float attackCooldown;
    bool isAggro;
    bool isStunned;
    float stunTimer;
    bool isLaunched;
    float launchTimer;
    
    // Visual state
    glm::vec3 color;
    float damageFlash;
    
    Enemy3D(glm::vec3 pos) : position(pos), velocity(0), size(1.2f, 2.0f, 1.2f),
                            rotation(0), health(100), maxHealth(100), isDead(false),
                            deathTimer(0), aggroRange(15.0f), attackRange(2.5f),
                            attackCooldown(0), isAggro(false), isStunned(false),
                            stunTimer(0), isLaunched(false), launchTimer(0),
                            color(1.0f, 0.2f, 0.2f), damageFlash(0) {}
};

// Enhanced game state
Player3D player3D;
std::vector<Enemy3D> enemies3D;
CombatState combatState;
float currentTime = 0;
int score = 0;
bool showDamageNumbers = true;
float cameraShake = 0;

// Damage numbers for visual feedback
struct DamageNumber {
    glm::vec3 position;
    float damage;
    float lifetime;
    glm::vec3 color;
};
std::vector<DamageNumber> damageNumbers;

// Particle effects
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float lifetime;
    glm::vec3 color;
    float size;
};
std::vector<Particle> particles;

// Function to update targeting system
void updateTargeting3D() {
    if (enemies3D.empty()) {
        combatState.targetedEnemyIndex = -1;
        return;
    }
    
    float closestDistance = FLT_MAX;
    int closestIndex = -1;
    
    for (int i = 0; i < enemies3D.size(); ++i) {
        if (enemies3D[i].isDead) continue;
        
        glm::vec3 toEnemy = enemies3D[i].position - player3D.position;
        float distance = glm::length(toEnemy);
        
        if (distance < 20.0f) { // Max targeting range
            toEnemy = glm::normalize(toEnemy);
            
            // Get player forward direction
            float radians = player3D.rotation * 3.14159f / 180.0f;
            glm::vec3 forward(sin(radians), 0, -cos(radians));
            
            // Check if enemy is in front (dot product)
            float dot = glm::dot(forward, toEnemy);
            if (dot > 0.3f) { // Within ~70 degree cone
                // Prioritize launched enemies for air combos
                if (enemies3D[i].isLaunched && player3D.isAirborne) {
                    closestIndex = i;
                    break;
                }
                
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestIndex = i;
                }
            }
        }
    }
    
    combatState.targetedEnemyIndex = closestIndex;
}

// Enhanced combat hit detection
void processCombatHit(float baseDamage, float range, bool isLauncher = false) {
    if (combatState.targetedEnemyIndex >= 0 && combatState.targetedEnemyIndex < enemies3D.size()) {
        Enemy3D& enemy = enemies3D[combatState.targetedEnemyIndex];
        if (!enemy.isDead) {
            float distance = glm::length(enemy.position - player3D.position);
            if (distance <= range) {
                // Calculate combo multiplier
                float comboMultiplier = 1.0f + (combatState.comboCount * 0.2f);
                float totalDamage = baseDamage * comboMultiplier;
                
                // Apply damage
                enemy.health -= totalDamage;
                enemy.damageFlash = 1.0f;
                
                // Create damage number
                DamageNumber dmg;
                dmg.position = enemy.position + glm::vec3(0, 2.5f, 0);
                dmg.damage = totalDamage;
                dmg.lifetime = 1.0f;
                dmg.color = (combatState.comboCount > 3) ? glm::vec3(1, 1, 0) : glm::vec3(1, 1, 1);
                damageNumbers.push_back(dmg);
                
                // Apply effects
                if (isLauncher) {
                    enemy.isLaunched = true;
                    enemy.launchTimer = 1.0f;
                    enemy.velocity.y = 15.0f;
                    cameraShake = 0.3f;
                } else {
                    // Knockback
                    glm::vec3 knockback = glm::normalize(enemy.position - player3D.position);
                    enemy.velocity += knockback * 5.0f;
                    enemy.isStunned = true;
                    enemy.stunTimer = 0.3f;
                }
                
                // Create hit particles
                for (int i = 0; i < 10; ++i) {
                    Particle p;
                    p.position = enemy.position;
                    p.velocity = glm::vec3(
                        (rand() % 10 - 5) * 0.5f,
                        rand() % 10 * 0.5f,
                        (rand() % 10 - 5) * 0.5f
                    );
                    p.lifetime = 0.5f;
                    p.color = enemy.color;
                    p.size = 0.1f;
                    particles.push_back(p);
                }
                
                if (enemy.health <= 0) {
                    enemy.isDead = true;
                    enemy.deathTimer = 2.0f;
                    score += 1000 * combatState.comboCount;
                    
                    // Big particle explosion on death
                    for (int i = 0; i < 30; ++i) {
                        Particle p;
                        p.position = enemy.position;
                        p.velocity = glm::vec3(
                            (rand() % 20 - 10) * 0.8f,
                            rand() % 15 * 0.8f,
                            (rand() % 20 - 10) * 0.8f
                        );
                        p.lifetime = 1.0f;
                        p.color = glm::vec3(1, 0.5f, 0);
                        p.size = 0.2f;
                        particles.push_back(p);
                    }
                }
            }
        }
    }
}

// Process dash attack
void processDashAttack() {
    if (combatState.targetedEnemyIndex >= 0 && combatState.targetedEnemyIndex < enemies3D.size()) {
        glm::vec3 toEnemy = enemies3D[combatState.targetedEnemyIndex].position - player3D.position;
        toEnemy.y = 0;
        if (glm::length(toEnemy) > 0.1f) {
            combatState.dashDirection = glm::normalize(toEnemy);
            combatState.isDashing = true;
            combatState.dashTimer = combatState.DASH_DURATION;
            player3D.currentAnimation = "dash";
            
            // Create dash trail particles
            for (int i = 0; i < 5; ++i) {
                Particle p;
                p.position = player3D.position;
                p.velocity = -combatState.dashDirection * 2.0f;
                p.lifetime = 0.3f;
                p.color = glm::vec3(0.5f, 0.5f, 1.0f);
                p.size = 0.3f;
                particles.push_back(p);
            }
        }
    }
}

// Update combat system
void updateCombat3D(float deltaTime) {
    // Update combo timer
    if (combatState.comboTimer > 0) {
        combatState.comboTimer -= deltaTime;
        if (combatState.comboTimer <= 0) {
            combatState.currentCombo = COMBO_NONE;
            combatState.comboCount = 0;
        }
    }
    
    // Update dash
    if (combatState.isDashing) {
        combatState.dashTimer -= deltaTime;
        if (combatState.dashTimer <= 0) {
            combatState.isDashing = false;
            combatState.currentCombo = COMBO_DASH;
            combatState.comboTimer = combatState.COMBO_WINDOW;
        } else {
            player3D.velocity = combatState.dashDirection * combatState.DASH_SPEED;
        }
    }
    
    // Update targeting
    updateTargeting3D();
}

// Render enhanced enemy with visual effects
void renderEnemy3D(const Enemy3D& enemy, int index) {
    if (enemy.isDead && enemy.deathTimer <= 0) return;
    
    glPushMatrix();
    glTranslatef(enemy.position.x, enemy.position.y, enemy.position.z);
    glRotatef(enemy.rotation, 0, 1, 0);
    
    // Apply visual effects
    float scale = 1.0f;
    glm::vec3 color = enemy.color;
    
    if (enemy.isDead) {
        scale = enemy.deathTimer / 2.0f;
        color *= scale;
    }
    
    if (enemy.damageFlash > 0) {
        color = glm::mix(color, glm::vec3(1, 1, 1), enemy.damageFlash);
    }
    
    if (enemy.isLaunched) {
        glRotatef(enemy.launchTimer * 720.0f, 1, 0, 0); // Spin when launched
    }
    
    glScalef(scale, scale, scale);
    
    // Draw enemy body
    glColor3f(color.r, color.g, color.b);
    glBegin(GL_QUADS);
    // Front face
    glVertex3f(-0.5f, -1.0f, 0.5f);
    glVertex3f(0.5f, -1.0f, 0.5f);
    glVertex3f(0.5f, 1.0f, 0.5f);
    glVertex3f(-0.5f, 1.0f, 0.5f);
    // Add other faces...
    glEnd();
    
    // Draw targeting indicator
    if (index == combatState.targetedEnemyIndex) {
        glPushMatrix();
        glTranslatef(0, 2.0f, 0);
        glRotatef(currentTime * 180.0f, 0, 1, 0);
        
        glLineWidth(3.0f);
        glBegin(GL_LINE_LOOP);
        glColor3f(1.0f, 1.0f, 0.0f);
        for (int i = 0; i < 16; ++i) {
            float angle = i * 3.14159f * 2.0f / 16.0f;
            glVertex3f(cos(angle) * 0.8f, 0, sin(angle) * 0.8f);
        }
        glEnd();
        glLineWidth(1.0f);
        
        glPopMatrix();
    }
    
    glPopMatrix();
}

// Main render function for 3D combat demo
void render3DCombat() {
    // Clear and setup
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    // Apply camera shake
    float shakeX = 0, shakeY = 0;
    if (cameraShake > 0) {
        shakeX = (rand() % 10 - 5) * cameraShake * 0.1f;
        shakeY = (rand() % 10 - 5) * cameraShake * 0.1f;
    }
    
    // Set camera
    gluLookAt(
        player3D.position.x + shakeX, player3D.position.y + 10 + shakeY, player3D.position.z + 15,
        player3D.position.x, player3D.position.y, player3D.position.z,
        0, 1, 0
    );
    
    // Render ground
    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_QUADS);
    glVertex3f(-50, 0, -50);
    glVertex3f(50, 0, -50);
    glVertex3f(50, 0, 50);
    glVertex3f(-50, 0, 50);
    glEnd();
    
    // Render player
    glPushMatrix();
    glTranslatef(player3D.position.x, player3D.position.y, player3D.position.z);
    glRotatef(player3D.rotation, 0, 1, 0);
    
    // Player color changes based on combo
    if (combatState.comboCount > 5) {
        glColor3f(1.0f, 0.8f, 0.0f); // Gold
    } else if (combatState.comboCount > 3) {
        glColor3f(0.8f, 0.8f, 1.0f); // Light blue
    } else {
        glColor3f(0.0f, 0.5f, 1.0f); // Blue
    }
    
    // Simple player cube for now
    glBegin(GL_QUADS);
    glVertex3f(-0.4f, -0.9f, 0.4f);
    glVertex3f(0.4f, -0.9f, 0.4f);
    glVertex3f(0.4f, 0.9f, 0.4f);
    glVertex3f(-0.4f, 0.9f, 0.4f);
    glEnd();
    
    glPopMatrix();
    
    // Render enemies
    for (int i = 0; i < enemies3D.size(); ++i) {
        renderEnemy3D(enemies3D[i], i);
    }
    
    // Render particles
    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        glColor4f(p.color.r, p.color.g, p.color.b, p.lifetime);
        glVertex3f(p.position.x, p.position.y, p.position.z);
    }
    glEnd();
    
    // Render damage numbers
    for (const auto& dmg : damageNumbers) {
        glPushMatrix();
        glTranslatef(dmg.position.x, dmg.position.y + (1.0f - dmg.lifetime), dmg.position.z);
        
        // Billboard the text to face camera
        glm::vec3 toCamera = glm::vec3(0, 10, 15) - dmg.position;
        float angle = atan2(toCamera.x, toCamera.z) * 180.0f / 3.14159f;
        glRotatef(-angle, 0, 1, 0);
        
        glColor4f(dmg.color.r, dmg.color.g, dmg.color.b, dmg.lifetime);
        
        // Simple damage number visualization
        glBegin(GL_LINES);
        // Draw a simple "!" for now
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0.5f, 0);
        glEnd();
        
        glPopMatrix();
    }
    
    // UI overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 800, 600, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    // Combo counter
    if (combatState.comboCount > 0) {
        glColor3f(1, 1, 0);
        // Draw combo indicator
        for (int i = 0; i < combatState.comboCount; ++i) {
            glBegin(GL_QUADS);
            glVertex2f(350 + i * 15, 50);
            glVertex2f(360 + i * 15, 50);
            glVertex2f(360 + i * 15, 60);
            glVertex2f(350 + i * 15, 60);
            glEnd();
        }
    }
    
    // Health bar
    glColor3f(0.2f, 0.2f, 0.2f);
    glBegin(GL_QUADS);
    glVertex2f(50, 550);
    glVertex2f(250, 550);
    glVertex2f(250, 570);
    glVertex2f(50, 570);
    glEnd();
    
    float healthPercent = player3D.health / player3D.maxHealth;
    glColor3f(1.0f - healthPercent, healthPercent, 0);
    glBegin(GL_QUADS);
    glVertex2f(50, 550);
    glVertex2f(50 + 200 * healthPercent, 550);
    glVertex2f(50 + 200 * healthPercent, 570);
    glVertex2f(50, 570);
    glEnd();
    
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// Update function for particles and effects
void updateEffects(float deltaTime) {
    // Update particles
    for (auto& p : particles) {
        p.position += p.velocity * deltaTime;
        p.velocity.y -= 9.8f * deltaTime; // Gravity
        p.lifetime -= deltaTime;
    }
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.lifetime <= 0; }),
        particles.end()
    );
    
    // Update damage numbers
    for (auto& dmg : damageNumbers) {
        dmg.lifetime -= deltaTime;
    }
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
    
    // Update enemy states
    for (auto& enemy : enemies3D) {
        if (enemy.damageFlash > 0) {
            enemy.damageFlash -= deltaTime * 4.0f;
        }
        if (enemy.isStunned) {
            enemy.stunTimer -= deltaTime;
            if (enemy.stunTimer <= 0) {
                enemy.isStunned = false;
            }
        }
        if (enemy.isLaunched) {
            enemy.launchTimer -= deltaTime;
            if (enemy.launchTimer <= 0) {
                enemy.isLaunched = false;
            }
        }
    }
}
