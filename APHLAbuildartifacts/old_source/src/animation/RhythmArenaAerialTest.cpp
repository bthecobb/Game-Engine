// Simple test to verify aerial combat mechanics
#include <iostream>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Test aerial state
struct AerialTest {
    bool isJumping = false;
    bool isDiving = false;
    bool canDive = false;
    int jumpCount = 0;
    float diveSpeed = 40.0f;
    float diveBounceMultiplier = 1.5f;
    
    glm::vec3 position = glm::vec3(0, 0, 0);
    glm::vec3 velocity = glm::vec3(0, 0, 0);
    
    void jump(float jumpForce) {
        if (jumpCount < 2) {
            velocity.y = jumpForce;
            isJumping = true;
            jumpCount++;
            
            if (jumpCount == 1) {
                canDive = true;
                std::cout << "First jump - can now dive!" << std::endl;
            }
        }
    }
    
    void startDive() {
        if (canDive && !isDiving) {
            isDiving = true;
            canDive = false;
            velocity.y = -diveSpeed;
            std::cout << "DIVING!" << std::endl;
        }
    }
    
    void onGroundHit(float normalJumpForce) {
        if (isDiving) {
            // Dive bounce
            velocity.y = normalJumpForce * diveBounceMultiplier;
            isDiving = false;
            jumpCount = 0; // Reset for another dive cycle
            std::cout << "DIVE BOUNCE! Launching higher!" << std::endl;
        } else {
            // Normal landing
            velocity.y = 0;
            jumpCount = 0;
            isJumping = false;
            canDive = false;
            std::cout << "Landed normally" << std::endl;
        }
    }
    
    void update(float deltaTime) {
        // Gravity
        if (position.y > 0 || velocity.y > 0) {
            velocity.y -= 25.0f * deltaTime;
        }
        
        // Update position
        position += velocity * deltaTime;
        
        // Ground check
        if (position.y <= 0 && velocity.y <= 0) {
            position.y = 0;
            onGroundHit(18.0f); // Normal jump force
        }
    }
    
    void render() {
        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        
        if (isDiving) {
            glColor3f(1.0f, 0.5f, 0.0f); // Orange when diving
            glRotatef(90, 1, 0, 0); // Head-first
        } else if (isJumping) {
            glColor3f(0.0f, 0.5f, 1.0f); // Blue when jumping
        } else {
            glColor3f(0.0f, 1.0f, 0.0f); // Green on ground
        }
        
        // Simple cube
        glBegin(GL_QUADS);
        glVertex3f(-0.5f, -0.5f, 0.5f);
        glVertex3f(0.5f, -0.5f, 0.5f);
        glVertex3f(0.5f, 0.5f, 0.5f);
        glVertex3f(-0.5f, 0.5f, 0.5f);
        glEnd();
        
        glPopMatrix();
    }
};

int main() {
    std::cout << "Aerial Combat Test" << std::endl;
    std::cout << "Space - Jump (twice for double jump)" << std::endl;
    std::cout << "D - Dive (after first jump)" << std::endl;
    std::cout << "Dive landing gives higher bounce!" << std::endl;
    
    // Initialize GLFW
    if (!glfwInit()) {
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Aerial Test", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    AerialTest test;
    double lastTime = glfwGetTime();
    
    bool spacePressed = false;
    bool dPressed = false;
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        
        // Input
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !spacePressed) {
            test.jump(18.0f);
            spacePressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
            spacePressed = false;
        }
        
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS && !dPressed) {
            test.startDive();
            dPressed = true;
        } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_RELEASE) {
            dPressed = false;
        }
        
        // Update
        test.update(deltaTime);
        
        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        // Simple 2D view
        glOrtho(-10, 10, -2, 20, -10, 10);
        
        // Ground
        glColor3f(0.5f, 0.5f, 0.5f);
        glBegin(GL_LINES);
        glVertex3f(-10, 0, 0);
        glVertex3f(10, 0, 0);
        glEnd();
        
        // Height markers
        glColor3f(0.3f, 0.3f, 0.3f);
        for (int i = 5; i <= 20; i += 5) {
            glBegin(GL_LINES);
            glVertex3f(-10, i, 0);
            glVertex3f(-9.5f, i, 0);
            glEnd();
        }
        
        test.render();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
    }
    
    glfwTerminate();
    return 0;
}
