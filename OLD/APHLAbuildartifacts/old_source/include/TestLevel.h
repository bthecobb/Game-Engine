#pragma once

#include "GameWorld.h"
#include <vector>
#include <memory>

class Platform : public GameObject {
public:
    Platform(glm::vec3 position, glm::vec3 size, bool visibleViews[4]);
    ~Platform() override = default;
    
    void update(float deltaTime) override;
    void render() override;
    
private:
    glm::vec3 m_color;
    unsigned int m_VAO, m_VBO, m_EBO;
    void setupMesh();
};

class Wall : public GameObject {
public:
    Wall(glm::vec3 position, glm::vec3 size, bool visibleViews[4]);
    ~Wall() override = default;
    
    void update(float deltaTime) override;
    void render() override;
    
private:
    glm::vec3 m_color;
    unsigned int m_VAO, m_VBO, m_EBO;
    void setupMesh();
};

class TestLevel {
public:
    TestLevel();
    ~TestLevel() = default;
    
    void createLevel(GameWorld* gameWorld);
    void createDimensionalPlatforms(GameWorld* gameWorld);
    void createWallRunSections(GameWorld* gameWorld);
    void createSecretAreas(GameWorld* gameWorld);
    
private:
    void addPlatform(GameWorld* gameWorld, glm::vec3 pos, glm::vec3 size, bool views[4]);
    void addWall(GameWorld* gameWorld, glm::vec3 pos, glm::vec3 size, bool views[4]);
};
