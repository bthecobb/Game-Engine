/*
 * Phase 2.6: Wall-Running Physics Integration Example
 * 
 * This file demonstrates how to integrate the WallRunningSystem 
 * with your existing game systems for momentum-conserving wall-running mechanics.
 */

#include "../include_refactored/Physics/WallRunningSystem.h"
#include "../include_refactored/Physics/CharacterController.h"
#include "../include_refactored/GameFeel/GameFeelSystem.h"
#include "../include_refactored/Combat/ComboManager.h"
#include "../include_refactored/Core/Coordinator.h"

using namespace CudaGame;

class WallRunningIntegrationDemo {
private:
    Core::Coordinator* m_coordinator;
    std::shared_ptr<Physics::WallRunningSystem> m_wallRunSystem;
    std::shared_ptr<GameFeel::GameFeelSystem> m_gameFeelSystem;
    std::shared_ptr<Combat::ComboManager> m_comboManager;

public:
    bool Initialize() {
        m_coordinator = Core::Coordinator::GetInstance();
        
        // Initialize systems
        m_wallRunSystem = std::make_shared<Physics::WallRunningSystem>();
        m_gameFeelSystem = std::make_shared<GameFeel::GameFeelSystem>();
        m_comboManager = std::make_shared<Combat::ComboManager>();
        
        // Register systems with coordinator
        m_coordinator->RegisterSystem<Physics::WallRunningSystem>(m_wallRunSystem);
        m_coordinator->RegisterSystem<GameFeel::GameFeelSystem>(m_gameFeelSystem);
        m_coordinator->RegisterSystem<Combat::ComboManager>(m_comboManager);
        
        // Configure wall-running system
        ConfigureWallRunningSystem();
        
        // Set up integration callbacks
        SetupSystemIntegration();
        
        return true;
    }
    
    void ConfigureWallRunningSystem() {
        // Configure wall-running physics parameters
        m_wallRunSystem->SetWallRunGravityScale(0.2f);      // Reduced gravity during wall-running
        m_wallRunSystem->SetWallRunMinSpeed(6.0f);          // Minimum speed required to start wall-running
        m_wallRunSystem->SetWallRunMaxTime(2.5f);           // Maximum wall-running duration
        m_wallRunSystem->SetWallDetectionRange(1.8f);       // Range for detecting walls
        m_wallRunSystem->SetMomentumConservation(0.85f);    // How much momentum to preserve (85%)
        m_wallRunSystem->SetMaxWallRunAngle(35.0f);         // Maximum wall angle in degrees
        
        // Enable debug visualization for development
        m_wallRunSystem->SetDebugVisualization(true);
    }
    
    void SetupSystemIntegration() {
        // Integrate wall-running with game feel system for enhanced feedback
        m_wallRunSystem->RegisterWallRunStartCallback([this](Core::Entity entity, const glm::vec3& normal) {
            // Trigger screen shake when starting wall-run
            GameFeel::ScreenShakeParams shakeParams;
            shakeParams.type = GameFeel::ScreenShakeType::Punch;
            shakeParams.intensity = 0.3f;
            shakeParams.duration = 0.2f;
            m_gameFeelSystem->TriggerScreenShake(shakeParams);
            
            // Apply slight hit-stop for impact feel
            m_gameFeelSystem->ApplyHitStop(0.05f);
            
            std::cout << \"[WallRun] Started wall-running! Surface normal: (\" 
                     << normal.x << \", \" << normal.y << \", \" << normal.z << \")\\n\";\n        });\n        \n        m_wallRunSystem->RegisterWallRunEndCallback([this](Core::Entity entity, bool jumped) {\n            if (jumped) {\n                // Trigger more intense effects for wall-jump\n                GameFeel::ScreenShakeParams shakeParams;\n                shakeParams.type = GameFeel::ScreenShakeType::Explosion;\n                shakeParams.intensity = 0.5f;\n                shakeParams.duration = 0.3f;\n                m_gameFeelSystem->TriggerScreenShake(shakeParams);\n                \n                std::cout << \"[WallRun] Wall-jumped with enhanced momentum!\\n\";\n            } else {\n                std::cout << \"[WallRun] Ended wall-run naturally.\\n\";\n            }\n        });\n    }\n    \n    Core::Entity CreateWallRunningCharacter() {\n        Core::Entity character = m_coordinator->CreateEntity();\n        \n        // Add required components\n        Physics::CharacterControllerComponent controller;\n        controller.maxWallRunTime = 3.0f;\n        controller.wallRunSpeed = 12.0f;\n        controller.jumpForce = 18.0f;\n        controller.maxAirJumps = 2;  // Double jump capability\n        \n        Physics::RigidbodyComponent rigidbody;\n        rigidbody.setMass(70.0f);  // Average human mass\n        rigidbody.friction = 0.1f;\n        rigidbody.restitution = 0.1f;\n        \n        Physics::ColliderComponent collider;\n        collider.type = Physics::ColliderType::CAPSULE;\n        collider.capsuleHeight = 1.8f;\n        collider.capsuleRadius = 0.3f;\n        \n        Rendering::TransformComponent transform;\n        transform.position = glm::vec3(0.0f, 1.0f, 0.0f);\n        transform.scale = glm::vec3(1.0f);\n        \n        // Add components to entity\n        m_coordinator->AddComponent(character, controller);\n        m_coordinator->AddComponent(character, rigidbody);\n        m_coordinator->AddComponent(character, collider);\n        m_coordinator->AddComponent(character, transform);\n        \n        return character;\n    }\n    \n    void CreateWallSurfaces() {\n        // Create some wall surfaces for testing\n        \n        // Vertical wall (good for wall-running)\n        Core::Entity wall1 = m_coordinator->CreateEntity();\n        Physics::WallSurface wallSurface1;\n        wallSurface1.normal = glm::vec3(-1.0f, 0.0f, 0.0f);  // Points left\n        wallSurface1.position = glm::vec3(5.0f, 2.0f, 0.0f);\n        wallSurface1.friction = 0.1f;\n        wallSurface1.canWallRun = true;\n        m_wallRunSystem->RegisterWallSurface(wall1, wallSurface1);\n        \n        // Another vertical wall at an angle\n        Core::Entity wall2 = m_coordinator->CreateEntity();\n        Physics::WallSurface wallSurface2;\n        wallSurface2.normal = glm::normalize(glm::vec3(-0.7f, 0.0f, -0.7f));\n        wallSurface2.position = glm::vec3(-3.0f, 2.0f, -3.0f);\n        wallSurface2.friction = 0.15f;\n        wallSurface2.canWallRun = true;\n        m_wallRunSystem->RegisterWallSurface(wall2, wallSurface2);\n        \n        // Steep wall (too steep for wall-running)\n        Core::Entity wall3 = m_coordinator->CreateEntity();\n        Physics::WallSurface wallSurface3;\n        wallSurface3.normal = glm::normalize(glm::vec3(-0.3f, 0.9f, 0.0f));  // Very steep\n        wallSurface3.position = glm::vec3(0.0f, 3.0f, 5.0f);\n        wallSurface3.friction = 0.2f;\n        wallSurface3.canWallRun = false;  // Disabled for this demo\n        m_wallRunSystem->RegisterWallSurface(wall3, wallSurface3);\n    }\n    \n    void Update(float deltaTime) {\n        // Update all systems\n        m_wallRunSystem->Update(deltaTime);\n        m_gameFeelSystem->Update(deltaTime);\n        m_comboManager->Update(deltaTime);\n    }\n    \n    void DemonstrateWallRunningFeatures() {\n        std::cout << \"\\n=== Wall-Running System Features ===\\n\";\n        std::cout << \"✓ Momentum Conservation: Preserves 85% of tangential velocity\\n\";\n        std::cout << \"✓ Gravity Reduction: 20% gravity during wall-running\\n\";\n        std::cout << \"✓ Wall Detection: 1.8m range with multi-directional raycasting\\n\";\n        std::cout << \"✓ Angle Validation: Maximum 35° from vertical\\n\";\n        std::cout << \"✓ Speed Requirements: Minimum 6.0 m/s horizontal velocity\\n\";\n        std::cout << \"✓ Time Limits: Maximum 2.5 seconds per wall-run\\n\";\n        std::cout << \"✓ Wall Jumping: Enhanced momentum on wall-jump exit\\n\";\n        std::cout << \"✓ System Integration: Game feel effects and combat bonuses\\n\";\n        std::cout << \"✓ Debug Visualization: Real-time wall normals and velocity vectors\\n\";\n        \n        std::cout << \"\\n=== Momentum Conservation Physics ===\\n\";\n        std::cout << \"• Tangential velocity projected onto wall plane\\n\";\n        std::cout << \"• Smooth transition between movement states\\n\";\n        std::cout << \"• Wall-kick velocity for enhanced air mobility\\n\";\n        std::cout << \"• Preserved momentum applied on wall-run exit\\n\";\n        \n        std::cout << \"\\n=== Integration with Other Systems ===\\n\";\n        std::cout << \"• Screen shake on wall-run start/end\\n\";\n        std::cout << \"• Hit-stop effects for impact feedback\\n\";\n        std::cout << \"• Potential combo system integration\\n\";\n        std::cout << \"• Debug visualization system\\n\";\n    }\n};\n\n// Example usage function\nvoid RunWallRunningDemo() {\n    WallRunningIntegrationDemo demo;\n    \n    if (demo.Initialize()) {\n        std::cout << \"Wall-Running System initialized successfully!\\n\";\n        \n        // Create test entities\n        Core::Entity character = demo.CreateWallRunningCharacter();\n        demo.CreateWallSurfaces();\n        \n        // Show features\n        demo.DemonstrateWallRunningFeatures();\n        \n        // Simulate some updates\n        for (int i = 0; i < 5; ++i) {\n            demo.Update(0.016f); // ~60 FPS\n        }\n        \n        std::cout << \"\\nWall-Running demo completed!\\n\";\n    } else {\n        std::cout << \"Failed to initialize Wall-Running demo.\\n\";\n    }\n}\n\n/*\n * Key Features Implemented:\n * \n * 1. MOMENTUM CONSERVATION:\n *    - Preserves tangential velocity when transitioning to walls\n *    - Smooth interpolation between movement states\n *    - Configurable conservation factor (0.0 - 1.0)\n * \n * 2. REALISTIC WALL-RUNNING PHYSICS:\n *    - Reduced gravity during wall-running (configurable scale)\n *    - Minimum speed requirements for wall-run initiation\n *    - Maximum wall-run duration with timer system\n *    - Wall angle validation (prevents running on ceilings)\n * \n * 3. ADVANCED WALL DETECTION:\n *    - Multi-directional raycasting for wall detection\n *    - Configurable detection range\n *    - Wall surface registration system\n *    - Validation for suitable wall-running surfaces\n * \n * 4. ENHANCED MOBILITY:\n *    - Wall-jumping with momentum boost\n *    - Preserved momentum application on exit\n *    - Integration with character controller states\n *    - Support for air jumps and advanced movement\n * \n * 5. SYSTEM INTEGRATION:\n *    - Callbacks for wall-run start/end events\n *    - Integration with GameFeelSystem for screen shake\n *    - Debug visualization support\n *    - Extensible for combat system integration\n * \n * 6. CONFIGURABLE PARAMETERS:\n *    - All physics parameters are tweakable\n *    - Runtime configuration support\n *    - Debug visualization toggle\n *    - Performance optimization settings\n */", "search_start_line_number": 1}]
