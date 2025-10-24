#include "Testing/TestFramework.h"
#include "Core/Coordinator.h"
#include "Rendering/RenderComponents.h"
#include "Physics/PhysicsComponents.h"
#include <memory>

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Rendering;
using namespace CudaGame::Physics;

// Test fixtures
class CoreSystemsTestSuite {
private:
    std::shared_ptr<Coordinator> coordinator;

public:
    void SetUp() {
        coordinator = std::make_shared<Coordinator>();
        coordinator->Initialize();
        
        // Register components for testing
        coordinator->RegisterComponent<TransformComponent>();
        coordinator->RegisterComponent<MeshComponent>();
        coordinator->RegisterComponent<RigidbodyComponent>();
    }

    void TearDown() {
        coordinator.reset();
    }

    // Entity Management Tests
    void TestEntityCreation() {
        Entity entity1 = coordinator->CreateEntity();
        Entity entity2 = coordinator->CreateEntity();
        
        ASSERT_NE(entity1, entity2);
        ASSERT_TRUE(entity1 >= 0);
        ASSERT_TRUE(entity2 >= 0);
    }

    void TestEntityDestruction() {
        Entity entity = coordinator->CreateEntity();
        coordinator->AddComponent(entity, TransformComponent{});
        
        ASSERT_TRUE(coordinator->HasComponent<TransformComponent>(entity));
        
        coordinator->DestroyEntity(entity);
        
        // After destruction, the entity should not have components
        ASSERT_FALSE(coordinator->HasComponent<TransformComponent>(entity));
    }

    // Component Management Tests
    void TestComponentAddition() {
        Entity entity = coordinator->CreateEntity();
        
        TransformComponent transform;
        transform.position = {1.0f, 2.0f, 3.0f};
        transform.rotation = {0.0f, 45.0f, 0.0f};
        transform.scale = {2.0f, 2.0f, 2.0f};
        
        coordinator->AddComponent(entity, transform);
        
        ASSERT_TRUE(coordinator->HasComponent<TransformComponent>(entity));
        
        const auto& retrievedTransform = coordinator->GetComponent<TransformComponent>(entity);
        ASSERT_NEAR(retrievedTransform.position.x, 1.0f, 0.001f);
        ASSERT_NEAR(retrievedTransform.position.y, 2.0f, 0.001f);
        ASSERT_NEAR(retrievedTransform.position.z, 3.0f, 0.001f);
    }

    void TestComponentRemoval() {
        Entity entity = coordinator->CreateEntity();
        coordinator->AddComponent(entity, TransformComponent{});
        
        ASSERT_TRUE(coordinator->HasComponent<TransformComponent>(entity));
        
        coordinator->RemoveComponent<TransformComponent>(entity);
        
        ASSERT_FALSE(coordinator->HasComponent<TransformComponent>(entity));
    }

    void TestMultipleComponents() {
        Entity entity = coordinator->CreateEntity();
        
        // Add multiple components
        coordinator->AddComponent(entity, TransformComponent{});
        coordinator->AddComponent(entity, MeshComponent{"test_mesh.obj"});
        
        ASSERT_TRUE(coordinator->HasComponent<TransformComponent>(entity));
        ASSERT_TRUE(coordinator->HasComponent<MeshComponent>(entity));
        ASSERT_FALSE(coordinator->HasComponent<RigidbodyComponent>(entity));
    }

    // Physics Component Tests
    void TestRigidbodyComponent() {
        Entity entity = coordinator->CreateEntity();
        
        RigidbodyComponent rb;
        rb.setMass(5.0f);
        rb.velocity = {1.0f, 0.0f, 0.0f};
        rb.restitution = 0.8f;
        
        coordinator->AddComponent(entity, rb);
        
        const auto& retrievedRb = coordinator->GetComponent<RigidbodyComponent>(entity);
        ASSERT_NEAR(retrievedRb.mass, 5.0f, 0.001f);
        ASSERT_NEAR(retrievedRb.inverseMass, 0.2f, 0.001f);
        ASSERT_NEAR(retrievedRb.velocity.x, 1.0f, 0.001f);
        ASSERT_NEAR(retrievedRb.restitution, 0.8f, 0.001f);
    }

    // Transform Component Matrix Tests
    void TestTransformMatrix() {
        Entity entity = coordinator->CreateEntity();
        
        TransformComponent transform;
        transform.position = {1.0f, 2.0f, 3.0f};
        transform.rotation = {0.0f, 0.0f, 0.0f}; // No rotation for simplicity
        transform.scale = {1.0f, 1.0f, 1.0f}; // Unit scale
        
        coordinator->AddComponent(entity, transform);
        
        const auto& retrievedTransform = coordinator->GetComponent<TransformComponent>(entity);
        glm::mat4 matrix = retrievedTransform.getMatrix();
        
        // Check that the translation component is correct
        ASSERT_NEAR(matrix[3][0], 1.0f, 0.001f); // x translation
        ASSERT_NEAR(matrix[3][1], 2.0f, 0.001f); // y translation
        ASSERT_NEAR(matrix[3][2], 3.0f, 0.001f); // z translation
    }

    // Performance Tests
    void TestMassEntityCreation() {
        const int entityCount = 10000;
        std::vector<Entity> entities;
        entities.reserve(entityCount);
        
        {
            BENCHMARK_START();
            for (int i = 0; i < entityCount; ++i) {
                Entity entity = coordinator->CreateEntity();
                entities.push_back(entity);
            }
            BENCHMARK_END("Entity Creation (10k entities)");
        }
        
        ASSERT_EQ(entities.size(), entityCount);
        
        // Cleanup
        {
            BENCHMARK_START();
            for (Entity entity : entities) {
                coordinator->DestroyEntity(entity);
            }
            BENCHMARK_END("Entity Destruction (10k entities)");
        }
    }

    void TestMassComponentOperations() {
        const int entityCount = 1000;
        std::vector<Entity> entities;
        entities.reserve(entityCount);
        
        // Create entities
        for (int i = 0; i < entityCount; ++i) {
            entities.push_back(coordinator->CreateEntity());
        }
        
        // Add components
        {
            BENCHMARK_START();
            for (Entity entity : entities) {
                coordinator->AddComponent(entity, TransformComponent{});
                coordinator->AddComponent(entity, MeshComponent{"test.obj"});
            }
            BENCHMARK_END("Component Addition (1k entities x 2 components)");
        }
        
        // Access components
        {
            BENCHMARK_START();
            for (Entity entity : entities) {
                auto& transform = coordinator->GetComponent<TransformComponent>(entity);
                transform.position.x += 1.0f;
            }
            BENCHMARK_END("Component Access (1k entities)");
        }
        
        // Remove components
        {
            BENCHMARK_START();
            for (Entity entity : entities) {
                coordinator->RemoveComponent<MeshComponent>(entity);
            }
            BENCHMARK_END("Component Removal (1k entities)");
        }
        
        // Cleanup
        for (Entity entity : entities) {
            coordinator->DestroyEntity(entity);
        }
    }
};

// Function to create and register the test suite
std::shared_ptr<TestSuite> CreateCoreSystemsTestSuite() {
    auto suite = std::make_shared<TestSuite>("Core Systems");
    auto fixture = std::make_shared<CoreSystemsTestSuite>();
    
    // Entity management tests
    suite->AddTest("Entity Creation", [fixture]() {
        fixture->SetUp();
        fixture->TestEntityCreation();
        fixture->TearDown();
    });
    
    suite->AddTest("Entity Destruction", [fixture]() {
        fixture->SetUp();
        fixture->TestEntityDestruction();
        fixture->TearDown();
    });
    
    // Component management tests
    suite->AddTest("Component Addition", [fixture]() {
        fixture->SetUp();
        fixture->TestComponentAddition();
        fixture->TearDown();
    });
    
    suite->AddTest("Component Removal", [fixture]() {
        fixture->SetUp();
        fixture->TestComponentRemoval();
        fixture->TearDown();
    });
    
    suite->AddTest("Multiple Components", [fixture]() {
        fixture->SetUp();
        fixture->TestMultipleComponents();
        fixture->TearDown();
    });
    
    // Physics component tests
    suite->AddTest("Rigidbody Component", [fixture]() {
        fixture->SetUp();
        fixture->TestRigidbodyComponent();
        fixture->TearDown();
    });
    
    // Transform tests
    suite->AddTest("Transform Matrix", [fixture]() {
        fixture->SetUp();
        fixture->TestTransformMatrix();
        fixture->TearDown();
    });
    
    // Performance tests
    suite->AddTest("Mass Entity Creation", [fixture]() {
        fixture->SetUp();
        fixture->TestMassEntityCreation();
        fixture->TearDown();
    });
    
    suite->AddTest("Mass Component Operations", [fixture]() {
        fixture->SetUp();
        fixture->TestMassComponentOperations();
        fixture->TearDown();
    });
    
    return suite;
}
