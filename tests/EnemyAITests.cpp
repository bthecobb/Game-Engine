#include "Testing/TestFramework.h"
#include "Core/Coordinator.h"
#include "Gameplay/EnemyAISystem.h"
#include "Gameplay/EnemyComponents.h"
#include "Gameplay/PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Rendering/RenderComponents.h"

using namespace CudaGame::Testing;
using namespace CudaGame::Core;
using namespace CudaGame::Gameplay;
using namespace CudaGame::Physics;
using namespace CudaGame::Rendering;

std::shared_ptr<TestSuite> CreateEnemyAITestSuite()
{
    auto suite = std::make_shared<TestSuite>("Enemy AI");

    suite->AddTest("NoPlayer_RemainsPatrol", [](){
        auto& coordinator = Coordinator::GetInstance();
        coordinator.Initialize();

        // Register components
        coordinator.RegisterComponent<EnemyAIComponent>();
        coordinator.RegisterComponent<EnemyCombatComponent>();
        coordinator.RegisterComponent<EnemyMovementComponent>();
        coordinator.RegisterComponent<PlayerMovementComponent>();
        coordinator.RegisterComponent<RigidbodyComponent>();
        coordinator.RegisterComponent<TransformComponent>();

        // Register system and set signature
        auto enemyAISystem = coordinator.RegisterSystem<EnemyAISystem>();
Signature sig;
        sig.set(coordinator.GetComponentType<EnemyAIComponent>());
        sig.set(coordinator.GetComponentType<EnemyCombatComponent>());
        sig.set(coordinator.GetComponentType<EnemyMovementComponent>());
        sig.set(coordinator.GetComponentType<RigidbodyComponent>());
        sig.set(coordinator.GetComponentType<TransformComponent>());
        coordinator.SetSystemSignature<EnemyAISystem>(sig);
        enemyAISystem->Initialize();

        // Create one enemy
        Entity e = coordinator.CreateEntity();
EnemyAIComponent ai; ai.aiState = AIState::PATROL; ai.detectionRange = 50.0f; ai.attackRange = 2.0f; ai.visionAngle = 120.0f; ai.facingDirection = {0,0,-1};
EnemyCombatComponent combat; combat.attackCooldown = 1.0f;
        EnemyMovementComponent move; move.speed = 5.0f;
        RigidbodyComponent rb; rb.setMass(50.0f);
        TransformComponent tr; tr.position = {0,0,10};
        coordinator.AddComponent(e, ai);
        coordinator.AddComponent(e, combat);
        coordinator.AddComponent(e, move);
        coordinator.AddComponent(e, rb);
        coordinator.AddComponent(e, tr);

        // Update without binding player; AI should remain PATROL
        enemyAISystem->Update(0.016f);
        auto& aiAfter = coordinator.GetComponent<EnemyAIComponent>(e);
ASSERT_EQ((int)aiAfter.aiState, (int)AIState::PATROL);
    });

    suite->AddTest("WithPlayer_TransitionsToChase", [](){
        auto& coordinator = Coordinator::GetInstance();
        coordinator.Initialize();

        // Register components
        coordinator.RegisterComponent<EnemyAIComponent>();
        coordinator.RegisterComponent<EnemyCombatComponent>();
        coordinator.RegisterComponent<EnemyMovementComponent>();
        coordinator.RegisterComponent<PlayerMovementComponent>();
        coordinator.RegisterComponent<RigidbodyComponent>();
        coordinator.RegisterComponent<TransformComponent>();

        // Register system and set signature
        auto enemyAISystem = coordinator.RegisterSystem<EnemyAISystem>();
Signature sig;
        sig.set(coordinator.GetComponentType<EnemyAIComponent>());
        sig.set(coordinator.GetComponentType<EnemyCombatComponent>());
        sig.set(coordinator.GetComponentType<EnemyMovementComponent>());
        sig.set(coordinator.GetComponentType<RigidbodyComponent>());
        sig.set(coordinator.GetComponentType<TransformComponent>());
        coordinator.SetSystemSignature<EnemyAISystem>(sig);
        enemyAISystem->Initialize();

        // Create player entity at origin
        Entity player = coordinator.CreateEntity();
        coordinator.AddComponent(player, PlayerMovementComponent{});
        TransformComponent playerTr; playerTr.position = {0,0,0};
        coordinator.AddComponent(player, playerTr);

        // Bind player to system
        enemyAISystem->SetPlayerEntity(player);

        // Create enemy 10 units on +Z, facing -Z
        Entity e = coordinator.CreateEntity();
EnemyAIComponent ai; ai.aiState = AIState::PATROL; ai.detectionRange = 50.0f; ai.attackRange = 2.0f; ai.visionAngle = 180.0f; ai.facingDirection = {0,0,-1};
EnemyCombatComponent combat; combat.attackCooldown = 1.0f;
        EnemyMovementComponent move; move.speed = 5.0f;
        RigidbodyComponent rb; rb.setMass(50.0f);
        TransformComponent tr; tr.position = {0,0,10};
        coordinator.AddComponent(e, ai);
        coordinator.AddComponent(e, combat);
        coordinator.AddComponent(e, move);
        coordinator.AddComponent(e, rb);
        coordinator.AddComponent(e, tr);

        enemyAISystem->Update(0.016f);

        auto& aiAfter = coordinator.GetComponent<EnemyAIComponent>(e);
ASSERT_EQ((int)aiAfter.aiState, (int)AIState::CHASE);
    });

    return suite;
}
