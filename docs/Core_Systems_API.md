# Core Systems API Reference

This document provides a detailed API reference for the core systems of the AAA Game Engine.

## Table of Contents

1. [Coordinator](#coordinator)
2. [System Manager](#system-manager)
3. [Component Manager](#component-manager)
4. [Entity Manager](#entity-manager)
5. [Core Components](#core-components)

---

## Coordinator

The `Core::Coordinator` is the central hub of the ECS architecture. It manages all entities, components, and systems.

### Methods

- `void Initialize()`: Initializes all managers (component, entity, system).
- `Entity CreateEntity()`: Creates a new entity and returns its ID.
- `void DestroyEntity(Entity entity)`: Destroys an entity and all its components.
- `void AddComponent(Entity entity, T component)`: Adds a component to an entity.
- `void RemoveComponent<T>(Entity entity)`: Removes a component from an entity.
- `T& GetComponent<T>(Entity entity)`: Gets a reference to a component for an entity.
- `bool HasComponent<T>(Entity entity)`: Checks if an entity has a specific component.
- `std::shared_ptr<T> RegisterSystem<T>()`: Creates and registers a new system.
- `void SetSystemSignature<T>(Signature signature)`: Sets the component signature for a system.

### Usage Example

```cpp
auto coordinator = std::make_shared<Core::Coordinator>();
coordinator->Initialize();

auto renderSystem = coordinator->RegisterSystem<RenderSystem>();

Signature renderSignature;
renderSignature.set(coordinator->GetComponentType<TransformComponent>());
renderSignature.set(coordinator->GetComponentType<MeshComponent>());
coordinator->SetSystemSignature<RenderSystem>(renderSignature);
```

---

## System Manager

The `SystemManager` manages the registration, lookup, and execution of all systems.

### Methods

- `std::shared_ptr<T> RegisterSystem<T>()`: Registers a new system.
- `void SetSignature<T>(Signature signature)`: Sets the component signature for a system.
- `void EntityDestroyed(Entity entity)`: Notifies all systems that an entity was destroyed.
- `void EntitySignatureChanged(Entity entity, Signature newSignature)`: Updates systems when an entity's signature changes.

---

## Component Manager

The `ComponentManager` manages the storage and access of all components.

### Methods

- `void RegisterComponent<T>()`: Registers a new component type.
- `ComponentType GetComponentType<T>()`: Gets the type ID for a component.
- `void AddComponent(Entity entity, T component)`: Adds a component to an entity.
- `void RemoveComponent<T>(Entity entity)`: Removes a component from an entity.
- `T& GetComponent<T>(Entity entity)`: Gets a reference to a component for an entity.

---

## Entity Manager

The `EntityManager` manages the creation, destruction, and tracking of all entities.

### Methods

- `Entity CreateEntity()`: Creates a new entity.
- `void DestroyEntity(Entity entity)`: Destroys an entity.
- `void SetSignature(Entity entity, Signature signature)`: Sets the component signature for an entity.
- `Signature GetSignature(Entity entity)`: Gets the component signature for an entity.

---

## Core Components

### TransformComponent

Represents the position, rotation, and scale of an entity in 3D space.

**Fields**:
- `glm::vec3 position`: The position of the entity.
- `glm::vec3 rotation`: The rotation of the entity (Euler angles).
- `glm::vec3 scale`: The scale of the entity.

### TagComponent

A simple component for tagging entities with a string identifier.

**Fields**:
- `std::string tag`: The tag for the entity.

### RelationshipComponent

Defines parent-child relationships between entities.

**Fields**:
- `Entity parent`: The parent of this entity.
- `std::vector<Entity> children`: The children of this entity.

---

*This documentation is part of the AAA Game Engine API documentation series.*
