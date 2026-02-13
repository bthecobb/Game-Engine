#pragma once

#include "ECS_Types.h"
#include <array>
#include <unordered_map>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace CudaGame {
namespace Core {

// The ComponentArray is one of these for each component type.
// It packs the components tightly in memory.
// When an entity is destroyed, the component is moved to the end
// and the size is decremented, so it's always tightly packed.
class IComponentArray {
public:
    virtual ~IComponentArray() = default;
    virtual void EntityDestroyed(Entity entity) = 0;
};

template<typename T>
class ComponentArray : public IComponentArray {
public:
    void InsertData(Entity entity, T component) {
        assert(mEntityToIndexMap.find(entity) == mEntityToIndexMap.end() && "Component added to same entity more than once.");

        // Put new entry at end and update the maps
        size_t newIndex = mSize;
        mEntityToIndexMap[entity] = newIndex;
        mIndexToEntityMap[newIndex] = entity;
        mComponentArray[newIndex] = component;
        ++mSize;
    }

    void RemoveData(Entity entity) {
        assert(mEntityToIndexMap.find(entity) != mEntityToIndexMap.end() && "Removing non-existent component.");

        // Copy element at end into deleted element's place to maintain density
        size_t indexOfRemovedEntity = mEntityToIndexMap[entity];
        size_t indexOfLastElement = mSize - 1;
        mComponentArray[indexOfRemovedEntity] = mComponentArray[indexOfLastElement];

        // Update map to point to moved spot
        Entity entityOfLastElement = mIndexToEntityMap[indexOfLastElement];
        mEntityToIndexMap[entityOfLastElement] = indexOfRemovedEntity;
        mIndexToEntityMap[indexOfRemovedEntity] = entityOfLastElement;

        mEntityToIndexMap.erase(entity);
        mIndexToEntityMap.erase(indexOfLastElement);

        --mSize;
    }

    T& GetData(Entity entity) {
        if (mEntityToIndexMap.find(entity) == mEntityToIndexMap.end()) {
            std::cerr << "\n\n=== COMPONENT RETRIEVAL ERROR ===\n";
            std::cerr << "Entity " << entity << " does not have component type: " << typeid(T).name() << "\n";
            std::cerr << "Entities with this component: ";
            for (const auto& pair : mEntityToIndexMap) {
                std::cerr << pair.first << " ";
            }
            std::cerr << "\n";
            std::cerr << "Total components of this type: " << mSize << "\n";
            std::cerr << "\nThis error typically occurs when:\n";
            std::cerr << "1. A system's signature doesn't match the components it's trying to access\n";
            std::cerr << "2. An entity was added to a system but doesn't have all required components\n";
            std::cerr << "3. A component was removed but the system wasn't notified\n";
            
#ifdef _WIN32
            std::cerr << "\nStack trace:\n";
            void* stack[100];
            unsigned short frames;
            SYMBOL_INFO* symbol;
            HANDLE process = GetCurrentProcess();
            
            SymInitialize(process, NULL, TRUE);
            frames = CaptureStackBackTrace(0, 100, stack, NULL);
            symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
            symbol->MaxNameLen = 255;
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            
            for (unsigned short i = 0; i < frames; i++) {
                SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
                std::cerr << i << ": " << symbol->Name << " at 0x" << std::hex << symbol->Address << std::dec << "\n";
            }
            
            free(symbol);
#else
            std::cerr << "\nCall stack trace would be here if debugger was attached\n";
#endif
            std::cerr << "================================\n\n";
            std::cerr.flush();
            __debugbreak(); // This will break into debugger if attached
        }

        return mComponentArray[mEntityToIndexMap[entity]];
    }

    void EntityDestroyed(Entity entity) override {
        if (mEntityToIndexMap.find(entity) != mEntityToIndexMap.end()) {
            RemoveData(entity);
        }
    }

    bool HasComponent(Entity entity) const {
        return mEntityToIndexMap.find(entity) != mEntityToIndexMap.end();
    }

    size_t GetSize() const { return mSize; }
    
    // Iterator support for systems
    T* begin() { return &mComponentArray[0]; }
    T* end() { return &mComponentArray[mSize]; }
    const T* begin() const { return &mComponentArray[0]; }
    const T* end() const { return &mComponentArray[mSize]; }

private:
    // The packed array of components (of generic type T),
    // set to a specified maximum amount, matching the maximum number
    // of entities allowed to exist simultaneously, so that each entity
    // has a unique spot.
    std::array<T, MAX_ENTITIES> mComponentArray;

    // Map from an entity ID to an array index.
    std::unordered_map<Entity, size_t> mEntityToIndexMap;

    // Map from an array index to an entity ID.
    std::unordered_map<size_t, Entity> mIndexToEntityMap;

    // Total size of valid entries in the array.
    size_t mSize = 0;
};

// The ComponentManager is in charge of talking to all the different ComponentArrays
// when a component needs to be added or removed.
class ComponentManager {
public:
    template<typename T>
    void RegisterComponent() {
        const char* typeName = typeid(T).name();

        assert(mComponentTypes.find(typeName) == mComponentTypes.end() && "Registering component type more than once.");

        // Add this component type to the component type map
        mComponentTypes.insert({typeName, mNextComponentType});

        // Create a ComponentArray pointer and add it to the component arrays map
        mComponentArrays.insert({typeName, std::make_shared<ComponentArray<T>>()});

        // Increment the value so that the next component registered will be different
        ++mNextComponentType;
    }

    template<typename T>
    ComponentType GetComponentType() {
        const char* typeName = typeid(T).name();

        assert(mComponentTypes.find(typeName) != mComponentTypes.end() && "Component not registered before use.");

        return mComponentTypes[typeName];
    }

    template<typename T>
    void AddComponent(Entity entity, T component) {
        GetComponentArray<T>()->InsertData(entity, component);
    }

    template<typename T>
    void RemoveComponent(Entity entity) {
        GetComponentArray<T>()->RemoveData(entity);
    }

    template<typename T>
    T& GetComponent(Entity entity) {
        return GetComponentArray<T>()->GetData(entity);
    }

    template<typename T>
    bool HasComponent(Entity entity) {
        return GetComponentArray<T>()->HasComponent(entity);
    }

    void EntityDestroyed(Entity entity) {
        // Notify each component array that an entity has been destroyed
        // If it has a component for that entity, it will remove it
        for (auto const& pair : mComponentArrays) {
            auto const& component = pair.second;
            component->EntityDestroyed(entity);
        }
    }

    template<typename T>
    std::shared_ptr<ComponentArray<T>> GetComponentArray() {
        const char* typeName = typeid(T).name();

        assert(mComponentTypes.find(typeName) != mComponentTypes.end() && "Component not registered before use.");

        return std::static_pointer_cast<ComponentArray<T>>(mComponentArrays[typeName]);
    }

private:
    // Map from type string to a component type
    std::unordered_map<std::string, ComponentType> mComponentTypes{};

    // Map from type string to a component array
    std::unordered_map<std::string, std::shared_ptr<IComponentArray>> mComponentArrays{};

    // The component type to be assigned to the next registered component - starting at 0
    ComponentType mNextComponentType{};
};

} // namespace Core
} // namespace CudaGame
