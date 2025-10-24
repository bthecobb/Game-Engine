#pragma once

#include <cstdint>
#include <bitset>

namespace CudaGame {
namespace Core {

// Defines an entity as a simple ID. The entity itself holds no data.
using Entity = std::uint32_t;
const Entity MAX_ENTITIES = 100000;  // Increased for AAA-scale entity counts

// Defines a component type ID.
using ComponentType = std::uint8_t;
const ComponentType MAX_COMPONENTS = 32;

// A signature is a bitset that defines which components an entity has.
// Each bit corresponds to a component type.
using Signature = std::bitset<MAX_COMPONENTS>;

} // namespace Core
} // namespace CudaGame

