#pragma once

// PhysX Static Linking Compatibility Header
// 
// This header provides stub implementations for deprecated PhysX registration functions
// that don't exist in PhysX 5.x static libraries but are still referenced by the inline
// PxCreatePhysics() function.
//
// For static linking, these registration functions are not needed because all modules
// are already compiled into the static libraries.

#ifdef PX_PHYSX_STATIC_LIB

#include <PxPhysicsAPI.h>

// Provide stub implementations for deprecated registration functions
// These do nothing because with static linking, the modules are already linked in
inline void PxRegisterArticulationsReducedCoordinate(physx::PxPhysics& /*physics*/) {
    // No-op: Articulations are already statically linked
}

inline void PxRegisterHeightFields(physx::PxPhysics& /*physics*/) {
    // No-op: HeightFields are already statically linked
}

// Override the inline PxCreatePhysics to use our stub implementations
#undef PxCreatePhysics

inline physx::PxPhysics* PxCreatePhysics(physx::PxU32 version,
                                         physx::PxFoundation& foundation,
                                         const physx::PxTolerancesScale& scale,
                                         bool trackOutstandingAllocations,
                                         physx::PxPvd* pvd,
                                         physx::PxOmniPvd* omniPvd)
{
    physx::PxPhysics* physics = PxCreateBasePhysics(version, foundation, scale, trackOutstandingAllocations, pvd, omniPvd);
    if (!physics)
        return nullptr;
    
    // Call our stub implementations (which do nothing)
    PxRegisterArticulationsReducedCoordinate(*physics);
    PxRegisterHeightFields(*physics);
    
    return physics;
}

#endif // PX_PHYSX_STATIC_LIB
