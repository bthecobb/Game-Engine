#pragma once

#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <vector>

namespace CudaGame {
namespace Physics {

// Axis-Aligned Bounding Box
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    
    AABB() = default;
    AABB(const glm::vec3& center, const glm::vec3& halfExtents) 
        : min(center - halfExtents), max(center + halfExtents) {}
    
    glm::vec3 getCenter() const { return (min + max) * 0.5f; }
    glm::vec3 getExtents() const { return max - min; }
    glm::vec3 getHalfExtents() const { return (max - min) * 0.5f; }
    
    bool contains(const glm::vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
    
    bool intersects(const AABB& other) const {
        return !(max.x < other.min.x || min.x > other.max.x ||
                 max.y < other.min.y || min.y > other.max.y ||
                 max.z < other.min.z || min.z > other.max.z);
    }
    
    AABB merge(const AABB& other) const {
        AABB result;
        result.min = glm::min(min, other.min);
        result.max = glm::max(max, other.max);
        return result;
    }
    
    float getSurfaceArea() const {
        glm::vec3 extents = getExtents();
        return 2.0f * (extents.x * extents.y + extents.x * extents.z + extents.y * extents.z);
    }
};

// Oriented Bounding Box
struct OBB {
    glm::vec3 center;
    glm::vec3 axes[3];      // Local coordinate axes
    glm::vec3 halfExtents;  // Half-widths along each axis
    
    OBB() = default;
    OBB(const glm::vec3& center, const glm::vec3& halfExtents, const glm::mat3& rotation = glm::mat3(1.0f)) 
        : center(center), halfExtents(halfExtents) {
        axes[0] = rotation[0];
        axes[1] = rotation[1];
        axes[2] = rotation[2];
    }
    
    bool intersects(const OBB& other) const;
    AABB getAABB() const;
};

// Sphere for collision detection
struct Sphere {
    glm::vec3 center;
    float radius;
    
    Sphere() = default;
    Sphere(const glm::vec3& center, float radius) : center(center), radius(radius) {}
    
    bool contains(const glm::vec3& point) const {
        return glm::distance(center, point) <= radius;
    }
    
    bool intersects(const Sphere& other) const {
        float distance = glm::distance(center, other.center);
        return distance <= (radius + other.radius);
    }
    
    bool intersects(const AABB& aabb) const;
};

// Capsule for character controllers
struct Capsule {
    glm::vec3 start;
    glm::vec3 end;
    float radius;
    
    Capsule() = default;
    Capsule(const glm::vec3& start, const glm::vec3& end, float radius) 
        : start(start), end(end), radius(radius) {}
    
    glm::vec3 getCenter() const { return (start + end) * 0.5f; }
    float getHeight() const { return glm::distance(start, end); }
    
    bool intersects(const Sphere& sphere) const;
    bool intersects(const AABB& aabb) const;
    bool intersects(const Capsule& other) const;
};

// Contact information for collision resolution
struct ContactPoint {
    glm::vec3 position;     // World space contact position
    glm::vec3 normal;       // Contact normal (from A to B)
    float penetration;      // Penetration depth
    Core::Entity entityA;
    Core::Entity entityB;
};

// Collision detection functions
namespace CollisionDetection {
    // AABB vs AABB
    bool testAABBvsAABB(const AABB& a, const AABB& b);
    bool testAABBvsAABB(const AABB& a, const AABB& b, ContactPoint& contact);
    
    // Sphere vs Sphere
    bool testSphereVsSphere(const Sphere& a, const Sphere& b);
    bool testSphereVsSphere(const Sphere& a, const Sphere& b, ContactPoint& contact);
    
    // Sphere vs AABB
    bool testSphereVsAABB(const Sphere& sphere, const AABB& aabb);
    bool testSphereVsAABB(const Sphere& sphere, const AABB& aabb, ContactPoint& contact);
    
    // Capsule vs Sphere
    bool testCapsuleVsSphere(const Capsule& capsule, const Sphere& sphere);
    bool testCapsuleVsSphere(const Capsule& capsule, const Sphere& sphere, ContactPoint& contact);
    
    // Capsule vs AABB
    bool testCapsuleVsAABB(const Capsule& capsule, const AABB& aabb);
    bool testCapsuleVsAABB(const Capsule& capsule, const AABB& aabb, ContactPoint& contact);
    
    // OBB vs OBB
    bool testOBBvsOBB(const OBB& a, const OBB& b);
    bool testOBBvsOBB(const OBB& a, const OBB& b, ContactPoint& contact);
    
    // Ray casting
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
        float maxDistance = 1000.0f;
    };
    
    struct RaycastHit {
        bool hit = false;
        glm::vec3 point;
        glm::vec3 normal;
        float distance;
        Core::Entity entity;
    };
    
    bool raycastAABB(const Ray& ray, const AABB& aabb, RaycastHit& hit);
    bool raycastSphere(const Ray& ray, const Sphere& sphere, RaycastHit& hit);
    bool raycastOBB(const Ray& ray, const OBB& obb, RaycastHit& hit);
    
    // Utility functions
    glm::vec3 closestPointOnAABB(const glm::vec3& point, const AABB& aabb);
    glm::vec3 closestPointOnOBB(const glm::vec3& point, const OBB& obb);
    glm::vec3 closestPointOnLineSegment(const glm::vec3& point, const glm::vec3& a, const glm::vec3& b);
    
    // Sweep tests for continuous collision detection
    bool sweepSphereVsAABB(const Sphere& sphere, const glm::vec3& velocity, const AABB& aabb, float& t);
    bool sweepAABBvsAABB(const AABB& a, const glm::vec3& velocity, const AABB& b, float& t);
}

} // namespace Physics
} // namespace CudaGame
