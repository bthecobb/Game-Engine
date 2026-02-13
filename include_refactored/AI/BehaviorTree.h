#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>

namespace CudaGame {
namespace AI {

enum class NodeState {
    RUNNING,
    SUCCESS,
    FAILURE
};

// Forward declaration
class BehaviorTree;

/**
 * [AAA Pattern] Behavior Tree Node
 * Base class for all logic nodes.
 */
class Node {
public:
    virtual ~Node() = default;
    virtual NodeState Tick(Core::Entity entity, float deltaTime) = 0;
};

/**
 * [AAA Pattern] Composite Node
 * Base for control flow nodes (Sequence, Selector).
 */
class Composite : public Node {
public:
    void AddChild(std::shared_ptr<Node> child) { m_children.push_back(child); }
protected:
    std::vector<std::shared_ptr<Node>> m_children;
};

/**
 * Selector: Returns SUCCESS if *any* child succeeds. (OR)
 */
class Selector : public Composite {
public:
    NodeState Tick(Core::Entity entity, float deltaTime) override {
        for (auto& child : m_children) {
            NodeState state = child->Tick(entity, deltaTime);
            if (state == NodeState::SUCCESS) return NodeState::SUCCESS;
            if (state == NodeState::RUNNING) return NodeState::RUNNING;
        }
        return NodeState::FAILURE;
    }
};

/**
 * Sequence: Returns SUCCESS if *all* children succeed. (AND)
 */
class Sequence : public Composite {
public:
    NodeState Tick(Core::Entity entity, float deltaTime) override {
        for (auto& child : m_children) {
            NodeState state = child->Tick(entity, deltaTime);
            if (state == NodeState::FAILURE) return NodeState::FAILURE;
            if (state == NodeState::RUNNING) return NodeState::RUNNING;
        }
        return NodeState::SUCCESS;
    }
};

/**
 * Action Leaf: Executes a lambda or function.
 */
class ActionNode : public Node {
public:
    using ActionFunc = std::function<NodeState(Core::Entity, float)>;
    
    ActionNode(ActionFunc func) : m_action(func) {}
    
    NodeState Tick(Core::Entity entity, float deltaTime) override {
        return m_action(entity, deltaTime);
    }
    
private:
    ActionFunc m_action;
};

/**
 * Decorator: Condition check.
 */
class ConditionNode : public Node {
public:
    using ConditionFunc = std::function<bool(Core::Entity)>;
    
    ConditionNode(ConditionFunc cond) : m_condition(cond) {}
    
    NodeState Tick(Core::Entity entity, float deltaTime) override {
        if (m_condition(entity)) return NodeState::SUCCESS;
        return NodeState::FAILURE;
    }
    
private:
    ConditionFunc m_condition;
};

} // namespace AI
} // namespace CudaGame
