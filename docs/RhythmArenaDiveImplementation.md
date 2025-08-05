# Quick Dive Implementation for RhythmArenaDemo

Add these specific code blocks to your existing RhythmArenaDemo.cpp:

## 1. Add to Player struct (after line 130):
```cpp
// Aerial combat
bool canDive = false;
bool isDiving = false;
float diveSpeed = 40.0f;
float diveBounceHeight = 27.0f; // Higher than jumpForce (18.0f)
glm::vec3 diveDirection;
```

## 2. In processInput() function, modify jump code:
```cpp
// Jump mechanics - add dive capability
if ((GetAsyncKeyState(VK_SPACE) & 0x8000) && !wasSpacePressed) {
    if (isGrounded) {
        player.velocity.y = jumpVelocity;
        isGrounded = false;
        jumpCount = 1;
        player.canDive = true; // NEW: Enable dive after first jump
    } else if (jumpCount < maxJumps && !player.isDiving) {
        if (player.canDive) {
            // NEW: Start dive instead of double jump
            player.isDiving = true;
            player.diveDirection = glm::normalize(glm::vec3(player.velocity.x, -1.0f, player.velocity.z));
            player.velocity = player.diveDirection * player.diveSpeed;
            player.canDive = false;
            std::cout << "DIVING!" << std::endl;
        } else {
            // Regular double jump
            player.velocity.y = jumpVelocity * 0.8f;
            jumpCount++;
            player.rotation.x += 360.0f;
        }
    }
}
```

## 3. In updatePlayer() function, add after ground collision check:
```cpp
// Check for dive landing
if (player.isDiving && player.onGround) {
    // Dive bounce - launch higher!
    player.velocity.y = player.diveBounceHeight;
    player.onGround = false;
    player.isDiving = false;
    player.canDive = true; // Can dive again after bounce
    std::cout << "DIVE BOUNCE!" << std::endl;
    
    // Damage nearby enemies
    for (auto& enemy : enemies) {
        float dist = glm::length(enemy.position - player.position);
        if (dist < 3.0f) {
            enemy.health -= 30.0f;
            enemy.velocity.y = 10.0f; // Launch enemy
            std::cout << "Dive impact damage!" << std::endl;
        }
    }
}
```

## 4. In renderPlayer() function, add rotation for dive:
```cpp
// Add dive rotation
if (player.isDiving) {
    glRotatef(90, 1, 0, 0); // Head-first dive
    // Optional: Add orange tint
    glColor3f(1.0f, 0.7f, 0.3f);
}
```

## 5. Reset dive state when landing normally:
In updatePlayer(), where player lands on ground:
```cpp
if (player.onGround) {
    player.canDoubleJump = true;
    player.hasDoubleJumped = false;
    player.isDiving = false; // NEW
    player.canDive = false;  // NEW
}
```

## Testing the Feature:
1. Jump once (Space)
2. Press Space again while in air to dive
3. Land to get massive upward bounce
4. Can dive again from the bounce!

## Visual Enhancements (Optional):
Add particle trail during dive:
```cpp
if (player.isDiving) {
    // Simple trail effect
    glBegin(GL_POINTS);
    glPointSize(5.0f);
    glColor4f(1.0f, 0.5f, 0.0f, 0.5f);
    for (int i = 0; i < 5; ++i) {
        glVertex3f(
            player.position.x + (rand() % 10 - 5) * 0.1f,
            player.position.y + i * 0.5f,
            player.position.z + (rand() % 10 - 5) * 0.1f
        );
    }
    glEnd();
}
```

This implementation adds the core dive mechanic without requiring external headers!
