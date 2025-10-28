#include <iostream>
#include "GameEngine.h"

int main() {
    std::cout << "Starting CUDA Particle Game..." << std::endl;
    
    GameEngine engine;
    
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize game engine!" << std::endl;
        return -1;
    }
    
    std::cout << "Game engine initialized successfully!" << std::endl;
    std::cout << "Click and drag to create particles!" << std::endl;
    std::cout << "Press ESC to exit." << std::endl;
    
    engine.run();
    
    engine.shutdown();
    
    std::cout << "Game shutdown complete." << std::endl;
    return 0;
}
