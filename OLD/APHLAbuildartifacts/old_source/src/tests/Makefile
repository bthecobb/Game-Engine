# Simple Makefile for automated test system
CXX = g++
CXXFLAGS = -std=c++11 -Wall -I../../include -I../../external
LDFLAGS = 

# Source files
SOURCES = TestAutomatedSystem.cpp ../animation/SimpleAutomatedTest.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = test_automated.exe

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	del /Q *.o ..\animation\*.o $(TARGET) 2>nul

run: $(TARGET)
	$(TARGET)

.PHONY: all clean run
