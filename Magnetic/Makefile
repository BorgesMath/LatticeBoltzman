CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
INCLUDES = -Iinclude

SRCS := $(wildcard src/*.cpp src/*.cc src/*.cxx src/*.c)
OBJS := $(patsubst src/%.cpp,build/%.o,$(patsubst src/%.cc,build/%.o,$(patsubst src/%.cxx,build/%.o,$(patsubst src/%.c,build/%.o,$(SRCS)))))
TARGET = lbm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS)

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.o: src/%.cc
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.o: src/%.cxx
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.o: src/%.c
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf build $(TARGET)

.PHONY: all clean
