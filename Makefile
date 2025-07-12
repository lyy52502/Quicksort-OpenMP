# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -march=native -ffast-math -funroll-loops -fopenmp
LDFLAGS = -lX11 -lm -fopenmp

# Target executable
TARGET = project

# Source files and Object files
SRCS = project.c 
OBJS = project.o 

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile project.c
project.o: project.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
