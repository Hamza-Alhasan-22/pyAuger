# Makefile for the standalone C++ Auger calculators.
#
# On NERSC Perlmutter:
#   module load gcc   # (or PrgEnv-gnu)
#   make
#
# On a generic Linux box with g++:
#   make
#
# To override compiler:
#   make CXX=icpx

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra
LDFLAGS  ?= -fopenmp

TARGETS = matrix_element_calc pair_generation_calc

.PHONY: all clean

all: $(TARGETS)

matrix_element_calc: matrix_element_calc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

pair_generation_calc: pair_generation_calc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
