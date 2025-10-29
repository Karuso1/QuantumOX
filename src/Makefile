# QuantumOX, a Tic Tac Toe engine supporting UTTTI.
# Copyright (C) 2025 Kartik Pawar
#
# QuantumOX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QuantumOX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

### ==========================================================================
### Section 1. Basic configuration
### ==========================================================================

KERNEL := $(shell uname -s)
ifeq ($(KERNEL),Linux)
	OS := $(shell uname -o)
endif

ifeq ($(OS),Windows_NT)
	target_windows = yes
endif

ifeq ($(target_windows),yes)
	EXE = quantumox.exe
else
	EXE = quantumox
endif

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

### Sources / headers
SRCS = main.cpp board.cpp engine.cpp search.cpp options.cpp
HEADERS = board.h constants.h engine.h search.h options.h utils.h
OBJS = $(SRCS:.cpp=.o)

### Compiler defaults
CXX ?= g++
CXXFLAGS ?= -Wall -Wextra -Wshadow -std=c++20
LDFLAGS ?=

### Default optimization / debug flags
ifeq ($(DEBUG),yes)
	CXXFLAGS += -g
else
	CXXFLAGS += -O3 -DNDEBUG
endif

### ==========================================================================
### Section 2. Architecture detection
### ==========================================================================
ifeq ($(ARCH),)
	# Call the user-provided natives.sh to detect a native architecture string.
	# The script should print the architecture name as the first whitespace-separated field.
	ifneq (,$(wildcard ../scripts/natives.sh))
		override ARCH := $(shell $(SHELL) ../scripts/natives.sh | cut -d " " -f 1)
	else
		ARCH := native
	endif
endif

ifeq ($(ARCH),native)
	# If script couldn't determine arch, fall back to the host's arch name
	ARCH := $(shell uname -m 2>/dev/null || echo native)
endif

### Simple arch flags
ifneq (,$(findstring x86,$(ARCH)))
	CXXFLAGS += -m64
endif

### ==========================================================================
### Section 3. Compiler selection and platform tweaks
### ==========================================================================
ifeq ($(COMP),)
	COMP = gcc
endif

ifeq ($(COMP),gcc)
	# use default g++
	CXX ?= g++
endif

ifeq ($(COMP),clang)
	CXX ?= clang++
	CXXFLAGS += -fno-exceptions
endif

# On non-Windows use pthreads
ifneq ($(findstring mingw,$(COMP)),mingw)
	CXXFLAGS += -DUSE_PTHREADS
	ifneq ($(OS),Android)
		LDFLAGS += -lpthread
	endif
endif

### ==========================================================================
### Section 4. Public targets
### ==========================================================================
.PHONY: help build all clean objclean strip install format

help:
	@echo "" && \
	echo "QuantumOX Build System" && \
	echo "-----------------------" && \
	echo "" && \
	echo "Usage:" && \
	echo "  make [target] [OPTIONS]" && \
	echo "" && \
	echo "Targets:" && \
	echo "  build           > Build optimized QuantumOX binary (default)" && \
	echo "  all             > Alias for build" && \
	echo "  clean           > Remove object files and build outputs" && \
	echo "  strip           > Strip debugging symbols from the binary" && \
	echo "  install         > Install QuantumOX to PREFIX (default /usr/local)" && \
	echo "  format          > Run clang-format on source files (if available)" && \
	echo "  help            > Show this message" && \
	echo "" && \
	echo "Supported architectures:" && \
	echo "  native                  > Detect best settings for host system (default)" && \
	echo "  x86-64-avx512icl        > x86_64 with AVX512 + VNNI (Intel Ice Lake / AMD Zen 4)" && \
	echo "  x86-64-vnni512          > x86_64 with AVX512 VNNI" && \
	echo "  x86-64-avx512           > x86_64 with AVX512 support" && \
	echo "  x86-64-avxvnni          > x86_64 with AVX-VNNI (256-bit)" && \
	echo "  x86-64-bmi2             > x86_64 with BMI2 instructions" && \
	echo "  x86-64-avx2             > x86_64 with AVX2 support" && \
	echo "  x86-64-sse41-popcnt     > x86_64 with SSE4.1 + POPCNT" && \
	echo "  x86-64                  > Generic 64-bit build" && \
	echo "  x86-32                  > Generic 32-bit build" && \
	echo "  ppc-64-vsx              > PowerPC 64-bit (VSX support)" && \
	echo "  ppc-64-altivec          > PowerPC 64-bit (AltiVec support)" && \
	echo "  armv8-dotprod           > ARMv8 with NEON + dot product" && \
	echo "  armv8                   > ARMv8 with NEON" && \
	echo "  armv7-neon              > ARMv7 with NEON + POPCNT" && \
	echo "  armv7                   > ARMv7 32-bit" && \
	echo "  loongarch64-lasx        > LoongArch64 with LASX SIMD" && \
	echo "  loongarch64-lsx         > LoongArch64 with LSX SIMD" && \
	echo "  loongarch64             > Generic LoongArch64 build" && \
	echo "  apple-silicon           > Apple ARM64 (M1/M2/M3 chips)" && \
	echo "  riscv64                 > RISC-V 64-bit" && \
	echo "" && \
	echo "Supported compilers:" && \
	echo "  gcc                     > GNU Compiler Collection (default)" && \
	echo "  clang                   > LLVM Clang C++ Compiler" && \
	echo "  mingw                   > MinGW for Windows builds" && \
	echo "" && \
	echo "Normal examples (recommended for most users):" && \
	echo "  make -j build ARCH=native COMP=gcc" && \
	echo "  make -j build ARCH=x86-64-avx2 COMP=clang" && \
	echo "  make -f Makefile2 -j install PREFIX=/usr/local/bin" && \
	echo "" && \
	echo "Advanced examples (for power users):" && \
	echo "  make -j build ARCH=x86-64-avx512icl COMP=gcc DEBUG=yes" && \
	echo "  make -j build ARCH=armv8-dotprod COMP=clang" && \
	echo "  make -j build ARCH=loongarch64-lasx COMP=gcc OPTIMIZE=yes" && \
	echo "  make -j build ARCH=riscv64 COMP=gcc CXXFLAGS='-O3 -flto'" && \
	echo "" && \
	echo ""

all: build

build: config-sanity $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Force rebuild of version-related file(s) if needed
misc.o: FORCE
FORCE:

### Compile rules
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

### ==========================================================================
### Section 5. Install / strip / format
### ==========================================================================
strip: $(EXE)
	-$(STRIP) $(EXE)

install: $(EXE)
	-mkdir -p -m 755 $(BINDIR)
	-cp $(EXE) $(BINDIR)/$(EXE)
	-$(STRIP) $(BINDIR)/$(EXE) || true

format:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i $(SRCS) $(HEADERS) -style=file; \
	else \
		echo "clang-format not found"; exit 1; \
	fi

### ==========================================================================
### Section 6. Clean targets
### ==========================================================================
clean: objclean
	@rm -f .depend *~ core

objclean:
	@rm -f $(OBJS) $(EXE)

### ==========================================================================
### Section 7. Dependency generation
### ==========================================================================
.depend: $(SRCS)
	-@$(CXX) -MM $(SRCS) > $@ 2> /dev/null

ifeq (, $(filter $(MAKECMDGOALS), help strip install clean objclean format config-sanity))
-include .depend
endif

### ==========================================================================
### Section 8. Config sanity checks
### ==========================================================================
config-sanity:
	@echo "Config:" && \
	echo "  COMP = $(COMP)" && \
	echo "  CXX = $(CXX)" && \
	echo "  ARCH = $(ARCH)" && \
	echo "  CXXFLAGS = $(CXXFLAGS)" && \
	echo "  LDFLAGS = $(LDFLAGS)" && \
	true

### EOF
