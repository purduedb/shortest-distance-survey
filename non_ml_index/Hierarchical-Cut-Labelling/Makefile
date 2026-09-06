# Auto-detect available CPU cores.
#   Linux+SLURM : sched_getaffinity returns only the SLURM-allocated cores.
#   Linux/macOS : falls back to multiprocessing.cpu_count().
#   No Python   : tries nproc (Linux), then sysctl (macOS), then hardcodes 4.
NTHREADS := $(shell \
	python -c "import os,multiprocessing; \
	           print(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else multiprocessing.cpu_count())" \
	2>/dev/null \
	|| nproc 2>/dev/null \
	|| sysctl -n hw.logicalcpu 2>/dev/null \
	|| echo 4)

CC  = g++ -std=c++20 -O3 -Wall -Wextra -fopenmp -DMULTI_THREAD_DISTANCES=$(NTHREADS) -o
TCC = g++ -std=c++20 -ggdb -Wall -Wextra -fopenmp -DMULTI_THREAD_DISTANCES=$(NTHREADS) -o
INC = src/road_network.cpp src/util.cpp

BUILD_DIR         = ./build
SAVED_INDEXES_DIR = ./saved_indexes
DATA_DIR         ?= ../../../data
DATA_NAME        ?= W_Jinan

all: compile build evaluate

compile:
	@echo "Compiling HCL binaries (MULTI_THREAD_DISTANCES=$(NTHREADS))..."
	mkdir -p $(BUILD_DIR)
	$(CC)  $(BUILD_DIR)/cut       src/main.cpp       $(INC)
	$(CC)  $(BUILD_DIR)/index     src/index.cpp      $(INC)
	$(CC)  $(BUILD_DIR)/inference src/inference.cpp  $(INC)
	$(CC)  $(BUILD_DIR)/query     src/query.cpp      $(INC)
	$(CC)  $(BUILD_DIR)/topcut    src/topcut.cpp     $(INC)
	$(TCC) $(BUILD_DIR)/test      src/test.cpp       $(INC)

build:
	@echo "Building HCL index for $(DATA_NAME)..."
	mkdir -p $(SAVED_INDEXES_DIR)
	$(BUILD_DIR)/index $(DATA_DIR)/$(DATA_NAME)/$(DATA_NAME).edges $(SAVED_INDEXES_DIR)/$(DATA_NAME).hl

evaluate:
	@echo "Evaluating HCL index for $(DATA_NAME) (1M random queries)..."
	$(BUILD_DIR)/query $(SAVED_INDEXES_DIR)/$(DATA_NAME).hl

clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(SAVED_INDEXES_DIR)/*

.PHONY: all compile build evaluate clean
