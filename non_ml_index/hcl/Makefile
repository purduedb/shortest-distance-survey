# Usage:
# SLURM Test Environment: sinteractive -N 1 -n 16 --gres=gpu:1 --partition=a30 --mem=10G --account=csml --qos standby --time 30
# Execute using: make all > non_ml_python_evaluation_output.log

DATA_DIR = ../../data
BUILD_DIR = ./Hierarchical-Cut-Labelling/build
SAVED_MODELS_DIR = ./saved_models
CC = g++ -std=c++20 -O3 -Wall -Wextra -o
INC = ../src/road_network.cpp ../src/util.cpp

all: compile build evaluate

compile:
	@echo "Compiling HCL index..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CC) index ../src/index.cpp $(INC)

build:
	@echo "Building HCL index..."
	$(BUILD_DIR)/index $(DATA_DIR)/W_Jinan/W_Jinan.edges $(SAVED_MODELS_DIR)/W_Jinan.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_Shenzhen/W_Shenzhen.edges $(SAVED_MODELS_DIR)/W_Shenzhen.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_Chengdu/W_Chengdu.edges $(SAVED_MODELS_DIR)/W_Chengdu.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_Beijing/W_Beijing.edges $(SAVED_MODELS_DIR)/W_Beijing.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_NewYork/W_NewYork.edges $(SAVED_MODELS_DIR)/W_NewYork.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_Chicago/W_Chicago.edges $(SAVED_MODELS_DIR)/W_Chicago.hl
	$(BUILD_DIR)/index $(DATA_DIR)/W_Shanghai/W_Shanghai.edges $(SAVED_MODELS_DIR)/W_Shanghai.hl

evaluate:
	@echo "Evaluating HCL index..."
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Jinan.hl --query_path $(DATA_DIR)/W_Jinan/real_workload/W_Jinan.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Shenzhen.hl --query_path $(DATA_DIR)/W_Shenzhen/real_workload/W_Shenzhen.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Chengdu.hl --query_path $(DATA_DIR)/W_Chengdu/real_workload/W_Chengdu.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Beijing.hl --query_path $(DATA_DIR)/W_Beijing/real_workload/W_Beijing.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_NewYork.hl --query_path $(DATA_DIR)/W_NewYork/real_workload/W_NewYork.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Chicago.hl --query_path $(DATA_DIR)/W_Chicago/real_workload/W_Chicago.queries  --eval_runs 10
	python evaluate.py --index_path $(SAVED_MODELS_DIR)/W_Shanghai.hl --query_path $(DATA_DIR)/W_Shanghai/real_workload/W_Shanghai.queries  --eval_runs 10

clean:
	rm -r $(SAVED_MODELS_DIR)/*
	rm -r $(BUILD_DIR)/*
