# ---- CUDA C Compiler Settings ----
CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 -arch=native

# ---- Include Directories ----
CORE_DIR := ../core
SRC := $(wildcard $(CORE_DIR)/*)
INC := -I $(CORE_DIR)

# ---- Directories ----
DATA_DIR := .
OT_DATA_DIR := $(DATA_DIR)/data/ot
DA_DATA_DIR := $(DATA_DIR)/data/da
DA_OUT_DATA_DIR := $(DATA_DIR)/data/da-out
LOG_DIR := ./log/

# ---- Stopping Criterior ----
GLOBAL_EPS = 1e-4

# ---- Optimal Transport Testsets ----
# Problem Sizes
OT_TESTS := "100 100" "500 500" "1000 1000" "2000 2000" "2000 3000" "3000 2000" "5000 5000" "10000 10000"
# Unnormed Step Size
OT_DROT_RHO := 2.0 1.0
# Unnormed Weight for Quadratic Regularizer of DROT
OT_DROT_R_WEIGHT := 0. 5e-4 1e-3 5e-3 1e-2 3e-2 6e-2 1e-1 2e-1 1e+0
# Weight for Entropic Regularizer of Sinkhorn's Algorithm
OT_SK_R_WEIGHT := 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e+0 1e+1 1e+2
# Learning Rate for L-BFGS Solver
OT_LBFGS_LR := 1e-0 1e-1

# ---- Domain Adapatation Testsets ----
# Problem Sizes & Number of Classes
DA_TESTS := "100 100 2" "500 500 2" "1000 1000 4" "1500 1000 2" "1020 1500 3"
# Unnormed Weight for Group-Lasso Regularizer (both DROT and SK)
DA_GL_WEIGHT := 1e-6 1e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2
# Weight for Entropic Regularizer of Sinkhorn's Algorithm
DA_SK_ER_WEIGHT := 1e-3 1e-2 1e-1 1e-0 1e+1

# Number of Repeated Tests (the first test will be dropped)
NTESTS := 51

# ---- DROT Default Parameters ----
# Unnormed Step Size
DROT_RHO := 2.0
# Use Warmup Initialization
DROT_USE_WARMUP = 1
# Max Iterations
DROT_MAX_ITER = 20000

# ---- Sinkhorn's Algorithm Default Parameters ----
SK_MAX_ITER = 20000

# ---- L-BFGS Solver's Default Parmaters ----
LBFGS_HISTORY = 10

.PHONY: benchmark gendata

# Benchmark for DROT
benchmark-drot: build
	mkdir -p $(LOG_DIR)
	echo "" > $(LOG_DIR)/quad_drot.log
	for i in $(OT_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		for rho in $(OT_DROT_RHO); do \
			for r_weight in $(OT_DROT_R_WEIGHT); do \
				./bin/multiexp_qr.o $(OT_DATA_DIR) $$n $$m $(NTESTS) $$rho $$r_weight $(DROT_USE_WARMUP) $(GLOBAL_EPS) $(DROT_MAX_ITER) 1 >> log/quad_drot.log; \
			done; \
		done; \
	done

benchmark-drot-da: build
	mkdir -p $(LOG_DIR)
	echo "" > $(LOG_DIR)/gl_drot.log
	for i in $(DA_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		n_class=$$3; \
		for r_weight in $(DA_GL_WEIGHT); do \
			mkdir -p $(DA_OUT_DATA_DIR)/drot/$$r_weight; \
			./bin/multiexp_glr.o $(DA_DATA_DIR) $(DA_OUT_DATA_DIR)/drot/$$r_weight $$n $$m $$n_class \
				$(NTESTS) $(DROT_RHO) $$r_weight $(DROT_USE_WARMUP) $(GLOBAL_EPS) $(DROT_MAX_ITER) 1 >> log/gl_drot.log; \
		done; \
	done

# Benchmark for Sinkhorn
benchmark-sk:
	mkdir -p $(LOG_DIR)
	echo "" > $(LOG_DIR)/sk_entropic.log
	for i in $(OT_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		for r_weight in $(OT_SK_R_WEIGHT); do \
			python other/sinkhorn/sk_entropic.py $(OT_DATA_DIR) $$n $$m $(GLOBAL_EPS) $(SK_MAX_ITER) $$r_weight $(NTESTS) 1 >> log/sk_entropic.log; \
		done; \
	done

benchmark-sk-da:
	mkdir -p $(LOG_DIR)
	echo "" > $(LOG_DIR)/sk_entropic_gl.log
	for i in $(DA_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		c=$$3; \
		for r_weight in $(DA_SK_ER_WEIGHT); do \
			for gl_weight in $(DA_GL_WEIGHT); do \
				mkdir -p $(DA_OUT_DATA_DIR)/sk/$$r_weight/$$gl_weight; \
				python other/sinkhorn/sk_entropic_gl.py $(DA_DATA_DIR) $(DA_OUT_DATA_DIR)/sk/$$r_weight/$$gl_weight $$n $$m $$c $(GLOBAL_EPS) \
					$(SK_MAX_ITER) $$r_weight $$gl_weight $(NTESTS) 1 >> log/sk_entropic_gl.log; \
			done; \
		done; \
	done

# Benchmark for L-BFGS Solver
benchmark-lbfgs:
	mkdir -p $(LOG_DIR)
	echo "" > $(LOG_DIR)/lbfgs.log
	for i in $(OT_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		for lr in $(OT_LBFGS_LR); do \
			for r_weight in $(OT_DROT_R_WEIGHT); do \
				python other/smoothot/lbfgs.py $(OT_DATA_DIR) $$n $$m $$r_weight $$lr $(GLOBAL_EPS) $(LBFGS_HISTORY) $(NTESTS) >> $(LOG_DIR)/lbfgs.log; \
			done; \
		done; \
	done

gendata:
	mkdir $(DATA_DIR)/data
	mkdir $(OT_DATA_DIR)
	mkdir $(DA_DATA_DIR)
	mkdir $(DA_OUT_DATA_DIR)
	for i in $(OT_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		python utils/generate_ot.py $(OT_DATA_DIR) $$n $$m $(NTESTS); \
	done
	for i in $(DA_TESTS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		c=$$3; \
		python utils/generate_da.py $(DA_DATA_DIR) $$n $$m $$c $(NTESTS); \
	done

build: bin/multiexp_qr.o bin/multiexp_glr.o bin/singleexp_qr.o bin/singleexp_glr.o

bin/multiexp_qr.o: $(SRC)
	mkdir -p bin
	rm -f bin/multiexp_qr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/multiexp_qr.o $(CORE_DIR)/multiexp_qr.cu

bin/multiexp_glr.o: $(SRC)
	mkdir -p bin
	rm -f bin/multiexp_glr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/multiexp_glr.o $(CORE_DIR)/multiexp_glr.cu

bin/singleexp_qr.o: $(SRC)
	mkdir -p bin
	rm -f bin/singleexp_qr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/singleexp_qr.o $(CORE_DIR)/singleexp_qr.cu

bin/singleexp_glr.o: $(SRC)
	mkdir -p bin
	rm -f bin/singleexp_glr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/singleexp_glr.o $(CORE_DIR)/singleexp_glr.cu

clean:
	rm -rf ./bin
	rm -rf $(DATA_DIR)/data
