CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 -arch=native

COREDIR := ../core
SRC := $(wildcard $(COREDIR)/*)
INC := -I $(COREDIR)

DATADIR := .
OTDATADIR := $(DATADIR)/data/ot
DADATADIR := $(DATADIR)/data/da
DAOUTDATADIR := $(DATADIR)/data/da-out

DATASETS := "100 100 2" "500 500 2" "1000 1000 2" "2000 2000 2" "2000 3000 2" "3000 2000 2" "5000 5000 2"
OTPARAMSETS := "0. 1e-4" "1e-5 1e-4" "5e-5 1e-4" "1e-4 1e-4" "5e-4 1e-4" "1e-3 1e-4" "5e-3 1e-4" "1e-2 1e-4" "5e-2 1e-4" "1e-1 1e-4" "5e-1 1e-4" "1. 1e-4"
DAPARAMSETS := "0. 1e-4" "1e-6 1e-4" "5e-6 1e-4" "1e-5 1e-4" "5e-5 1e-4" "1e-4 1e-4"
NTESTS := 21

DROTRHO := 2.0
DROTUSEWARMUP = 1
DROTMAXITER = 20000

.PHONY: benchmark gendata

benchmark: build
	mkdir -p log
	for i in $(DATASETS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		c=$$3; \
		for j in $(OTPARAMSETS); do \
			set -- $$j; \
			r_weight=$$1; \
			eps=$$2; \
			./bin/multiexp_qr.o $(OTDATADIR) $$n $$m $(NTESTS) $(DROTRHO) $$r_weight $(DROTUSEWARMUP) $$eps $(DROTMAXITER) 1 >> log/quad_drot.log; \
		done; \
		for j in $(DAPARAMSETS); do \
			set -- $$j; \
			r_weight=$$1; \
			eps=$$2; \
			./bin/multiexp_glr.o $(DADATADIR) $(DAOUTDATADIR) $$n $$m $$c $(NTESTS) $(DROTRHO) $$r_weight $(DROTUSEWARMUP) $$eps $(DROTMAXITER) 1 >> log/gl_drot.log; \
		done; \
	done

gendata:
	mkdir $(DATADIR)/data
	mkdir $(OTDATADIR)
	mkdir $(DADATADIR)
	mkdir $(DAOUTDATADIR)
	for i in $(DATASETS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		c=$$3; \
		python utils/generate_ot.py $(OTDATADIR) $$n $$m $(NTESTS); \
		python utils/generate_da.py $(DADATADIR) $$n $$m $$c $(NTESTS); \
	done

build: bin/multiexp_qr.o bin/multiexp_glr.o bin/singleexp_qr.o bin/singleexp_glr.o

bin/multiexp_qr.o: $(SRC)
	mkdir -p bin
	rm -f bin/multiexp_qr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/multiexp_qr.o $(COREDIR)/multiexp_qr.cu

bin/multiexp_glr.o: $(SRC)
	mkdir -p bin
	rm -f bin/multiexp_glr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/multiexp_glr.o $(COREDIR)/multiexp_glr.cu

bin/singleexp_qr.o: $(SRC)
	mkdir -p bin
	rm -f bin/singleexp_qr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/singleexp_qr.o $(COREDIR)/singleexp_qr.cu

bin/singleexp_glr.o: $(SRC)
	mkdir -p bin
	rm -f bin/singleexp_glr.o
	$(CXX) $(CXXFLAGS) $(INC) -o bin/singleexp_glr.o $(COREDIR)/singleexp_glr.cu