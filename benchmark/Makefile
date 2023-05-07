CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 -arch=native

COREDIR := ../core
SRC := $(wildcard $(COREDIR)/*)
INC := -I $(COREDIR)

DATADIR := .
OTDATADIR := $(DATADIR)/data/ot
DADATADIR := $(DATADIR)/data/da
DAOUTDATADIR := $(DATADIR)/data/da-out

TESTSETS := "100 100 2 1e-3 1e-4 10000" "500 500 2 0. 1e-4 10000" \
"1000 1000 2 1e-3 1e-4 10000" "2000 2000 2 0. 1e-4 10000" \
"2000 3000 2 1e-5 1e-4 10000" "3000 2000 2 0. 1e-4 10000" \
"5000 5000 2 0. 1e-4 10000"
NTESTS := 11

DROTRHO := 2.0
DROTUSEWARMUP = 1

.PHONY: benchmark

benchmark: build
	mkdir -p log
	for i in $(TESTSETS); do \
		set -- $$i; \
		n=$$1; \
    	m=$$2; \
		c=$$3; \
		r_weight=$$4; \
		eps=$$5; \
		max_iters=$$6; \
		./bin/multiexp_qr.o $(OTDATADIR) $$n $$m $(NTESTS) $(DROTRHO) $$r_weight $(DROTUSEWARMUP) $$eps $$max_iters 1 >> log/quad_drot.log; \
		./bin/multiexp_glr.o $(DADATADIR) $(DAOUTDATADIR) $$n $$m $$c $(NTESTS) $(DROTRHO) $$r_weight $(DROTUSEWARMUP) $$eps $$max_iters 1 >> log/gl_drot.log; \
	done

gendata:
	mkdir -p $(OTDATADIR)
	mkdir -p $(DADATADIR)
	mkdir -p $(DAOUTDATADIR)
	for i in $(TESTSETS); do \
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