CXX=nvcc
CXXFLAGS=-Icuda_helper -O3

SRCDIR := src
INC := -I ../core/

TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

.PHONY: build clean
build:
	mkdir -p ./bin/
	rm -f ./bin/drot_qr.cu.o
	$(CXX) $(CXXFLAGS) -c -o ./bin/drot_qr.cu.o src/drot_qr.cu.cc ${INC} ${TF_CFLAGS} ${TF_LFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
	rm -f ./bin/reg_drot_extend.so
	g++ -Wall -shared -o ./bin/drot_extend.so ./src/drot_extend.cc ./bin/drot_qr.cu.o ${TF_CFLAGS} -fPIC -lcudart ${TF_LFLAGS}
	realpath ./bin/reg_drot_extend.so

test:
	python test.py