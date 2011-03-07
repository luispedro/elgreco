CXXOPTS = -O3 -g
SOURCES = \
	load.cpp \
	lda.cpp \
	main.cpp
		
all: lda _lda.so
.PHONY: all

lda: $(SOURCES)
	g++ $(CXXOPTS) -fopenmp -o $@ $^ -lboost_program_options -lgsl -lgslcblas


lda_wrap.cxx: lda.i lda.h
	swig -python -c++ lda.i


lda_wrap.o: lda_wrap.cxx lda.h
	g++ -O2 -fPIC -c lda_wrap.cxx -I /usr/include/python2.6

lda.o: lda.cpp lda.h
	g++ -O2 -fopenmp -fPIC -c lda.cpp

load.o: load.cpp lda.h
	g++ -O2 -fPIC -c load.cpp

_lda.so: lda_wrap.o lda.o load.o
	g++ -fopenmp -shared $^ -o $@ -lgsl -lgslcblas

clean:
	rm -f *.o lda _lda.so lda_wrap.cxx
