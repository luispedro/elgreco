CXXFLAGS += -O3 -g
SOURCES = \
	load.cpp \
	lda.cpp \
	main.cpp

		
all: lda _lda.so _elgreco_random.so
.PHONY: all

lda: $(SOURCES)
	g++ $(CXXFLAGS) -fopenmp -o $@ $^ -lboost_program_options -lgsl -lgslcblas


lda_wrap.cxx: lda.i lda.h
	swig -python -c++ lda.i
lda_wrap.o: lda_wrap.cxx lda.h
	g++ -fPIC $(CXXFLAGS) -c lda_wrap.cxx `python-config --includes`

elgreco_random_wrap.cxx: elgreco_random.i elgreco_random.h
	swig -python -c++ elgreco_random.i
elgreco_random_wrap.o: elgreco_random_wrap.cxx elgreco_random.h
	g++ -fPIC $(CXXFLAGS) -c elgreco_random_wrap.cxx `python-config --includes`
_elgreco_random.so: elgreco_random_wrap.o
	g++ -fopenmp $(CXXFLAGS) -shared $^ -o $@ -lgsl -lgslcblas

lda.o: lda.cpp lda.h
	g++ $(CXXFLAGS) -fopenmp -fPIC -c lda.cpp

load.o: load.cpp lda.h
	g++ $(CXXFLAGS) -fPIC -c load.cpp

_lda.so: lda_wrap.o lda.o load.o
	g++ -fopenmp $(CXXFLAGS) -shared $^ -o $@ -lgsl -lgslcblas

clean:
	rm -f *.o lda _lda.so lda_wrap.cxx elgreco_random_wrap.cxx _elgreco_random.so
