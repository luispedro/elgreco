CXXOPTS = -O3 -g
SOURCES = \
	load.cpp \
	lda.cpp \
	main.cpp

		
all: lda _lda.so
.PHONY: all

lda: $(SOURCES)
	g++ $(CXXOPTS) $(CXXFLAGS) -fopenmp -o $@ $^ -lboost_program_options -lgsl -lgslcblas


lda_wrap.cxx: lda.i lda.h
	swig -python -c++ lda.i


lda_wrap.o: lda_wrap.cxx lda.h
	g++ -O2 -fPIC $(CXXFLAGS) -c lda_wrap.cxx `python-config --includes`

lda.o: lda.cpp lda.h
	g++ -O2 $(CXXFLAGS) -fopenmp -fPIC -c lda.cpp

load.o: load.cpp lda.h
	g++ -O2 $(CXXFLAGS) -fPIC -c load.cpp

_lda.so: lda_wrap.o lda.o load.o
	g++ -fopenmp $(CXXFLAGS) -shared $^ -o $@ -lgsl -lgslcblas

clean:
	rm -f *.o lda _lda.so lda_wrap.cxx
