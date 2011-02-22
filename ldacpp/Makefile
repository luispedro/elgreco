CXXOPTS = -O3 -g
SOURCES = \
	load.cpp \
	lda.cpp \
	main.cpp
		

lda: $(SOURCES)
	g++ $(CXXOPTS) -o $@ $^ -lboost_program_options -lgsl -lgslcblas

