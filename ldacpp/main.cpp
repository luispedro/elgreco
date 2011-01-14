#include <fstream>
#include <iostream>

#include "lda.h"
#include "load.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Wrong number of arguments." << std::endl;
        return 1;
    }
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
    }
    lda_data data = load(fin);
    std::cout << "Loaded " << data.nr_docs() << " documents." << std::endl;
    lda_parameters params;
    params.nr_topics = 10;
    params.nr_iterations = 1000;
    params.alpha = .1;
    params.beta = .1;
    lda_state final_state = lda(params, data);
    return 0;
}

