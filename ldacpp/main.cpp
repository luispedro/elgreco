#include <fstream>
#include <iostream>

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
    return 0;
}

