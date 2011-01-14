#include <iostream>
#include <string>
#include <sstream>
#include "lda.h"

lda_data load(std::istream& in) {
    int nr_docs = 0;
    std::string line;
    lda_data res;
    while (std::getline(in, line)) {
        std::stringstream linein(line);
        int nelems = 0;
        linein >> nelems;
        int val, count;
        char c;
        std::vector<int> curdoc;
        curdoc.reserve(nelems);
        while (linein >> val >> c >> count) {
            for (int i = 0; i != count; ++i) curdoc.push_back(val);
            --nelems;
        }
        if (!linein.eof() || nelems) break;
        res.docs.push_back(curdoc);
    }
    return res;
}
