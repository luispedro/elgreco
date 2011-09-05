#include <iostream>
#include <string>
#include <sstream>
#include "lda.h"

lda::lda_data lda::load(std::istream& in) {
    std::string line;
    lda_data res;
    int max_term = 0;
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
            if (val > max_term) max_term = val;
            --nelems;
        }
        if (!linein.eof() || nelems) break;
        res.push_back_doc(curdoc, std::vector<floating>(), std::vector<bool>());
    }
    return res;
}
