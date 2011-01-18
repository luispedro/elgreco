#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>

#include "lda.h"


int main(int argc, char** argv) {
    using namespace lda;
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input-file", po::value<std::string>(), "Input data file")
        ("k", po::value<unsigned>(), "Nr of topics")
        ("iters", po::value<unsigned>(), "Nr of iterations")
        ("alpha", po::value<float>()->default_value(.1), "Alpha value")
        ("beta", po::value<float>()->default_value(.1), "beta value")
    ;

    po::positional_options_description p;
    p.add("input-file", 1);
    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv)
                .options(desc)
                .positional(p)
                .run(),
        vm);
    po::notify(vm);

    std::ifstream fin(vm["input-file"].as<std::string>().c_str());
    lda_data data = load(fin);
    std::cout << "Loaded " << data.nr_docs() << " documents." << std::endl;
    lda_parameters params;
    params.nr_topics = vm["k"].as<unsigned>();
    params.nr_iterations = vm["iters"].as<unsigned>();
    params.alpha = vm["alpha"].as<float>();
    params.beta = vm["beta"].as<float>();
    lda_state final_state = lda::lda(params, data);
    return 0;
}

