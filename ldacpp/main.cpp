#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>

#include "lda.h"

int main(int argc, char** argv) {
    using namespace lda;
    namespace po = boost::program_options;
    po::options_description req("Required Parameters");
    po::options_description opts("Options");
    req.add_options()
        ("input-file", po::value<std::string>(), "Input data file")
        ("k", po::value<unsigned>(), "Nr of topics")
        ("iters", po::value<unsigned>(), "Nr of iterations")
    ;
    opts.add_options()
        ("help", "produce help message")
        ("alpha", po::value<float>()->default_value(.1), "Alpha value")
        ("beta", po::value<float>()->default_value(.1), "beta value")
    ;

    po::positional_options_description p;
    p.add("input-file", 1);
    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv)
                .options(req)
                //.options(opts)
                .positional(p)
                .run(),
        vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("input-file") || !vm.count("k") || !vm.count("iters")) {
        std::cout << req << opts << std::endl;
        if (vm.count("help")) return 0;
        return 1;
    }

    std::ifstream fin(vm["input-file"].as<std::string>().c_str());
    lda_data data = load(fin);
    std::cout << "Loaded " << data.nr_docs() << " documents." << std::endl;
    lda_parameters params;
    params.nr_topics = vm["k"].as<unsigned>();
    params.nr_iterations = vm["iters"].as<unsigned>();
    params.alpha = 0.1;// vm["alpha"].as<float>();
    params.beta =0.1; // vm["beta"].as<float>();
    //lda_state final_state = lda::lda(params, data);
    ::lda::lda state(data, params);
    state.forward();
    for (int i = 0; i != params.nr_iterations; ++i) {
        std::cout << state.logP() << '\n';
        state.gibbs();
    }
    return 0;
}

