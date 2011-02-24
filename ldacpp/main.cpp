#include <boost/program_options.hpp>
#include <fstream>

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
        ("seed", po::value<unsigned>()->default_value(2), "Seed value (for random numbers)")
        ("verbose", po::value<int>()->default_value(0), "Verbosity level")
    ;

    po::options_description allopts("Command Line Options");
    allopts.add(req).add(opts);
    po::positional_options_description p;
    p.add("input-file", 1);
    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv)
                .options(allopts)
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
    const int verbose = vm["verbose"].as<int>();
    if (verbose) std::cout << "Loaded " << data.nr_docs() << " documents." << std::endl;

    lda_parameters params;
    params.nr_topics = vm["k"].as<unsigned>();
    params.nr_iterations = vm["iters"].as<unsigned>();
    params.seed = vm["seed"].as<unsigned>();
    params.alpha = vm["alpha"].as<float>();
    params.beta = vm["beta"].as<float>();
    //lda_state final_state = lda::lda(params, data);
    ::lda::lda state(data, params);
    state.forward();
    for (int i = 0; i != params.nr_iterations; ++i) {
        std::cout << state.logP() << '\n';
        state.step();
    }
    return 0;
}

