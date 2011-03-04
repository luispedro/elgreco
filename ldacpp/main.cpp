#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <fstream>

#include "lda.h"

int main(int argc, char** argv) {
    using namespace lda;
    namespace po = boost::program_options;
    po::options_description req("Required Parameters");
    po::options_description opts("Options");
    req.add_options()
        ("input-file", po::value<std::string>(), "Input data file")
        ("mode", po::value<std::string>()->default_value("estimate"), "Mode")
        ("k", po::value<unsigned>(), "Nr of topics")
    ;
    opts.add_options()
        ("help", "produce help message")
        ("iters", po::value<unsigned>()->default_value(200), "Nr of iterations")
        ("sampler", po::value<std::string>()->default_value("uncollapsed"), "Sampler to use")
        ("print-iters", po::value<unsigned>()->default_value(10), "Iterations between printing logP (set to zero for no output)")
        ("alpha", po::value<float>()->default_value(.1), "Alpha value")
        ("beta", po::value<float>()->default_value(.1), "beta value")
        ("seed", po::value<unsigned>()->default_value(2), "Seed value (for random numbers)")
        ("topics-file", po::value<std::string>()->default_value("topics.txt"), "Topics file name.")
        ("words-file", po::value<std::string>()->default_value("words.txt"), "Words file name.")
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

    if (vm.count("help") || !vm.count("input-file") || !vm.count("k")) {
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
    std::string mode = vm["mode"].as<std::string>();
    std::string sampler = vm["sampler"].as<std::string>();
    if (mode == "estimate") {
        boost::scoped_ptr<lda_base> state;
        if (sampler == "uncollapsed") {
            state.reset(new lda_uncollapsed(data, params));
        } else {
            state.reset(new lda_collapsed(data, params));
        }
        state->forward();
        const unsigned print_iters = vm["print-iters"].as<unsigned>();
        for (int i = 0; i != params.nr_iterations; ++i) {
            state->step();
            if (print_iters && (i % print_iters) == 0) {
                std::cout << state->logP() << '\n';
            }
        }
        std::ofstream topicsf(vm["topics-file"].as<std::string>().c_str());
        state->print_topics(topicsf);
        std::ofstream wordsf(vm["words-file"].as<std::string>().c_str());
        state->print_words(wordsf);
    } else {
        ::lda::lda_uncollapsed state(data, params);
        std::ifstream topicsf(vm["topics-file"].as<std::string>().c_str());
        std::ifstream wordsf(vm["words-file"].as<std::string>().c_str());
        state.load(topicsf, wordsf);
        std::cout << state.logP() << '\n';
    }
    return 0;
}

