#include <fstream>
#include <iostream>

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
	#include <nlohmann/json.hpp>
	#pragma clang diagnostic pop
#else
	#include <nlohmann/json.hpp>
#endif

#include <argparse/argparse.hpp>

#include "algorithms.h"
#include "perf_utils.h"

// #include <fenv.h>
// #include <xmmintrin.h>

int main(int argc, char** argv)
{
	// 	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

	// 	fesetround(FE_TOWARDZERO);

	// #define CSR_FLUSH_TO_ZERO (1 << 15)
	// 	unsigned csr = __builtin_ia32_stmxcsr();
	// 	csr |= CSR_FLUSH_TO_ZERO;
	// 	__builtin_ia32_ldmxcsr(csr);

	argparse::ArgumentParser program("diffuse");

	std::string alg;
	program.add_argument("--alg").help("Algorithm to use").required().store_into(alg);

	std::string params_file;
	program.add_argument("--params").help("A file with the algorithm specific parameters").store_into(params_file);

	std::string problem_file;
	program.add_argument("--problem")
		.help("A file describing the problem instance")
		.required()
		.store_into(problem_file);

	bool double_precision;
	program.add_argument("--double").help("Use double precision").flag().store_into(double_precision);

	bool verbose;
	program.add_argument("-v").help("Add verbosity").flag().store_into(verbose);

	auto& group = program.add_mutually_exclusive_group();

	bool validate;
	group.add_argument("--validate")
		.help("Check the validity of the algorithm by running single iteration per each dimension and comparing to the "
			  "reference implementation")
		.flag()
		.store_into(validate);

	std::string output_file;
	program.add_argument("--run")
		.help("Run the algorithm. Optionally provide an output file path.")
		.nargs(argparse::nargs_pattern::optional)
		.store_into(output_file);

	bool benchmark;
	group.add_argument("--benchmark")
		.help("The run of the algorithm will be benchmarked and outputed to standard output")
		.flag()
		.store_into(benchmark);

	bool profile;
	group.add_argument("--profile").help("The run will be profiled using PAPI counters.").flag().store_into(profile);

	try
	{
		// program.parse_args({ "./diffuse", "--alg", "full_lapack", "--problem", "../example-problems/toy.json",
		// "--validate"});
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	algorithms algs(double_precision, verbose);

	max_problem_t problem;

	try
	{
		problem = problems::read_problem(problem_file);
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		return 1;
	}

	nlohmann::json params;

	if (!params_file.empty())
	{
		std::ifstream ifs(params_file);
		if (!ifs)
		{
			std::cerr << "Cannot open file " << params_file << std::endl;
			return 1;
		}

		ifs >> params;
	}

	try
	{
		if (validate)
		{
			algs.validate(alg, problem, params);
		}
		else if (program.is_used("--run"))
		{
			algs.run(alg, problem, params, output_file);
		}
		else if (benchmark)
		{
			algs.benchmark(alg, problem, params);
		}
		else if (profile)
		{
			perf_counter::enable();
			algs.run(alg, problem, params, output_file);
		}
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl;
		return 1;
	}

	return 0;
}
