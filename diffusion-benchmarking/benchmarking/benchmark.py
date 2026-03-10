"""
Benchmarking Data Generator and Runner

This script automates the generation of benchmarking data and the execution of diffusion benchmarks for various algorithms and parameter combinations.

Features:
- Parses a benchmark JSON file describing groups, problem sizes, algorithms, and parameter sweeps.
- Generates input problem files and parameter files for each group and algorithm.
- Iterates over all combinations of algorithm parameters using Cartesian product.
- Runs the specified executable for each problem/parameter combination and collects results.
- Supports running benchmarks for all groups or a selected group.

Usage:
    python benchmark.py <benchmark_json_path> <executable_path> [-g GROUP_NAME]

Arguments:
    benchmark_json_path: Path to the benchmark JSON file.
    executable_path: Path to the diffusion benchmark executable.
    -g, --group: (Optional) Name of the group to benchmark. If omitted, all groups are processed.

JSON Schema:
    {
        "name": <benchmark_name>,
        "groups": {
            <group_name>: {
                "sizes": ["<x_dim>x<y_dim>x<z_dim>x<substrates_count>x<iterations>", ...],
                "data_type": <"float" or "double">,
                "default_params": {<param_name>: <value>, ...},
                "runs": {
                    <run_name>: {
                        "alg": <algorithm_name>,
                        "params": {<param_name>: [<values>], ...}
                    },
                    ...
                }
            },
            ...
        }
    }

Outputs:
- Problem and parameter files are generated in a directory structure under <benchmark_name>/<group_name>/.
- Results are saved as CSV files for each run/parameter/problem combination.
"""

import csv
import sys
import json
import itertools
import os
from typing import TypedDict, List, Dict, Any
import argparse
import subprocess


class RunDict(TypedDict):
    """
    TypedDict for a single run configuration.
    Fields:
        alg (str): Algorithm name.
        params (Dict[str, List[int]]): Parameter sweep dictionary.
    """
    alg: str
    params: Dict[str, List[int]]


class GroupDict(TypedDict):
    """
    TypedDict for a benchmark group configuration.
    Fields:
        sizes (List[str]): List of problem size strings.
        default_params (Dict[str, Any]): Default parameters for the group.
        data_type (str): Data type ("float" or "double").
        runs (Dict[str, RunDict]): Dictionary of run configurations.
    """
    sizes: List[str]
    default_params: Dict[str, Any]
    data_type: str
    runs: Dict[str, RunDict]


class Benchmarking:
    def __init__(self, executable_path: str, groups: dict[str, GroupDict], benchmarking_name: str, selected_group: str):
        self.executable_path = executable_path
        self.groups = groups
        self.benchmarking_name = benchmarking_name
        self.selected_group_name = selected_group
        self.benchmarking_cases: int = 0
        self.progress: int = 0

        print("Printing OMP env vars:")
        for k, v in os.environ.items():
            if k.startswith("OMP"):
                print(f"{k}={v}")

    def group_dir(self, group_name: str):
        return f"{self.benchmarking_name}/{group_name}"

    def data_dir(self, group_name: str):
        return f"{self.group_dir(group_name)}/data"

    def run_dir(self, group_name: str, alg_name: str):
        return f"{self.group_dir(group_name)}/{alg_name}"

    def params_dir(self, group_name: str, alg_name: str, param_name: str):
        return f"{self.group_dir(group_name)}/{alg_name}/{param_name}"

    def benchmark(self):
        self.generate_data()
        self.run()

    def generate_data(self):
        print("Generating data...")

        if self.selected_group_name == "":
            for group_name, group in groups.items():
                self.generate_group_data(group_name, group)
        else:
            if not self.selected_group_name in groups.keys():
                print(
                    f"Error: Group '{selected_group_name}' not found.", file=sys.stderr)
                sys.exit(1)
            self.generate_group_data(self.selected_group_name,
                                     groups[self.selected_group_name])

    def run(self):
        print("Running...")

        if self.selected_group_name == "":
            for group_name, group in groups.items():
                self.run_single_group(group_name, group)
        else:
            self.run_single_group(self.selected_group_name,
                                  groups[self.selected_group_name])

    def merge_run(self, run_name: str, run_paths: list[str]):
        """
        Merge csv files from all groups and parameters into single file for each run.

        Args:
            run_name (str): Run name.
            run_paths (list[str]): List of all csv files related to the run.
        """
        merged_rows: list[list[str]] = []
        expected_header: list[str] | None = None

        for file_path in run_paths:
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                try:
                    header = next(reader)
                except StopIteration:
                    print(f"Warning: {file_path} is empty.")
                    continue

                if expected_header is None:
                    expected_header = header
                elif header != expected_header:
                    print(f"Header mismatch in file {file_path}:")
                    print(f"Expected: {expected_header}")
                    print(f"Found   : {header}")
                    return

                merged_rows.extend(list(reader))

        if expected_header is None:
            print(f"No data for run {run_name}")
            return

        # Write merged CSV
        with open(f"{run_name}.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(expected_header)
            writer.writerows(merged_rows)

    def merge_groups(self):
        all_runs: set[str] = set()
        for group in self.groups.values():
            all_runs.update(group["runs"].keys())

        runs_csvs: dict[str, list[str]] = dict()

        for run in all_runs:
            for group in self.groups:
                run_dir = self.run_dir(group, run)
                if not os.path.exists(run_dir):
                    continue
                for param in os.listdir(run_dir):
                    param_dir = os.path.join(run_dir, param)
                    csv_files = [os.path.join(param_dir, f) for f in os.listdir(
                        param_dir) if f.endswith(".csv")]
                    if run not in runs_csvs:
                        runs_csvs[run] = []
                    runs_csvs[run].extend(csv_files)

        for run, run_data in runs_csvs.items():
            self.merge_run(run, run_data)

    def create_problems_for_group(self, group_name: str, sizes: list[str]):
        """
        Generate problem JSON files for each specified size in a group.

        Args:
            group_name (str): Group name.
            sizes (list[str]): List of problem size strings (e.g., "32x32x32x1x1").
        """
        for problem_size in sizes:
            parts = [int(x) for x in problem_size.split('x')]
            dims = len(parts) - 2
            if (dims < 2):
                print(
                    f"Error: Incorrect problem size: {problem_size}", file=sys.stderr)
                sys.exit(1)
            iterations: int = int(parts[-1])
            substrate_count: int = int(parts[-2])
            nx = int(parts[0])
            ny = int(parts[1])
            nz = int(parts[2]) if dims == 3 else 1

            template: dict[str, object] = {
                "dims": dims,
                "dx": 20,
                "dy": 20,
                "dz": 20,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "substrates_count": substrate_count,
                "iterations": iterations,
                "dt": 0.01,
                "diffusion_coefficients": 10000,
                "decay_rates": 0.01,
                "initial_conditions": 1000,
                "gaussian_pulse": False
            }

            data_dir = self.data_dir(group_name)
            os.makedirs(data_dir, exist_ok=True)

            output_path = f"{data_dir}/{problem_size}.json"
            with open(output_path, "w") as outfile:
                json.dump(template, outfile, indent=2)

    def create_param_file_for_alg(self, run_directory: str, default_params: dict[str, object],  params: dict[str, list[int]]):
        """
        Generate parameter JSON files for each combination of algorithm parameters.

        Args:
            run_directory (str): Directory to store parameter files.
            default_params (dict): Default parameters to include in each file.
            params (dict): Dictionary of parameter lists to sweep over.
        """
        keys = list(params.keys())
        values = [params[k] for k in keys]

        run_cases = 0

        for combination in itertools.product(*values):
            run_cases += 1
            combo_dict = dict(zip(keys, combination))
            combo_dict: dict[str, object] = {**default_params, **combo_dict}

            args = "_".join(str(x) for x in combination)
            args = args.replace(' ', '')
            args = args.replace('[', '')
            args = args.replace(']', '')
            args = args.replace(',', '-')

            param_dir = f"{run_directory}/{args}"
            os.makedirs(param_dir, exist_ok=True)

            output_path = f"{run_directory}/{args}/params.json"
            with open(output_path, "w") as outfile:
                json.dump(combo_dict, outfile, indent=2)

        return run_cases

    def generate_group_data(self, group_name: str, group: GroupDict):
        """
        Generate all problem and parameter files for a benchmark group.

        Args:
            bench_name (str): Benchmark name.
            group_name (str): Group name.
            group (GroupDict): Group configuration dictionary.
        """
        group_dir = self.group_dir(group_name)
        os.makedirs(group_dir, exist_ok=True)

        self.create_problems_for_group(group_name, group["sizes"])

        group_cases = 0

        for run_name, run in group["runs"].items():
            if run_name == "data":
                print("Run name can not be named 'data'.",  file=sys.stderr)
                sys.exit(1)
            run_dir = self.run_dir(group_name, run_name)
            os.makedirs(run_dir, exist_ok=True)

            group_cases += self.create_param_file_for_alg(
                run_dir, group["default_params"], run["params"])

        self.benchmarking_cases += group_cases * len(group["sizes"])

    def diffuse(self, problem_path: str, params_path: str, alg: str, double: bool):
        """
        Run the benchmark executable for a given problem and parameter set.

        Args:
            executable_path (str): Path to the benchmark executable.
            problem_path (str): Path to the problem JSON file.
            params_path (str): Path to the parameter JSON file.
            alg (str): Algorithm name.
            double (bool): Whether to use double precision.
        """
        problems_descriptor = os.path.basename(problem_path)
        out_csv_path = os.path.join(os.path.dirname(
            params_path), f"{os.path.splitext(problems_descriptor)[0]}.csv")

        cmd = [
            self.executable_path,
            "--alg", alg,
            "--problem", problem_path,
            "--params", params_path,
            "--benchmark"
        ]

        if double:
            cmd.append("--double")

        self.progress += 1
        status = f"[{self.progress} / {self.benchmarking_cases}]"
        run_descriptor = " ".join(cmd)

        if os.path.exists(out_csv_path):
            print(f"{status} Skipped! {run_descriptor}")
            return

        try:
            with open(out_csv_path, "w") as outfile:
                subprocess.run(cmd, stdout=outfile,
                               stderr=subprocess.STDOUT, check=True)
            print(f"{status} Finished! {run_descriptor}")
        except subprocess.CalledProcessError:
            os.remove(out_csv_path)
            print(f"{status} Failed! {run_descriptor}")
        except KeyboardInterrupt:
            if os.path.exists(out_csv_path):
                os.remove(out_csv_path)
                print(f"{status} {run_descriptor} interrupted by user")
                sys.exit(1)

    def run_single_group(self, group_name: str, group: GroupDict):
        """
        Run all benchmarks for a single group, iterating over problems and parameter sets.

        Args:
            executable (str): Path to the benchmark executable.
            bench_name (str): Benchmark name.
            group_name (str): Group name.
            group (GroupDict): Group configuration dictionary.
        """
        group_dir = self.group_dir(group_name)
        double: bool = group["data_type"] == "double"

        for run_name in os.listdir(group_dir):
            if run_name == "data":
                continue

            run_dir = self.run_dir(group_name, run_name)
            alg = group["runs"][run_name]["alg"]

            for param_name in os.listdir(run_dir):
                if not os.path.isdir(os.path.join(run_dir, param_name)):
                    continue
                params_path = os.path.join(
                    group_dir, run_name, param_name, "params.json")

                for problem_file in os.listdir(os.path.join(group_dir, "data")):
                    problem_path = os.path.join(
                        group_dir, "data", problem_file)

                    self.diffuse(problem_path, params_path, alg, double)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking data generator")
    parser.add_argument("benchmark_path", help="Path to JSON file")
    parser.add_argument("executable_path", help="Path to executable")
    parser.add_argument(
        "-g", "--group", required=False, action='store', default="", help="Group name to benchmark")
    parser.add_argument(
        "--prefix", required=False, action='store', default="", help="Benchmarking prefix. Useful when running the same benchmark file on various machines.")
    parser.add_argument(
        "--merge", required=False, action='store_true', help="Merge results across all groups (incompatible with --group)")

    args = parser.parse_args()

    if args.merge and args.group:
        parser.error("--merge and --group cannot be used together.")

    args = parser.parse_args()
    json_path = args.benchmark_path
    exe_path = args.executable_path
    prefix = args.prefix

    selected_group_name = args.group

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    benchmarking_name = prefix + data["name"]
    groups = data["groups"]

    b = Benchmarking(exe_path, groups, benchmarking_name, selected_group_name)

    if not args.merge:
        b.benchmark()
    else:
        b.merge_groups()
