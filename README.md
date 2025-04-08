# Package Performance LLM

This repository contains code for evaluating hardware performance for large language models (LLMs).

## Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

- Git
- Conda package manager
- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/package_performance_llm.git
cd package_performance_llm
```

2. Create and activate the conda environment using the provided environment file:
```bash
conda env create -f env.yaml
conda activate performance_llm
```

3. Navigate to the hardware performance directory:
```bash
cd hardware_performance
```

4. Run the performance evaluation script:
```bash
python run.py
```

## Usage

The `run.py` script will evaluate the performance of various hardware configurations for running LLMs. Results will be output to the terminal and saved to the specified output directory.

## Citation

If you use this work, please cite the following papers:

```bibtex
@inproceedings{maestro_micro2019,
    author    = {Hyoukjun Kwon and
                             Prasanth Chatarasi and
                             Michael Pellauer and
                             Angshuman Parashar and
                             Vivek Sarkar and
                             Tushar Krishna},
    title     = {Understanding Reuse, Performance, and Hardware Cost of {DNN} Dataflow:
                             {A} Data-Centric Approach},
    booktitle = {Proceedings of the 52nd Annual {IEEE/ACM} International Symposium
                             on Microarchitecture, {MICRO}},
    pages     = {754--768},
    publisher = {{ACM}},
    year      = {2019},
}

@article{maestro_toppicks2020,
    author    = {Hyoukjun Kwon and
                             Prasanth Chatarasi and
                             Vivek Sarkar and
                             Tushar Krishna and
                             Michael Pellauer and
                             Angshuman Parashar},
    title     = {{MAESTRO:} {A} Data-Centric Approach to Understand Reuse, Performance,
                             and Hardware Cost of {DNN} Mappings},
    journal   = {{IEEE} Micro},
    volume    = {40},
    number    = {3},
    pages     = {20--29},
    year      = {2020},
}
```
