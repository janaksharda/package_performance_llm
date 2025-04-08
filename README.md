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
conda activate env
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
