# Dgen

This script generates SFT datasets for Code using a Domain-Specific Language (DSL) approach across multiple (opt.) repositories.

## Data availablity

For the API information, we release them in `./API_knowledge`

For the generated data, we release them seperately.


## Requirements

- Python 3.9+
- Required Python packages (list them here, e.g., numpy, pandas, matplotlib)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/tszdanger/Dgen.git
   cd Dgen
   ```


## Usage

Run the script using the following command:

```bash
python APIcoverage_PYskeleton_DSL_multi.py [arguments]
```

For further dataset processing, please refer to [Process](./deduplicate])



### Arguments

- `--repo`: Repository name (default: 'PYSkeletonDSL')
- `--model_name`: Model name to use (choices: 'deepseek', 'gpt-3.5-turbo', 'deepseek-silicon', and more... we also support all models on replicate, but there are some security issues thus we don't release it. default: 'deepseek')
- `--use_full_coverage`: Use full coverage (flag)
- `--ske_mean_num`: Skeleton mean number (default: 4)
- `--difficulty_level`: Difficulty level (choices: 1, 2, default: 2)
- `--api_knowledge_dir`: API knowledge directory (default: '/export/d3/zjli/API_coverage_LLM/API_knowledge')
- `--prompt_template_dir`: Prompt template directory (default: './prompt_template_multi.json')
- `--total_runtimes`: Total number of questions to generate (default: 3001)
- `--library_names`: Library names to use (default: ['Python'])
- `--generation_identifier`: Generation identifier (default: 'generalpython')
- `--use_simple`: Use simple mode (flag)
- `--step0_template_num`: Step 0 template number (default: '3')

### Example

```
python APIcoverage_PYskeleton_DSL_multi.py --repo MyRepo --model_name deepseek --ske_mean_num 5 --difficulty_level 1 --library_names numpy pandas --generation_identifier MyGen --use_simple
```

This command will run the script with the specified parameters, using the repo 'MyRepo', the deepseek model, a skeleton mean number of 5, difficulty level 1, numpy and pandas libraries, and the simple mode.

## Output

The script generates two types of output files in the `./data/{repo}/{generation_identifier}/{model_name}` directory:

1. A JSON file containing all generated questions and their metadata.
2. A JSONL file with the same information, allowing for easier appending of new questions.

The filename format is:
`{repo}_level{difficulty_level}_SIMPLE{USE_SIMPLE}_{APInumbers}items_{Length}_{TOTAL_RUNTIMES}.json(l)`

## Note

Ensure you have the necessary permissions and API keys set up for the models you intend to use.




