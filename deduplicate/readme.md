# Python Code Dataset Processor

## Deduplicate.py


### Code Deduplication and Refinement Tool

This tool is designed to process and refine code examples stored in JSONL files. It performs deduplication, rewrites similar problems, and generates improved versions of the code.

### Features

- Deduplicates code examples within a file
- Rewrites similar problems to create unique examples
- Removes comments from code (optional)
- Supports multiple language models for code refinement
- Customizable selection rules for processing specific files

### Requirements

- Python 3.x
- Required Python packages: `tqdm`, `fuzzywuzzy`, and custom modules for API querying

### Usage

1. Set the `basedir` variable in the `main()` function to point to your data directory.
2. Customize the `selected_rules()` function if you need specific file selection criteria.
3. Run the script:

```
python Deduplicate.py
```

4. The script will process eligible JSONL files and create new files prefixed with 'FIX_' containing the refined and deduplicated content.

### Configuration

- Adjust the `prompt_template_dir` variable to point to your JSON file containing prompt templates.
- Modify the `usedmodel` variable to choose the desired language model for code refinement.



## Dataset_prone.py

This project provides a set of tools for processing and refining SFT code datasets. It includes functionality for deduplication, comment removal, and dataset statistics generation.

### Features

- Traverse and process JSONL files containing Python code snippets
- Remove comments from code (optional)
- Deduplicate data based on problem descriptions
- Generate statistics about the processed dataset
- Flexible selection rules for including/excluding specific data sources

### Usage

1. Set the `basedir` variable in the `main()` function to the directory containing your JSONL files.
2. Adjust the `outputdir` variable in the `readfile()` function to specify where processed data should be saved.
3. Run the script:

```bash
python Dataset_prone.py
```


