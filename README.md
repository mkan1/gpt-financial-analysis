# Financial Analysis Made Easy
In today's complex financial landscape, the ability to efficiently analyze financial statements is crucial for investors, businesses, and regulators. However, with the vast amounts of data and intricate reporting structures, extracting specific insights from these documents can be challenging. Our project aims to leverage deep learning and other recent AI advancements to quickly and succintly provide high-quality answers to user-generated questions about financial statements.

## Dataset
We plan to leverage the FinQA dataset created by [Chen et al.](https://arxiv.org/abs/2109.00122) Using earnings reports of S&P 500 companies, it contains 8,281 financial question-answer pairs, along with their numerical reasoning processes.

## Model
This project leverages the OpenAI API to [fine-tune](https://platform.openai.com/docs/guides/fine-tuning) the `ada` base model. `ada` is a generic next word predictor that does not come with any instruction following training. We chose it due to budget constraints: it is the cheapest and fastest model to train.

## Usage
`finQA.ipynb` contains all the code needed to replicate our project results from start to finish. It processes the data from the original FinQA paper, fine-tunes the model, evaluates the result, etc. Users will, however, need to include their own OpenAI API key in a `api_key.txt` file.

The fully processed training and validation data are stored in `formatted_data_prepared.jsonl` and `formatted_valid_data_prepared.jsonl` respectivley. Overall model performance statistics can also be found in `training_stats.csv`.

This repository was created for the final project for Deep Learning at AIT by Max Kan and Taichi Kato.
