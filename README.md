# Financial Analysis Made Easy
In today's complex financial landscape, the ability to efficiently analyze financial statements is crucial for investors, businesses, and regulators. However, with the vast amounts of data and intricate reporting structures, extracting specific insights from these documents can be challenging. Our project aims to leverage deep learning and other recent AI advancements to quickly and succintly provide high-quality answers to user-generated questions about financial statements.

## Dataset
We plan to leverage the FinQA dataset created by [Chen et al.](https://arxiv.org/abs/2109.00122) Using earnings reports of S&P 500 companies, it contains 8,281 financial question-answer pairs, along with their numerical reasoning processes.

## Model
We plan to [fine-tune](https://platform.openai.com/docs/guides/fine-tuning) the ChatGPT model using the publically available [API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) to improve performance in answering finance-related queries.

## Usage
The train_basic.py contains all code needed for a benchmark model using a simple seq2seq model. You can use it to pocess the dataset, train a seq2seq model, run inference on it, and evaluate the model using BLEU. Simply run `python train.py`


Future work will be added to the same repo.


This repository was created for the final project for Deep Learning at AIT by Max Kan and Taichi Kato.
