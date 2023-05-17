# Financial Analysis Made Easy
In today's complex financial landscape, the ability to efficiently analyze financial statements is crucial for investors, businesses, and regulators. However, with the vast amounts of data and intricate reporting structures, extracting specific insights from these documents can be challenging. Our project aims to leverage deep learning and other recent AI advancements to quickly and succintly provide high-quality answers to user-generated questions about financial statements.

## Dataset
We plan to leverage the FinQA dataset created by [Chen et al.](https://arxiv.org/abs/2109.00122) Using earnings reports of S&P 500 companies, it contains 8,281 financial question-answer pairs, along with their numerical reasoning processes.

## Model
This project leverages the OpenAI API to [fine-tune](https://platform.openai.com/docs/guides/fine-tuning) the GPT-3 base model. Specifically, we're using `ada`.

## Usage
`finQA.ipynb` contains all the code needed to replicate our project results from start to finish. It processes the data from the original FinQA paper, fine-tunes the model, evaluates the result, etc. Users will, however, need to include their own OpenAI API key in a `api_key.txt` file. `evaluate.py` is also provided to reproduce accuracy figures for the fine-tuned model.

The fully processed training and validation data are stored in `formatted_data_prepared.jsonl` and `formatted_valid_data_prepared.jsonl` respectivley. Overall model performance statistics can also be found in `training_stats.csv`.

This repository was created for the final project for Deep Learning at AIT by Max Kan and Taichi Kato.

-------


# Deep Learning for Financial Question Answering
### Max Kan and Taichi Kato

**Abstract:**
This project presents a novel approach to Question Answering (QA) specifically tailored for financial documents. We propose a method to fine-tune a generative pre-trained transformer (GPT-3) for financial QA tasks. The paper evaluates the performance of the proposed method and demonstrates substantial improvements over existing solutions, highlighting the potential for future applications of deep learning in financial analysis.

## 1. Introduction
Financial documents contain vast amounts of valuable data, intricately embedded within dense text and complex tables. The ability to effectively and efficiently extract specific insights from these documents is paramount for many stakeholders, including investors, regulators, and businesses. However, the complexity and abundance of this information present significant challenges.

## 2. Previous Solutions
Traditional methods of extracting information from financial documents involve either manual extraction or Extractive QA methods, which are based on models trained on generic datasets like the Stanford Question Answering Dataset (SQuAD). These methods, while effective for certain types of queries, struggle with the specialized language and structure of financial documents. They are also not designed to generate new text, limiting their utility for more complex queries. Symbolic operators, like Mathematica or Wolfram Alpha, provide another avenue for answering queries, but these methods often require manual inputs and lack the natural language understanding capabilities of deep learning models.

## 3. Our Solution — Dataset
To address these challenges, we leverage a specialized dataset, FinQA, which contains financial question-answer pairs derived from the earnings reports of S&P 500 companies. It comprises 8,281 question-answer pairs and includes detailed numerical reasoning processes.

## 4. Our Solution — Generative QA
We transform the problem from an extractive QA task to a generative one, modeling the mapping between a context-question pair to a corresponding answer. Generative QA models, such as GPT-3, are suited for this task as they are designed to generate new text based on the provided context and input.

## 5. Data Pre-processing
We pre-process the data to accommodate the requirements of our model. The text is tokenized and encoded into numerical representations which can be processed by our deep learning models.

## 6. Our Solution — Models/Evaluation
We initially used a simple seq2seq model as a benchmark, achieving an accuracy of 9%. To improve the performance, we moved to transformer-based models, specifically the fine-tuned T5-small and fine-tuned text-davinci-03 models. These models provided substantial improvements, with the fine-tuned text-davinci-03 model achieving an accuracy of 61%, approaching the state-of-the-art performance of 68.96% by the ELASTIC model.

## 7. Our Solution — Conclusion
Our research indicates that while language models like GPT-3 can show considerable prowess in handling intricate reasoning tasks, they are not infallible. We have observed instances of miscalculations and hallucinations, indicating the models may occasionally generate inaccurate or irrelevant responses. These are inherent limitations of purely data-driven, neural network-based models, which lack the grounded semantics provided by symbolic systems.

However, we find the analogy provided by Stephen Wolfram in his article “What is ChatGPT Doing and Why Does it Work?” highly relevant. Wolfram suggests that these language models are analogous to an elaborate lookup system, one that is trained to match prompts to responses based on patterns it has seen during training. They do not actually ‘understand’ in the human sense, but instead generate text that ‘sounds right’ based on patterns seen in its training data.

While this quality allows language models to generate impressively coherent and contextually relevant text, it also leads to their limitations. Without any real-world grounding or internal symbolic processes, these models may produce responses that ‘sound right’ but are logically flawed or incorrect.

Symbolic systems, such as Mathematica or Wolfram Alpha, offer a powerful complement to these language models. These systems process information using grounded, formal logic, providing the ability to handle complex computations and reasoning tasks with high accuracy. However, these systems often lack the intuitive interfaces and natural language processing capabilities of neural network models.

The combination of these two approaches - the contextual awareness and natural language capabilities of language models, with the robust logical processing of symbolic systems - could present a compelling solution. Language models can serve as intuitive interfaces for users to interact with symbolic systems, translating natural language queries into formal logic, and returning the results in a human-readable format.
This idea opens up new avenues for future research. Exploring ways to integrate these two systems could yield a robust, intelligent system capable of effectively handling complex reasoning tasks in financial documents, and potentially in many other domains as well.

We propose further work to enhance the robustness of the fine-tuned text-davinci-03 model and to reduce instances of miscalculations and hallucinations. An exploration into combining neural network-based models with symbolic programs to create a more accurate and reliable solution would be a significant contribution to this field. A more comprehensive investigation into how the model can be applied to different types of financial documents, such as financial news or analyst reports, could also yield valuable insights. The potential of such a combination in providing an intuitive, yet powerful interface for symbolic reasoning tasks is immense and exciting.
