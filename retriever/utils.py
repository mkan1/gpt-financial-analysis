import time
import os
import sys
import shutil
import io
import subprocess
import re
import zipfile
import json
import copy
import torch
import random
import collections
import math
import numpy as np
import torch.nn.functional as F
from config import parameters as conf
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig

# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
print(os.popen('stty size', 'r').read())
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')


# def get_current_git_version():
#     import git
#     repo = git.Repo(search_parent_directories=True)
#     sha = repo.head.object.hexsha
#     return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def read_txt(input_path, log_file):
    """Read a txt file into a list."""

    write_log(log_file, "Reading: %s" % input_path)
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


def read_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in input_data:
        examples.append(read_mathqa_entry(entry, tokenizer))

    return input_data, examples, op_list, const_list


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 option,
                                 is_training,
                                 ):
    """Converts a list of DropExamples into InputFeatures."""
    res = []
    res_neg = []
    for (example_index, example) in tqdm(enumerate(examples)):
        features, features_neg = example.convert_single_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            option=option,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        res.extend(features)
        res_neg.extend(features_neg)

    return res, res_neg



def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


class DataLoader:
    def __init__(self, is_training, data, batch_size=64, shuffle=True):
        """
        Main dataloader
        """
        self.data_pos = data[0]
        self.data_neg = data[1]
        self.batch_size = batch_size
        self.is_training = is_training
        
        
        if self.is_training:
            random.shuffle(self.data_neg)
            if conf.option == "tfidf":
                self.data = self.data_pos + self.data_neg
            else:
                num_neg = len(self.data_pos) * conf.neg_rate
                self.data = self.data_pos + self.data_neg[:num_neg]
        else:
            self.data = self.data_pos + self.data_neg
            
            
        self.data_size = len(self.data)
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1

        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # drop last batch
        if self.is_training:
            bound = self.num_batches - 1
        else:
            bound = self.num_batches
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        if conf.option == "tfidf":
            random.shuffle(self.data)
        else:
            random.shuffle(self.data_neg)
            num_neg = len(self.data_pos) * conf.neg_rate
            self.data = self.data_pos + self.data_neg[:num_neg]
            random.shuffle(self.data)
        return

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)
        

        batch_data = {"input_ids": [],
                      "input_mask": [],
                      "segment_ids": [],
                      "filename_id": [],
                      "label": [],
                      "ind": []
                      }
        for each_data in self.data[start_index: end_index]:

            batch_data["input_ids"].append(each_data["input_ids"])
            batch_data["input_mask"].append(each_data["input_mask"])
            batch_data["segment_ids"].append(each_data["segment_ids"])
            batch_data["filename_id"].append(each_data["filename_id"])
            batch_data["label"].append(each_data["label"])
            batch_data["ind"].append(each_data["ind"])


        return batch_data




def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext



def retrieve_evaluate(all_logits, all_filename_ids, all_inds, output_prediction_file, ori_file, topn):
    '''
    save results to file. calculate recall
    '''
    
    res_filename = {}
    res_filename_inds = {}
    
    for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
        
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1],
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)
            
        
        
    with open(ori_file) as f:
        data_all = json.load(f)
        
    # take top ten
    all_recall = 0.0
    all_recall_3 = 0.0
    
    for data in data_all:
        this_filename_id = data["id"]
        
        this_res = res_filename[this_filename_id]
        
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        # sorted_dict = sorted_dict[:topn]
        
        gold_inds = data["qa"]["gold_inds"]
        
        # table rows
        table_retrieved = []
        text_retrieved = []

        # all retrieved
        table_re_all = []
        text_re_all = []
        
        correct = 0
        correct_3 = 0
        
        for tmp in sorted_dict[:topn]:
            if "table" in tmp["ind"]:
                table_retrieved.append(tmp)
            else:
                text_retrieved.append(tmp)
                
            if tmp["ind"] in gold_inds:
                correct += 1

        for tmp in sorted_dict:
            if "table" in tmp["ind"]:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)
                
        for tmp in sorted_dict[:3]:
            if tmp["ind"] in gold_inds:
                correct_3 += 1
                
        all_recall += (float(correct) / len(gold_inds)) 
        all_recall_3 += (float(correct_3) / len(gold_inds)) 
        
        data["table_retrieved"] = table_retrieved
        data["text_retrieved"] = text_retrieved

        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all

        
    with open(output_prediction_file, "w") as f:
        json.dump(data_all, f, indent=4)
        
    res_3 = all_recall_3 / len(data_all)
    res = all_recall / len(data_all)
    
    res = "Top 3: " + str(res_3) + "\n" + "Top 5: " + str(res) + "\n"
    
    
    return res
                
                
        
        
def retrieve_evaluate_private(all_logits, all_filename_ids, all_inds, output_prediction_file, ori_file, topn):
    '''
    save results to file. calculate recall
    '''
    
    res_filename = {}
    res_filename_inds = {}
    
    for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
        
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1],
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)

    with open(ori_file) as f:
        data_all = json.load(f)
    
    for data in data_all:
        this_filename_id = data["id"]
        
        this_res = res_filename[this_filename_id]
        
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        # table rows
        table_retrieved = []
        text_retrieved = []

        # all retrieved
        table_re_all = []
        text_re_all = []
        
        for tmp in sorted_dict[:topn]:
            if "table" in tmp["ind"]:
                table_retrieved.append(tmp)
            else:
                text_retrieved.append(tmp)


        for tmp in sorted_dict:
            if "table" in tmp["ind"]:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)
        
        data["table_restrieved"] = table_retrieved
        data["text_retrieved"] = text_retrieved

        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all

        
    with open(output_prediction_file, "w") as f:
        json.dump(data_all, f, indent=4)

    return "private, no res"

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()

def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                num = text
            else:
                num = None
    return num


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            assert cur_num_idx != -1
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "filename_id question all_positive \
            pre_text post_text table"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 filename_id,
                 retrieve_ind,
                 tokens,
                 input_ids,
                 segment_ids,
                 input_mask,
                 label):

        self.filename_id = filename_id
        self.retrieve_ind = retrieve_ind
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if conf.pretrained_model in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens


def _detokenize(tokens):
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program



# def get_tf_idf_query_similarity(allDocs, query):
#     """
#     vectorizer: TfIdfVectorizer model
#     docs_tfidf: tfidf vectors for all docs
#     query: query doc

#     return: cosine similarity between query and all docs
#     """
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     vectorizer = TfidfVectorizer(stop_words='english')
#     docs_tfidf = vectorizer.fit_transform(allDocs)
    
#     query_tfidf = vectorizer.transform([query])
#     cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    
#     # print(cosineSimilarities)
#     return cosineSimilarities


def wrap_single_pair(tokenizer, question, context, label, max_seq_length,
                    cls_token, sep_token):
    '''
    single pair of question, context, label feature
    '''
    
    question_tokens = tokenize(tokenizer, question)
    this_gold_tokens = tokenize(tokenizer, context)

    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }
    
    return this_input_feature

def convert_single_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features = []
    features_neg = []
    
    question = example.question
    all_text = example.pre_text + example.post_text

    if is_training:
        for gold_ind in example.all_positive:

            this_gold_sent = example.all_positive[gold_ind]
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 1, max_seq_length,
                cls_token, sep_token)

            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = gold_ind
            pos_features.append(this_input_feature)
            
        num_pos_pair = len(example.all_positive)
        num_neg_pair = num_pos_pair * conf.neg_rate
            
        pos_text_ids = []
        pos_table_ids = []
        for gold_ind in example.all_positive:
            if "text" in gold_ind:
                pos_text_ids.append(int(gold_ind.replace("text_", "")))
            elif "table" in gold_ind:
                pos_table_ids.append(int(gold_ind.replace("table_", "")))

        all_text_ids = range(len(example.pre_text) + len(example.post_text))
        all_table_ids = range(1, len(example.table))
        
        all_negs_size = len(all_text) + len(example.table) - len(example.all_positive)
        if all_negs_size < 0:
            all_negs_size = 0
                    
        # test: all negs
        # text
        for i in range(len(all_text)):
            if i not in pos_text_ids:
                this_text = all_text[i]
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question, this_text, 0, max_seq_length,
                    cls_token, sep_token)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "text_" + str(i)
                features_neg.append(this_input_feature)
            # table      
        for this_table_id in range(len(example.table)):
            if this_table_id not in pos_table_ids:
                this_table_row = example.table[this_table_id]
                this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question, this_table_line, 0, max_seq_length,
                    cls_token, sep_token)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "table_" + str(this_table_id)
                features_neg.append(this_input_feature)
                
    else:
        pos_features = []
        features_neg = []
        question = example.question

        ### set label as -1 for test examples
        for i in range(len(all_text)):
            this_text = all_text[i]
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, this_text, -1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "text_" + str(i)
            features_neg.append(this_input_feature)
            # table      
        for this_table_id in range(len(example.table)):
            this_table_row = example.table[this_table_id]
            this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, this_table_line, -1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "table_" + str(this_table_id)
            features_neg.append(this_input_feature)

    return pos_features, features_neg


def read_mathqa_entry(entry, tokenizer):

    filename_id = entry["id"]
    question = entry["qa"]["question"]
    if "gold_inds" in entry["qa"]:
        all_positive = entry["qa"]["gold_inds"]
    else:
        all_positive = []

    pre_text = entry["pre_text"]
    post_text = entry["post_text"]
    table = entry["table"]

    return MathQAExample(
        filename_id=filename_id,
        question=question,
        all_positive=all_positive,
        pre_text=pre_text,
        post_text=post_text,
        table=table)
      
    
if __name__ == '__main__':

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # text = "this is a -10"
    # res = tokenize(tokenizer, text, apply_basic_tokenization=False)
    
    # text = "<a>test test</a>"
    # print(cleanhtml(text))
    
    # root_path = "/mnt/george_bhd/zhiyuchen/"
    # outputs = root_path + "outputs/"
    
    # json_in = outputs + "test_20210408011241/results/loads/1/valid/nbest_predictions.json"
    # retrieve_evaluate(json_in)
    print("main method of utils.py")
    