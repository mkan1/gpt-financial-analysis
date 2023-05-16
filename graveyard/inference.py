import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

f = open('dataset/train.json')
data_dict = json.load(f)
data_list = [
    {
        "context": '\n'.join(data['pre_text'] 
                                + data['post_text'] 
                                + [json.dumps(data['table_ori']), json.dumps(data['table'])]),
        "question": data['qa']['question'],
        "answer": data['qa']['answer']
    } for data in data_dict
]

print(data_list[0])


# for i in range(min(len(data_list), 5)):
#     input_ids = tokenizer("question: " + data_list[i]["question"] + ", given context: " + data_list[i]["context"], return_tensors="pt").input_ids
#     outputs = model.generate(input_ids)
#     # print("contex:", "what is " + data_list[i]["question"] + ", given context: " + data_list[i]["context"])
#     print("predicted:", tokenizer.decode(outputs[0], skip_special_tokens=True))
#     print("actual:", data_list[i]["answer"])
#     print("\n")