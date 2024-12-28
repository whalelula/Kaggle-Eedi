

VER = "llm_n6"
NUM_PROC = 8
DATA_PATH_TRAIN = "../data/misconception_data_all_train_gpt-4o.csv"
DATA_PATH_VAL = "../data/val_misconceptions.csv"
DATA_PATH_MAP = "../data/misconception_mapping.csv"
MODEL_PATH = "../qwen2-5-32b-instruct-quant"
BATCH_SIZE = 4  #
DEBUG = False
from transformers import AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class CustomSimCSEModel(nn.Module):
    def __init__(self, path, config, quantization_config, emb_size=1024, sentence_pooling_method='last', normlized=True, temperature=0.02):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, config=config, quantization_config=quantization_config)
        self.config = self.model.config
        #self.proj_head = nn.Linear(config.hidden_size, emb_size)
        self.sentence_pooling_method = sentence_pooling_method
        self.normlized = normlized
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.0)
        
    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
        
    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            return self.last_token_pool(hidden_state, mask)
        
    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                             return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        #p_reps = p_reps.to(torch.float32)
        #p_reps = self.proj_head(p_reps)
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()
    
    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
        
    def forward(self, query, doc):
        query_emb = self.encode(query)
        doc_emb = self.encode(doc)
        scores = self.compute_similarity(query_emb, doc_emb) / self.temperature
        scores = scores.view(query_emb.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        loss = self.cross_entropy(scores, target)
        return dict(
            loss=loss,
            scores=scores,
            query_emb=query_emb,
            doc_emb=doc_emb,
        )
#
import os
import copy
from dataclasses import dataclass
import random
import math

import numpy as np
import pandas as pd
import torch
import datasets
from datasets import Dataset
import transformers
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from tqdm import tqdm
from functools import partial
from torch import Tensor

#
@dataclass
class Config:
    debug: bool = DEBUG
    output_dir: str = f"output-{VER}"
    checkpoint: str = MODEL_PATH
    q_max_length: int = 256
    p_max_length: int = 50
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = 8
    per_device_eval_batch_size: int = 8
    n_epochs: int = 2 if debug else 16
    freeze_layers: int = 0
    lr: float = 3e-5
    warmup_steps: int = 3 if debug else 50
    lora_r: int = 128
    lora_alpha: float = 256
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    temperature: float = 0.05

# %%
config = Config()

# %%
training_args = TrainingArguments(
    output_dir = config.output_dir,
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=4 if config.debug else 50,
    eval_strategy="steps" if config.debug else "epoch",
    eval_steps=4 if config.debug else None,
    save_strategy="steps" if config.debug else "epoch",
    save_steps=4 if config.debug else None,
    optim=config.optim_type,
    fp16=False, 
    bf16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
    max_grad_norm=1.0,

    gradient_checkpointing=True, # this doesn't work correctly for some reason
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_first_step=True,
    lr_scheduler_type='cosine', # "cosine" or "linear" or "constant" (default is linear)
    metric_for_best_model="recall_at_25",
    greater_is_better=True,  
    save_total_limit=1,
    load_best_model_at_end=False,
    #resume_from_checkpoint=True
)

# %%
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj","o_proj",
                    "down_proj","up_proj","gate_proj"
                    ],
    #layers_to_transform=[i for i in range(32) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    #task_type=TaskType.FEATURE_EXTRACTION,
    #modules_to_save=["proj_head"]
)

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    #llm_int8_skip_modules=["proj_head"]
)

# %%
model_config = AutoConfig.from_pretrained(config.checkpoint)

# %%
tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

# %%
#tokenizer.truncation_side = "left"

# %%
model = CustomSimCSEModel(config.checkpoint, config=model_config, quantization_config=bnb_config, temperature=config.temperature)

# %%
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# %%
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
# load data
misconception_mapping = pd.read_csv(DATA_PATH_MAP)
df = pd.read_csv(DATA_PATH_TRAIN)
#ext_df = pd.read_csv(DATA_PATH_TRAIN2)
val_misconid_df = pd.read_csv(DATA_PATH_VAL).head(100)
#ext_df['origin_QuestionId'] = ext_df['QuestionId'].map(lambda x:int(x.split('_')[-1]))
#ext_df = ext_df[~ext_df['origin_QuestionId'].isin(val_misconid_df['QuestionId'])].reset_index(drop=True)
#del ext_df['origin_QuestionId']

#print(f"Ext Train data length: {len(ext_df)}")

#df = pd.concat([ext_df, df], ignore_index=True)
print(f"Concat Train data length: {len(df)}")

# %%
task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def create_train_df(train_df, misconception_mapping, is_train=True):
    train_data = []
    for _,row in train_df.iterrows():
        for c in ['A','B','C','D']:
            if is_train:
                misconception_id = row[f"Misconception{c}Id"]
                if np.isnan(misconception_id):
                    misconception_id = -1
                    doc_text = row[f'Misconception{c}Name']
                misconception_id = int(misconception_id)
            if c == row['CorrectAnswer']:
                continue
            if f'Answer{c}Text' not in row:
                continue
            real_answer_id = row['CorrectAnswer']
            real_text = row[f'Answer{real_answer_id}Text']
            query_text =f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{row[f'Answer{c}Text']}"
            row['query'] = get_detailed_instruct(task_description,query_text)
            row['answer_name'] = c
            if is_train and misconception_id != -1:
                doc_text = misconception_mapping.iloc[misconception_id]['MisconceptionName']
            row['doc'] = doc_text
            row['answer_id'] = misconception_id
            train_data.append(copy.deepcopy(row))
    new_train_df = pd.DataFrame(train_data)
    return new_train_df

# %%
new_train_df = create_train_df(df, misconception_mapping)

# %%
new_train_df.shape

# %%
#val_misconid_df = pd.read_csv(DATA_PATH_VAL)
val_misconceptions = val_misconid_df['answer_id'].tolist()

# %%
train_df = new_train_df[~new_train_df['answer_id'].isin(val_misconceptions)].reset_index(drop=True)
val_df = new_train_df[new_train_df['answer_id'].isin(val_misconceptions)].reset_index(drop=True)

print(f"训练集样本数量: {len(train_df)}")
print(f"验证集样本数量: {len(val_df)}")

print(f"训练集占比: {len(train_df)/len(new_train_df):.1%}")
print(f"验证集占比: {len(val_df)/len(new_train_df):.1%}")

# %%
train_df = train_df.dropna(subset=['query','doc'])

# %%
train_df['order_index'] = list(range(len(train_df)))
val_df['order_index'] = list(range(len(val_df)))

# %%
if config.debug:
    train_df = train_df.head(256)
    val_df = val_df.head(64)
    print(f"Debug 训练集样本数量: {len(train_df)}")
    print(f"Debug 验证集样本数量: {len(val_df)}")

# %%
train_ds = Dataset.from_pandas(train_df[['query','doc']])
val_ds = Dataset.from_pandas(val_df[['query','doc']])

# %%
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = config.q_max_length
    passage_max_len: int = config.p_max_length

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score


    def mask_pad_token(self,q):
        if random.random()>0.9:
            tensor = q['input_ids'].float()
            # 创建一个与原始张量形状相同的随机张量
            mask = torch.rand(tensor.shape)

            # 设置阈值，将大于阈值的部分设置为1，小于阈值的部分设置为0
            mask = (mask > 0.9).float()

            # 使用mask张量将原始张量中的一部分元素设置为2
            tensor = tensor * (1 - mask) + 2 * mask
            tensor = tensor.long()
            q['input_ids'] = tensor
        return q


    def __call__(self, features):
        query = [f["query"] for f in features]
        passage = [f["doc"] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        q_collated = self.mask_pad_token(q_collated)

        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        d_collated = self.mask_pad_token(d_collated)

        return {"query": q_collated, "doc": d_collated}

# %%
data_collator = EmbedCollator(tokenizer)

# %%
#metric related
misconception_mapping['query'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']

# %%
def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

# %%
def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

# %%
def inference(df, model, tokenizer, max_length, device):
    batch_size = 8
    sentences = list(df['query'].values)
    pids = list(df['order_index'].values)
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in range(0, len(sentences), batch_size):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
        features = batch_to_device(features, device)
        with torch.no_grad():
            embeddings = model.encode(features)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result

def recall(actual, predicted, k=25):
    return len(set(actual) & set(predicted[:k])) / len(actual)

def mrecall(actual, predicted, k=25):
    return np.mean([recall(a,p,k) for a,p in zip(actual, predicted)])

def compute_metric(p, df, misconception_mapping, model, tokenizer, max_length, device):
    top_ks = [25]
    query_embeddings = inference(df, model, tokenizer, max_length, device)
    doc_embeddings = inference(misconception_mapping, model, tokenizer, config.p_max_length, device)
    sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(doc_embeddings.values())])
    index_text_embeddings_index = {index: paper_id for index, paper_id in
                                         enumerate(list(doc_embeddings.keys()))}
    predicts_test = []
    for _, row in df.iterrows():
        query_id = row['order_index']
        query_em = query_embeddings[query_id].reshape(1, -1)
        
        cosine_similarity = np.dot(query_em, sentence_embeddings.T).flatten()

        sort_index = np.argsort(-cosine_similarity)[:max(top_ks)]
        pids = [index_text_embeddings_index[index] for index in sort_index]
        predicts_test.append(pids)

    metric_dict = {}
    for i in top_ks:
        recall_score = mrecall([[data] for data in df['answer_id'].values],predicts_test, i)
        map_score = mapk([[data] for data in df['answer_id'].values],predicts_test, i)
        metric_dict[f"recall_at_{i}"] = recall_score
        metric_dict[f"map_at_{i}"] = map_score
    return metric_dict

trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=train_ds, 
                  eval_dataset=val_ds, 
                  data_collator=data_collator,
                  compute_metrics=partial(compute_metric, df=val_df, misconception_mapping=misconception_mapping, model=model, tokenizer=tokenizer, max_length=512, device="cuda")
                  )

trainer.can_return_loss = True
trainer.train()
trainer.save_model(f"{config.output_dir}/last")
