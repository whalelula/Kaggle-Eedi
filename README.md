# Eedi - Mining Misconceptions in Mathematics

Refer to [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview) for the overview of the task and data. 

### 1 Data Synthesis

Used GPT-4o to generate data with the prompt below:
```
Your task is to generate Multiple Choice Questions (MCQs) that diagnose the following misconceptions:
<misconceptions>
{cluster_misconceptions}
</misconceptions>

Here are reference MCQs that demonstrate the questions and misconceptions:
<reference_mcqs>
{reference_mcqs}
</reference_mcqs>

First, in your first key-value pair (key is "reference_analysis"), analyze the reference MCQs carefully:
1. Understand how to derive the correct answer and how the wrong answers map to respective misconceptions
2. Note the style, difficulty level, and precision of language used

Then, in your first key-value pair (key is "MCQ_generation"), generate {num_mcqs} new MCQs that diagnose this conception "{missing_misconception}" , following these guidelines:
1. brainstorm mathematical contexts where this misconception commonly appears
2. generate the problem, solve it yourself, design 1 right answer, and 3 wrong answers leading to specific misconceptions
3. Make questions challenging enough that students must demonstrate real understanding
4. ensure the answers are plausible and stem from genuine misconceptions, not careless errors
5. Use precise mathematical language matching the style of reference MCQs
6. Keep the construct name and subject name as short as possible
7. Follow the json format of reference MCQs.


Additionaly, for each answer of a MCQ, add a key-value pair and write down your reasoning process.
For example: {{'SubjectName':'write subject of this question','ConstructName':'write construct of this question','QuestionText':'write question text',\
'MisconceptionAName':'write down a misconception related to this question as the misconceptionA ','ReasoningA':'write down how misconceptionA leads to answerA','AnswerAText':'write down the answer misonceptionA leads to',\
'MisconceptionBName':'write down a misconception related to this question as the misconceptionB ','ReasoningB':'write down how misconceptionB leads to answerB','AnswerBText':'write down the answer misonceptionB leads to',\
'MisconceptionCName':'write down a misconception related to this question as the misconceptionC ','ReasoningC':'write down how misconceptionC leads to answerC','AnswerCText':'write down the answer misonceptionC leads to',\
'MisconceptionDName':'write down a misconception related to this question as the misconceptionD ','ReasoningD':'write down how misconceptionD leads to answerD','AnswerDText':'write down the answer misonceptionD leads to',\
'CorrectAnswer':'write the correct answer(A/B/C/D)','CorrectAnswerReasoning':'write down why the correct answer is correct'}}
```
Essentials:
- Chain-of-Thought prompting: tell the model to write down thinking process, and to solve the math problem first.
- Few-shot examples: use bge-embedding for misconceptions, and select most similar 4 misconceptions for the target misconception by calculating cosine similarity. by using these 5 similar misconceptions as referece, GPT can refer to similar mathmatical contexts, and put similar misconceptions in one MCQ. These data will help to improve model's ability of distinguishing similar misconception in the training process.
- Detailed Instructions: will avoid misunderstandings, and teach gpt to provide data with better alignment to the official data.
- LLM-as-judge: use gpt-4o to filter out wrong solutions (MCQs that give the wrong solution as the correct answer), in which the misconception is also not consistent. 

Also included MalAlgoQA dataset and other public data from the forum.  

### 2 Retriever

Referencing the simCSE structure(see below), use contrastive learning to fine-tune several qwen models and ensemble them to lift the scores. Evaluated the performance by map@25 and recall@25. And choose the model mainly based on recall@25, because it's more important during the retrieving stage. 
Hyper-parameters: per_device_batch_size = 4, n_epochs = 16, learning_rate = 3e-5, lora_rank = 128, gradient_accumulation_steps=8
```
class CustomSimCSEModel(nn.Module):
    def __init__(self, path, config, device, quantization_config, top_linear=True, emb_size=1024, sentence_pooling_method='last', normlized=True, temperature=0.02):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, config=config, quantization_config=quantization_config, trust_remote_code=True, device_map=device)
        self.config = self.model.config
        self.top_linear = top_linear
        if self.top_linear:
            self.proj_head = nn.Linear(config.hidden_size, emb_size)
            self.proj_head.to(device)
        self.sentence_pooling_method = sentence_pooling_method
        self.normlized = normlized
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
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
        p_reps = p_reps.to(torch.float32)
        if self.top_linear:
            p_reps = self.proj_head(p_reps)
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
```
### 3 Ensemble
ensembled qwen-14b-awq, qwen-14b-instruct, qwen-32b-instruct for the retrieving stage. 

### 4 Re-ranker
Due to time limit and resource constraint, we didn't fine-tune the re-ranker. We use qwen-32b-instruct directly to give the misconception rationale, and rank the retrieved misconception according to semantic similarity with the rationale give by qwen-32b-instruct. 
```
Here is a mathematics question about 
Curriculum knowledge: {constructName}({subjectName})
Question: {problem}
Incorrect Answer: {wrong_ans}
Correct Answer: {correctAnswerValue}
    
You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
No need to give the reasoning process and do not use "The misconception is" to start your answers.
There are some relative and possible misconceptions below to help you make the decision:

{retrival}
```


# References
- https://www.kaggle.com/datasets/wuwenmin/bge-large-en-v1-5
- https://www.kaggle.com/datasets/syzong/qwen2-5-14b-instruct
- https://www.kaggle.com/datasets/gmhost/qwen2-5-32b-instruct-quant
- https://www.kaggle.com/datasets/abdurrafae/vllm-t4-fix
- https://www.kaggle.com/datasets/eugenkrylov/vllm-0-6-3-post1-wheels
- https://www.kaggle.com/datasets/emiz6413/lmsys-wheel-files
- https://www.kaggle.com/datasets/nbroad/hf-libraries
- https://www.kaggle.com/models/anhvth226/2211-lora-14b/Transformers/default/1
- https://www.kaggle.com/models/anhvth226/qw14b-awq/Transformers/default/1
- https://www.kaggle.com/code/ironbar/making-wheels-of-necessary-packages-for-vllm
- https://www.kaggle.com/models/takanashihumbert/qwen2.5/Transformers/32b-instruct-awq/1
