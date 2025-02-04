# Eedi - Mining Misconceptions in Mathematics

Sharing our 55th place solution to the kaggle competition, please refer to [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview) for the overview of the task and data. 

# 1 Thoughts

Even the most advanced LLM is not good at counterfactual reasoning ([MalAlgoQA](https://arxiv.org/abs/2407.00938)), likely due to the lack of negative examples(what-if scenarios) during pre-training. Therefore, refinement of extra data and fine-tuning is needed in this competition to enhance the robustness of pre-trained models on distinguishing specific misconception rationales. Our general thoughts for winning this competition are :

- data curation: Use GPT-4o to generate extra data and LLM-as-judge to ensure the quality. LLM-as-judge also needs extra prompt crafting, becuase [LLM cannot self-correct reasoning yet](https://arxiv.org/abs/2310.01798)(LLMs is bad at self-correct reasoning without human feedbacks.)
- contrastive learning: contrastive learning should help lifting the discriminative power over misconception rationales.
- chain-of-thought distillation: due to computation constraint, we cannot fine-tune the largest LLM, but distillation of CoT generated by larger models should help lift the reasoning ability of smaller models. (However, we didn't have the time to actually implement this step.)


# 2 Pipeline

## 2.1 Data Synthesis

Firstly, use GPT-4o to generate MultipleChoice-QA for missing misconceptions. 
Used the prompt showing below:
```
You are a Mathematics teacher. Your task is to generate Multiple Choice Questions (MCQs) that diagnose the following misconceptions:
    <misconceptions>
    {cluster_misconceptions}
    </misconceptions>

Here are reference MCQs that demonstrate the questions and misconceptions:
    <reference_mcqs>
    {reference_mcqs}
    </reference_mcqs>

First, analyze the reference MCQs carefully:
    1. Understand how to derive the correct answer and how the wrong answers map to respective misconceptions
    2. Note the style, difficulty level, and precision of language used
    3. List misconceptions still need coverage

Then, genrate {num_mcqs} new MCQs that diagnose misconceptions not already covered by the reference MCQs.
For each needed misconception, in your <reasoning> tag, show your process following these guidelines:
    1. demonstrate the misconception the question is covering and explain why is it a misconception using your math knowledge 
    2. brainstorm mathematical contexts where it commonly appears
    3. solve the probelm yourself and then design wrong answers leading to specific miesconceptions
    4. Make questions challenging enough that students must demonstrate real understanding
    5. ensure the answers are plausible and originate from genuine misconceptions, not careless errors
    6. Use precise mathematical language matching the style of reference MCQs
    7. Keep the construct name and subject name as short as possible
```
The essential is to properly guide LLM NOT to miss any critical thinking step. For example, "This sentence "demonstrate the misconception the question is covering and explain why is it a misconception using your math knowledge " is important for the model to truly understand the misconception and design questions that lead to it. For example, this misconception "When measuring a reflex angle, gives the acute or obtuse angle that sums to 360 instead" wants to examine student's understanding that a reflex angle is more than 180° and less than 360°, especially that a relfex angle should be more than 180°.

BEFORE adding this sentence, generated result is below. In this question, students don't have to know that a reflex should be more than 180° to get the correct answer.
```
Subject: Measuring Angles  Construct: Measure reflex angles
Question: What is the measure of the reflex angle if the angle between the two lines is \( 110^\circ \)?
AnswerA: \( 250^\circ \)
AnswerB: \( 70^\circ \)
AnswerC: \( 110^\circ \)
AnswerD: \( 180^\circ \)
CorrectAnswer: A.
Misconception behind answerB: Gives the acute angle instead of the reflex by subtracting from \( 180^\circ \)
Misconception behind answerC: Gives the angle as it is, without calculating the reflex angle 
Misconception behind answerD: Confuses the concepts of supplementary and reflex angles 
```

AFTER adding this sentence, geneared result is below. Though the answers are still flawed(e.g. answerD is also correct. which could be refied later by llm-as-judge),this question better matches the misconception and succefully recalled the knowledge that a reflex angle should be more than 180°.
```
Question: Identify the reflex angle among the following options:
![Diagram with angles labeled: (a) 45° (b) 135° (c) 225° (d) 315°]()
  
AnswerA: \( 45^{\circ} \)  
AnswerB: \( 135^{\circ} \)  
AnswerC: \( 225^{\circ} \)  
AnswerD: \( 315^{\circ} \)  
  
CorrectAnswer: C  
  
Misconception behind answer A: Confuses a reflex angle with an acute angle.  
Misconception behind answer B: Confuses a reflex angle with an obtuse angle.  
Misconception behind answer D: Believes that angles above 270° are reflex angles, but does not correctly identify a reflex angle that needs to be greater than 180° and less than 360°. 
```

Then, use GPT-4o as judge to refine the misconception rationales to fix logical flawes like answer D above. 
Used the prompt showing below:
```
You will analyze how well an incorrect answer reflects a suspected misconception in a mathematics problem. Your goal is to determine whether there is a clear, logical connection between the misconception and the wrong answer.

Here is the problem with both correct and incorrect answers. The suspected misconception is also provided:
<problem>
{generated_mcqs} 
</problem>

First, analyze the problem in <thinking>:
<thinking>
1. Solve the problem to verify the correct answer
2. Examine how someone holding the suspected misconception would approach the problem
3. Trace the logical path from misconception to incorrect answer
4. Identify any gap or inconsistencie in this connection
</thinking>

Then examine whether the misconception perfectly leads to the incorrect answer, score from 1-5:
<evaluation>
- 5: Perfect - wrong answer is direct result of misconception
- 4: Strong - clear logical path from misconception to answer
- 3: Moderate - connection exists but has some gaps
- 2: Weak - connection is unclear or requires assumptions
- 1: None - misconception does not explain wrong answer
</evaluation>

Important guidelines:
- Focus solely on the logical connection between misconception and wrong answer
- Do not speculate about other possible misconceptions
- Be specific about how the misconception leads to the error
- Flag and deduct scores if any assumptions are required to connect misconception to answer
- Consider whether a student with this misconception would consistently arrive at this wrong answer
```
The judge successfully detected the flaws in answer D.
```
Answer D (315°): Score 3 (Moderate) — The misconception that angles greater than 270° could be reflex angles leads to a selection of 315°, though the reason for this choice weakens the overall logical connection
```

Additionally, to add to the diversity. We also included the MalAlgoQA dataset. 

## 2.2 Retriever

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


## 2.3 Re-ranker
Due to time limit, we didn't fine-tune the re-ranker. We use qwen-32b-instruct directly to give the misconception rationale, and rank the retrieved misconception according to semantic similarity with the rationale give by qwen-32b-instruct. 
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
-https://www.kaggle.com/datasets/wuwenmin/bge-large-en-v1-5
-https://www.kaggle.com/datasets/syzong/qwen2-5-14b-instruct
-https://www.kaggle.com/datasets/gmhost/qwen2-5-32b-instruct-quant
-https://www.kaggle.com/datasets/abdurrafae/vllm-t4-fix
-https://www.kaggle.com/datasets/eugenkrylov/vllm-0-6-3-post1-wheels
-https://www.kaggle.com/datasets/emiz6413/lmsys-wheel-files
-https://www.kaggle.com/datasets/nbroad/hf-libraries
-https://www.kaggle.com/models/anhvth226/2211-lora-14b/Transformers/default/1
-https://www.kaggle.com/models/anhvth226/qw14b-awq/Transformers/default/1
-https://www.kaggle.com/code/ironbar/making-wheels-of-necessary-packages-for-vllm
-https://www.kaggle.com/models/takanashihumbert/qwen2.5/Transformers/32b-instruct-awq/1
