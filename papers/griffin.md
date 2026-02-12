_2024-3-1_

# **Griffin: Mixing Gated Linear Recurrences with** **Local Attention for Efficient Language Models**


**Soham De** [* 1] **, Samuel L. Smith** [* 1] **, Anushan Fernando** [*1] **, Aleksandar Botev** [*1] **, George Cristian-Muraru** [*1] **,**
**Albert Gu** [2] **, Ruba Haroun** [1] **, Leonard Berrada** [1] **, Yutian Chen** [1] **, Srivatsan Srinivasan** [1] **, Guillaume Desjardins** [1] **,**
**Arnaud Doucet** [1] **, David Budden** [1] **, Yee Whye Teh** [1] **, Razvan Pascanu** [1] **, Nando De Freitas** [1] **and Caglar Gulcehre** [1]

*Equal contributions, 1Google DeepMind, 2Work done while at Google DeepMind


**Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but they are**
**difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences, and Griffin,**
**a hybrid model that mixes gated linear recurrences with local attention.** **Hawk exceeds the reported**
**performance of Mamba on downstream tasks, while Griffin matches the performance of Llama-2 despite**
**being trained on over 6 times fewer tokens. We also show that Griffin can extrapolate on sequences signifi-**
**cantly longer than those seen during training. Our models match the hardware efficiency of Transformers**
**during training, and during inference they have lower latency and significantly higher throughput. We**
**scale Griffin up to 14B parameters, and explain how to shard our models for efficient distributed training.**

### **1. Introduction**


Recurrent neural networks (RNNs) played a central role in the early days of deep learning and NLP research (Elman, 1990; Siegelmann and Sontag, 1991; Hochreiter and Schmidhuber, 1997; Mikolov et al.,
2010; Bahdanau et al., 2014; Sutskever et al., 2014), and achieved practical success in many applications,
including Google’s first end to end machine translation system (Wu et al., 2016). However in recent
years, both deep learning and NLP have been dominated by the Transformer architecture (Vaswani
et al., 2017), which interleaves multi-layer perceptrons (MLPs) and multi-head attention (MHA).
Transformers achieve better performance than RNNs in practice and are also very efficient at utilizing
modern hardware (Kaplan et al., 2020). Transformer-based large language models trained on massive
datasets collected from the web have achieved remarkable success (Brown et al., 2020; Rae et al., 2021;
Hoffmann et al., 2022; Touvron et al., 2023; Achiam et al., 2023; Gemini Team Google, 2023).


Despite their successes, Transformers are difficult to scale efficiently to long sequences due to the
quadratic complexity of global attention. Additionally, the linear growth of the Key-Value (KV) cache
with the sequence length makes Transformers slow during inference. Although Multi-Query Attention
(MQA) (Shazeer, 2019) partially mitigates this issue by reducing the cache size by a constant factor,
the cache still grows linearly in sequence length. Recurrent language models present a compelling
alternative as they compress the entire sequence into a fixed-sized hidden state which is updated
iteratively. However to replace Transformers, new RNN models must demonstrate not only comparable
performance at scale but also achieve similar hardware efficiency (Gu et al., 2021a; Mehta et al., 2022;
Smith et al., 2022; Orvieto et al., 2023b; Dao et al., 2022b; Poli et al., 2023; Gu and Dao, 2023).


In this work, we propose the RG-LRU layer, a novel gated linear recurrent layer, around which we design
a new recurrent block to replace MQA. We build two new models using this recurrent block: **Hawk**,
a model which interleaves MLPs with recurrent blocks, and **Griffin**, a hybrid model which interleaves
MLPs with a mixture of recurrent blocks and local attention (Beltagy et al., 2020). We show that:


1. Hawk and Griffin exhibit power law scaling between held-out loss and training FLOPs, up to and beyond 7B parameters (Figure 1(a)), as previously observed for Transformers (Kaplan et al., 2020).


2. Griffin achieves slightly lower held-out loss than strong Transformer baselines at all model scales.


_Corresponding author(s):_ _{sohamde, slsmith}@google.com_


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


|Col1|MQA<br>Hawk<br>Griffin|
|---|---|
|||
|||
|||


|Col1|Col2|Col3|1.8x<br>1.0x|Col5|Col6|Col7|3.3x<br>1.0x|Col9|Col10|Col11|6.6x<br>1.0x|Col13|Col14|Col15|Col16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||



(b) Maximum throughput at 1B parameter scale.



512 1024 2048 4096
Num tokens decoded



3.0


2.5


2.0



10 [18] 10 [19] 10 [20] 10 [21] 10 [22]

Training Flops


(a) Scaling curve during training



20000


15000


10000


5000


0



Figure 1 | a) Hawk, Griffin and our MQA Transformer baseline all show power law scaling between
held-out loss and training FLOPs, with Griffin achieving the lowest held-out loss at all FLOPs budgets.
The largest Griffin model shown has 14B parameters. b) Hawk and Griffin achieve significantly higher
throughput than our MQA Transformer, especially when the length of the sample increases.


3. We overtrain Hawk and Griffin on 300B tokens at a range of model scales. Hawk-3B exceeds the
reported performance of Mamba-3B (Gu and Dao, 2023) on downstream tasks, despite being
trained on half as many tokens. Griffin-7B and Griffin-14B match the performance of Llama-2
(Touvron et al., 2023) despite being trained on roughly 7 times fewer tokens (Section 3.2).


4. Both Hawk and Griffin achieve comparable training efficiency to Transformers on TPU-v3. Since
diagonal RNN layers are memory bound, we achieve this with a kernel for the RG-LRU layer,
implemented in Pallas (Bradbury et al., 2018), that minimizes memory transfers (Section 4).


5. During inference, both Hawk and Griffin achieve significantly higher throughput than MQA Transformers (Figure 1(b)), and they achieve lower latency when sampling long sequences (Section 5).


6. Griffin performs better than Transformers when evaluated on sequences longer than those seen
during training, and can also efficiently learn copying and retrieval tasks from training data
(Section 6). However, Hawk and Griffin perform less well than Transformers when we evaluate
pre-trained models on copying and exact-retrieval tasks without fine-tuning.

### **2. Model Architecture**


All our models contain the following components: (i) _a residual block_, (ii) _an MLP block_, and (iii) _a_
_temporal-mixing block_ . While (i) and (ii) are the same across all models, we consider three temporal
mixing blocks: _global Multi-Query Attention_ (MQA), _local (sliding-window) MQA_ and our proposed
_recurrent block_ . As part of the recurrent block we use the Real-Gated Linear Recurrent Unit (RG-LRU)

- a novel recurrent layer inspired by the Linear Recurrent Unit (Orvieto et al., 2023b).


The residual block, as shown in Figure 2(a), defines the global structure of our models and is inspired
by pre-norm Transformers (Xiong et al., 2020). After embedding the input sequence we pass it through
_𝑁_ such blocks ( _𝑁_ denoting the model depth), and then we apply RMSNorm (Zhang and Sennrich,
2019) to produce the final activations. To compute the token probabilities we apply a final linear layer
followed by a softmax. The weights of this layer are shared with the input embedding layer.


**2.1.** **Residual block**
The residual block contains two components, applied in order. The first component takes the hidden
state _𝑥_ and applies an RMSNorm (Zhang and Sennrich, 2019), followed by the temporal-mixing block.


2


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models





























Figure 2 | a) The main backbone of our mode architecture is the residual block, which is stacked _𝑁_
times. b) The gated MLP block that we use. c) The recurrent block that we propose as an alternative
to Multi Query Attention (MQA). It uses our proposed RG-LRU layer, defined in Section 2.4.


We then merge the output with a skip connection from _𝑥_ through addition. Similarly, the second
component applies RMSNorm, followed by the MLP block and then merges its output with a skip
connection from the input of the RMSNorm. This block is illustrated in Figure 2 (a).


**2.2.** **MLP block**
We use a gated MLP block (Dauphin et al., 2017) (illustrated in Figure 2(b)), which creates two
branches from its input of dimension _𝐷_ . We apply a linear layer with output dimension _𝑀𝐷_ on each
branch, where _𝑀_ denotes the expansion factor. For simplicity, we use _𝑀_ = 3 throughout this work. We
apply a GeLU non-linearity (Hendrycks and Gimpel, 2016) on one of the branches before merging them
by element-wise multiplication, similar to GeGeLU (Shazeer, 2020). However, in our MLP block, we
apply a final linear layer with output dimension _𝐷_ on the outputs of the GeGeLU layer.


**2.3.** **Temporal-mixing blocks**
The temporal-mixing block is the component of our model that aggregates hidden layer activations
at different temporal locations in the sequence. We consider three temporal-mixing blocks: global MQA
(Shazeer, 2019), local MQA (Beltagy et al., 2020) and our proposed _Recurrent block_ .


**Global multi-query attention** Unless otherwise stated, we use MQA rather than MHA to improve
the inference speeds of our Transformer baselines (Shazeer, 2019). We use a fixed head dimension
_𝐷ℎ𝑒𝑎𝑑_ = 128, and we fix the number of attention heads _𝐻_ such that _𝐻𝐷ℎ𝑒𝑎𝑑_ = _𝐷_ . This requires the model
dimension _𝐷_ to be a multiple of 128. We do not use any absolute positional embeddings, but we use
Rotary Position Embedding (RoPE) (Su et al., 2021) as a relative positional embedding.


**Local sliding window attention** One of the key disadvantages of using global attention is that its
computational complexity grows quadratically in the sequence length. To address this, several works
have started to adopt _local attention_ (Beltagy et al., 2020), also known as sliding window attention.
It allows each position to attend only to a fixed number of tokens in the past. This not only reduces
the computational FLOPs but also bounds the size of the KV cache to the size of window, making it no
longer quadratic in the sequence length. All other details are the same as the global MQA.


3


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


**Recurrent block** Our recurrent block (Figure 2(c)) is similar to the GSS block (Mehta et al., 2022)
and the block used by Mamba (Gu and Dao, 2023). We take the input of dimension _𝐷_ and apply two
linear layers with output dimension _𝐷𝑅𝑁𝑁_ in parallel, creating two branches. On the first branch, we
apply a small separable Conv1D layer, inspired by the Shift-SSM in H3 (Dao et al., 2022b), with a
temporal filter dimension of 4. Note that this Conv1D layer is very small, with just 4 _𝐷𝑅𝑁𝑁_ parameters.
We follow the Conv1D layer with our proposed RG-LRU layer (defined below.) On the second branch
we apply a GeLU nonlinearity and then merge the branches by element-wise multiplication. We then
apply a final linear layer with output dimension _𝐷_ .


**2.4.** **Real-Gated Linear Recurrent Unit (RG-LRU)**
Our proposed RG-LRU layer has a simple recurrence inspired by the Linear Recurrent Unit (LRU)
(Orvieto et al., 2023b), but incorporates a gating mechanism motivated by the literature on non-linear
RNNs, in particular LSTMs (Hochreiter and Schmidhuber, 1997) and GRUs (Chung et al., 2014). The
equations describing the layer are as follows:


_𝑟𝑡_ = _𝜎_ ( _𝑊𝑎𝑥𝑡_ + _𝑏𝑎_ ) _,_ _recurrence gate_ (1)

_𝑖𝑡_ = _𝜎_ ( _𝑊𝑥_ _𝑥𝑡_ + _𝑏𝑥_ ) _,_ _input gate_ (2)

_𝑎𝑡_ = _𝑎_ _[𝑐𝑟][𝑡]_ _,_ (3)


~~√~~
_ℎ𝑡_ = _𝑎𝑡_ ⊙ _ℎ𝑡_ −1 + 1− _𝑎_ [2] _𝑡_ [⊙(] _[𝑖][𝑡]_ [⊙] _[𝑥][𝑡]_ [)] _[.]_ (4)


The output of the layer is _𝑦𝑡_ = _ℎ𝑡_, and the non-linearity _𝜎_ in the equations is the sigmoid function. The
recurrent weight _𝑎_ in Equation (4) is diagonal. Hence all operations are element-wise. We parameterize
_𝑎_ in Equation (3) as _𝑎_ = _𝜎_ (Λ), where Λ is a learnable parameter. This guarantees that 0 ≤ _𝑎_ ≤ 1, ensuring
that the recurrence is stable. The variable _𝑐_ is a scalar-valued constant set to 8. For numerical stability,
in practice we compute _𝑎_ _[𝑐𝑟][𝑡]_ in log-space (see Appendix A). The layer has gates on both the input _𝑥_ and
the recurrent weight _𝑎_ . However, neither gate depends on the recurrent state _ℎ𝑡_ −1, which ensures that
the computation can be executed efficiently on device. We initialize both _𝑊𝑎_ and _𝑊𝑥_ using LeCun init
(LeCun et al., 2002). We initialize Λ such that _𝑎_ _[𝑐]_ is uniformly distributed between 0 _._ 9 and 0 _._ 999 at the
start of training, similar to Orvieto et al. (2023b). Unlike many recent works in the SSM literature, the
RG-LRU does not use initialization inspired by the theory of orthogonal polynomials (Gu et al., 2020),
and it also is not defined as the discretization of an underlying continuous system (Gu et al., 2021a).
Unlike the original LRU layer, we do not use complex algebra in the recurrence. While using complex
recurrences would lead to a more expressive layer (Orvieto et al., 2023a) we found that complex
recurrences were not beneficial for language modelling in practice, as also observed by Gu and Dao
(2023). [1]


**Gate behaviour** The _input gate 𝑖𝑡_ is similar to the one in LSTM, which can filter (or scale down) the
input _𝑥𝑡_ . However, to our knowledge, our recurrence gate _𝑟𝑡_ is different from other gating mechanisms
in the literature. For example, the _selection mechanism_ proposed in Mamba (Gu and Dao, 2023) is
comparable to the _update gate_ of GRUs which interpolates between the previous state and and the current
observation _𝑥𝑡_ . Its effect on the hidden state allows it to _reset its state and forget any information it holds_
_from the past_, similar to the forget gate in the LSTM. In contrast, our recurrence gate can approximately
interpolate between the standard LRU update from Orvieto et al. (2023a) and the previous hidden
state, which allows it to effectively _discard the input and preserve all information from the previous history_
(see Appendix A for further details). We believe the key role of this gate is to enable the model to achieve
super-exponential memory by reducing the influence of uninformative inputs.


1We suggest ablating the use of complex numbers for other modalities and provide more information about the
complex-valued version of the RG-LRU layer in Appendix B.


4


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models

### **3. Recurrent Models Scale as Efficiently as Transformers**


Scaling studies provide important insights into how to tune the hyperparameters of a model and its
behaviour at scale. Here, we define the models evaluated in our studies, and provide scaling curves
up to and beyond 7B parameters. Finally, we assess the performance of our models on downstream
tasks. We consider 3 model families in this work; (1) a MQA-Transformer baseline, (2) Hawk; our pure
RNN model, and (3) Griffin; our hybrid model which mixes recurrent blocks with local attention. We
define the key model hyper-parameters for models across a range of scales in Appendix C.


**MQA Transformer baseline** Our Transformer baseline uses the residual pattern and the gated MLP
blockdescribedinSection 2, incombinationwithMQA(Shazeer, 2019)andRoPE(Suetal., 2021).


**Hawk** The Hawk architecture uses the same residual pattern and MLP block as our Transformer
baseline, but we use the recurrent block introduced in Section 2.3 with a RG-LRU layer (see Section 2.4)
as our temporal mixing block, instead of MQA. We expand the width of the recurrent block by a factor
of approximately [4] _[𝐷][𝑅𝑁𝑁]_ [≈] [4] _[𝐷]_ [/][3) in order to roughly match the parameter count of a MHA block]

3 [(i.e.]
when both use the same model dimension _𝐷_ . [2] See Appendix C for precise hyper-parameters.


**Griffin** The key advantage of recurrent blocks over global attention is that they use a fixed state size
to summarize the sequence, whereas the size of MQA’s KV cache grows proportional to sequence length.
Sincelocalattention(Section 2.3)hasthesameproperty, mixingrecurrentblockswithlocalattentionpreserves this benefit. We have found this combination extremely effective, since local attention accurately
models the recent past, while the recurrent layers can transmit information across long sequences.


Griffin uses the same residual pattern and MLP block as our Transformer baseline. However unlike
both our MQA Transformer baseline and the Hawk model, Griffin uses a mixture of recurrent blocks
and MQA blocks. Specifically, we employ a layered structure by alternating two residual blocks with
a recurrent block followed by one residual block which uses the local (MQA) attention block described
in Section 2.3. Unless otherwise stated, the local attention window size is fixed to 1024 tokens.


**3.1.** **Scaling curves**
We present our main scaling results in Figure 1(a). All three model families are trained at a range of
model scales from 100M to 7B parameters, with an additional Griffin model with 14 billion parameters.
We increase the number of training tokens to be roughly proportional to the number of parameters
of the model, as prescribed by the Chinchilla scaling laws (Hoffmann et al., 2022). Models are trained
on the MassiveText dataset (Hoffmann et al., 2022), previously used to train Gopher (Rae et al., 2021)
and Chinchilla (Hoffmann et al., 2022), although we use a slightly different data subset distribution.
A sequence length of 2048 tokens was used (see Section 6 for results with longer sequences.) All
experiments use the AdamW optimizer (Loshchilov and Hutter, 2017). We tune the learning rate,
weight decay and _𝛽_ 2 parameters for small models, and use these runs to identify scaling rules for these
hyper-parameters which predict their optimal values for the 7B and 14B models.


All three model families demonstrate a linear scaling relationship between the validation loss and
training FLOPs (see Figure 1(a); note both axes are in log scale), as previously observed for Transformers
by Brown et al. (2020). Notably, Griffin achieves lower validation loss than the Transformer baseline
across all FLOPs budgets despite not using any global attention layers. Hawk on the other hand achieves
slightly higher validation loss, but this gap appears to close as the training budget increases.


2Note that we match parameters with MHA attention block, though our Transformer baseline and Griffin ended up
relying on MQA attention in order to improve inference efficiency. This means that our recurrent blocks have slightly more
parameters than the corresponding MQA blocks.


5


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


Table 1 | Character normalized accuracy. Hawk is competitive with our Transformer baseline, and
exceeds the reported performance of Mamba despite being trained on half as many tokens. Griffin
outperforms our Transformer baseline, and matches the performance of Llama-2 despite being trained
on roughly 7 times fewer tokens. We report unnormalized accuracy with partial scoring for WinoGrande.






|Model Model Training<br>Type Size Tokens|MMLU HellaSwag PIQA WinoGrande ARC-E ARC-C|Average|
|---|---|---|
|**Mamba**<br>3B<br>600B<br>7B<br>2T<br>**Llama-2**<br>13B<br>2T|26.2<br>71.0<br>78.1<br>65.9<br>68.2<br>41.7<br>45.3<br>77.2<br>78.8<br>69.2<br>75.2<br>45.9<br>**54.8**<br>80.7<br>80.5<br>72.8<br>77.3<br>49.4|58.5<br>65.3<br>69.3|
|**MQA**<br>1B<br>300B<br>**Transformer**<br>3B<br>300B<br>**(Baseline)**<br>6B<br>300B<br>1B<br>300B<br>3B<br>300B<br>**Hawk**<br>7B<br>300B<br>**Grifn**<br>1B<br>300B<br>3B<br>300B<br>7B<br>300B<br>14B<br>300B|28.9<br>64.8<br>75.0<br>62.0<br>60.2<br>35.4<br>31.7<br>71.0<br>77.6<br>66.1<br>68.1<br>39.2<br>38.9<br>77.0<br>79.5<br>70.4<br>74.1<br>45.2<br>29.7<br>63.3<br>76.1<br>57.2<br>60.6<br>34.6<br>31.3<br>71.7<br>78.8<br>66.5<br>68.4<br>40.2<br>35.0<br>77.6<br>80.0<br>69.9<br>74.4<br>45.9<br>29.5<br>67.2<br>77.4<br>65.2<br>67.0<br>36.9<br>32.6<br>73.5<br>78.1<br>67.2<br>71.5<br>41.4<br>39.3<br>78.6<br>81.0<br>72.6<br>75.4<br>47.9<br>49.5<br>**81.4**<br>**81.8**<br>**74.1**<br>**79.1**<br>**50.8**|54.4<br>59.0<br>64.2<br>53.6<br>59.5<br>63.8<br>57.2<br>60.7<br>65.8<br>**69.5**|



**3.2.** **Evaluation on downstream tasks**
In order to compare to other models in the literature, we train all our models for 300B tokens before
evaluating on downstream tasks. The two external baselines that we compare to are Mamba-3B (Gu and
Dao, 2023), the strongest small recurrent model reported in the literature to date, and Llama-2 (Touvron
et al., 2023), a widely used open Transformer model. Both external baselines have been trained on
significantly more than 300B tokens – Mamba has been trained on 600B tokens, twice more, and Llama-2
has been trained on 2T tokens, nearly seven times more. We note however that both Mamba and Llama-2
were trained on different datasets and with different hyper-parameter tuning strategies, which may
partially explain our strong performance. We therefore also include our own MQA transformer baseline,
trained on the same data and with the same hyper-parameter tuning budget as Hawk and Griffin.


We provide an evaluation on downstream tasks in Table 1. We find that both Hawk and Griffin achieve
very strong performance. In line with other works, we report character normalized accuracy on MMLU,
HellaSwag, PIQA, ARC-E and ARC-C, while we report absolute accuracy on WinoGrande with partial
scoring. The performance of Hawk improves significantly as we increase the model size, and Hawk-3B
achieves stronger performance on downstream tasks than Mamba-3B, despite being trained on half as
many tokens. Griffin-3B significantly outperforms Mamba-3B, and Griffin-7B and Griffin-14B achieve
performance competitive with Llama-2, despite being trained on nearly 7 times fewer tokens. Hawk
is also competitive with our MQA Transformer baseline, while Griffin outperforms this baseline.

### **4. Training Recurrent Models Efficiently on Device**


We encountered two main engineering challenges when developing and scaling our models. First, how
to efficiently shard our models across multiple devices. Second, how to efficiently implement linear
recurrences to maximize training efficiency on TPUs. We address both of these challenges in this section,
before providing an empirical comparison of the training speed of Griffin and our MQA baseline.


6


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


**4.1.** **Model parallelism for large scale training**
As our model increases in size, we cannot fit the model on a single device during training, even with a
batch size of 1 per-device. We therefore use model parallelism to shard our large models across devices
during training. Since communication costs across different training devices are expensive, efficiently
sharding the model is critical for fast training at scale.


**MLP and MQA block** For our gated-MLP block we use Megatron-style sharding (Shoeybi et al., 2019),
which requires a single all-reduce operation in both the forward and the backward pass. Similarly, we
apply the same strategy to the linear layers in the attention block, and additionally shard the attention
mechanism over its heads (Narayanan et al., 2021).


**Recurrent Block** The recurrent block contains two linear layers per branch. This allows us to apply
Megatron sharding to these layers in an equivalent fashion. The Conv1D layer operates independently
across channels, enabling us to split its parameters across devices without incurring any communication
overhead. To avoid additional cross-device communication, we use block-diagonal weights for the
gates in the RG-LRU (see equations 1 and 2), instead of dense matrices. For all experiments in this
paper, we use 16 blocks for both the recurrence gate and the input gate (such that _𝑊𝑥_ and _𝑊𝑎_ each
have _𝐷_ [2] _𝑅𝑁𝑁_ [/][16 parameters). The diagonal structure of the recurrence offers the same advantage as the]
Conv1D, allowing parameter sharding and computation without any communication. With this strategy,
the recurrent block’s communication requirements are equivalent to those of the MLP block.


**Other considerations** Optimizer states can consume significant memory, exceeding the size of the
model parameters themselves. To address this, we employ ZeRO parallelism (Rajbhandari et al., 2020),
distributing both optimizer states and model parameters across the batch shards. We also use bfloat16
representation for model parameters and activations, minimizing any data transfer overhead.


**4.2.** **Efficient linear recurrences on device**
Current deep learning accelerators are optimized for classical architectures which are composed largely
of matrix multiplications and convolutions. These operations have a high FLOPs-to-byte ratio, motivating
the development of specialized hardware units like Nvidia GPUs’ TensorCores (Markidis et al., 2018) and
Google TPUs’ MXUs (Norrie et al., 2021; Jouppi et al., 2021, 2023). Classical RNNs also benefit from this
due to their dense recurrence matrices. In contrast, our proposed RG-LRU layer, like other diagonal RNN
models, has a low FLOPs-to-byte ratio. This fundamental difference poses a computational challenge,
as existing accelerators lack optimization for such workloads. Since we run all our experiments on
TPU-v3, we focus on developing an efficient implementation tailored to this device [3] .


**Challenges for linear diagonal RNNs** One of the main challenges of utilizing a device like the TPU-v3
for the RG-LRU is that the update equation of the hidden state in eq. (4) is a pure elementwise operation.
For each element update it requires to load 6 bytes (assuming bfloat16 we need 2 bytes for each of the
variables _ℎ𝑡_ −1 _,𝑎𝑡,𝑥𝑡_ ) and write 2 bytes (the hidden state _ℎ𝑡_ ) while the computation only executes 6 FLOPs
(number of arithmetic operations in eq. 4) per element. This translates to a low FLOPs-to-byte ratio
of 0 _._ 75 – significantly below the device’s capacity for elementwise operations of 4 _._ 2 (see Appendix 3).
Execution time is therefore dominated by memory transfers between HBM and VMEM, making the
computation memory bound.


**A custom linear scan** [To address this we have written a custom Pallas kernel for the computation](http://go/jax-pallas)
of eq. (4) using a _linear scan_ . This allows us to minimize memory transfers, by keeping the hidden state


3The conclusions drawn here do not necessarily apply to other accelerators.


7


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models







1.13x

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||



2k 4k 8k
Sequence length


(c) 7B



|MQA Griffin|Col2|Col3|Col4|Col5|1.46x|Col7|
|---|---|---|---|---|---|---|
|1.00x<br>1.17x<br>0.93x<br>0.96x|1.00x<br>1.17x<br>0.93x<br>0.96x|1.00x<br>1.17x<br>0.93x<br>0.96x|1.00x<br>1.17x<br>0.93x<br>0.96x|1.00x<br>1.17x<br>0.93x<br>0.96x||0.98x|
|1.00x<br>1.17x<br>0.93x<br>0.96x|||||||
||||||||


2k 4k 8k
Sequence length


(a) 400m


|1.12x<br>1.00x0.97x 0.99x|Col2|Col3|Col4|Col5|Col6|1.01x|
|---|---|---|---|---|---|---|
|1.00x<br>1.12x<br>0.97x<br>0.99x|||||||
||||||||



2k 4k 8k
Sequence length


(b) 1B







Figure 3 | Training durations per step computed relative to our MQA baseline at 2K sequence length as we
vary the model size and sequence length for Griffin and MQA. Let us note that as we increase the sequence
length we lower the batch size proportionally, such that the total number of tokens per batch stays fixed.


in VMEM all the time, and also to perform memory transfers in larger chunks rather than one at a time.
In practice, this translates to almost 3x speed up over the native Jax implementation of the linear scan.
Additionally, we observe 10-20% lower training times per step of the full Hawk model, relative to the
same model using the native Jax implementation (see Appendix D.2 for more details.)


**Why we do not use convolutions or associative scans?** The initial appeal of linear recurrence models
stemmed from their high parallelizability, enabled by the associativity of their computations. This
permitted efficient execution on device via convolutions (Gu et al., 2021b) or prefix-sum algorithms (the
associative scan) (Smith et al., 2022). However, the RG-LRU’s gating mechanism on _𝑎𝑡_ is not compatible
with the convolutional view. Although we can still use the associative scan in principle, the associative
scan reduces the number of FLOPs required but does not reduce memory overheads, which is our primary
bottleneckinpractice. EmpiricallyweobservethatonaTPU-v3theassociativescanissignificantlyslower
that the native Jax linear scan (see Appendix D.2 for more details.) We speculate that the random access
nature of the tree recombination of the parallel prefix-sum algorithm makes is poorly suited for the TPU
architecture, leading to even slower memory transfers – the main bottleneck of this operation.


**4.3.** **Training speed on longer sequences**
We compare the training speeds across different model sizes and sequence lengths to investigate the
computational advantages of our models during training. For each model size, we keep the total number
of tokens per batch fixed, meaning that as we increase the sequence length, we proportionally decrease
the number of sequences. In Figure 3, we plot the relative runtimes of our Griffin model compared to that
of the MQA baseline at 2048 sequence length. At the lowest sequence length, the two models have similar
trainingtime, butasweincreasethesequencelengththeTransformerbecomesslower, whileGriffin’sruntime remains the same. The drop in speed for the baseline is more pronounced at smaller model sizes and
decreasesatlargermodelsizes. Thiscanbeexplainedbythefactthatallmodelscontainalargenumberof
linear layers. Their computation scales _𝑂_ ( _𝑇𝐷_ [2] ), while the RG-LRU is _𝑂_ ( _𝑇𝐷_ ) vs _𝑂_ ( _𝑇_ [2] _𝐷_ ) of global attention.
This means that as we increase the model width _𝐷_ compared to the sequence length _𝑇_, the linear layers
become the primary computational bottleneck, minimizing the efficiency gains from the RNN block.
Therefore, replacing Transformers with Hawk or Griffin offers the most significant wall-clock time improvement when sequence length is sufficiently large relative to model width to ensure the attention computationconstitutesamajorportionofthetotalcomputationtime. Wealsonotethatinpractice, ourMQA


8


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


baseline has slightly fewer parameters than Griffin at the same model scale (and performs fewer FLOPs).
This explains why Griffin trains slightly slower than our MQA baseline at 7B for short sequences.

### **5. Inference Speed**


Inference in LLMs is composed of two stages. In the “prefill” stage, we receive and process the prompt.
This step is effectively performing a forward pass of the model. Since the prompt can be processed in
parallel across the sequence, most model operations are compute bound during this stage. We therefore
expect the relative speeds of Transformers and recurrent models during the prefill stage to be similar
to the relative speeds of the same models during training, which we discussed in Section 4.


Prefillisfollowedbya“decode”stage, inwhichwesampletokensauto-regressivelyfromthemodel. Aswe
show below, recurrent modelshave lowerlatency andhigher throughput during thedecoding stage, especially for longer sequence lengths where the key-value (KV) cache used in attention can get large.


There are two main metrics to consider when evaluating inference speed. The first is latency, which
measures the time taken to generate a specified number of tokens at a certain batch size. The second is
throughput, which measures the largest number of tokens per second that can be generated on a single
device when sampling a specified number of tokens. Since throughput is given by tokens sampled times
batch size divided by latency, one can improve throughput either by reducing the latency or by reducing
memory usage to enable the use of larger batch sizes on device. Latency can be useful to consider for
real-time applications that require a quick response time. Throughput is also useful to consider as it can
tell us the maximum number of tokens we could sample from a particular model in a given time. This
property is useful when considering other language applications such as Reinforcement Learning from
Human Feedback (RLHF) or scoring language model outputs such as done in AlphaCode (Li et al., 2022)
where being able to output a large number of tokens in a given time is an appealing feature.


**5.1.** **A simple model of the decode step**
All components of language models are memory bound during decoding as long as batch size isn’t too
big (i.e. _𝐵_ ≲ 128- see Appendix F.1 for details) and we will assume this for the remainder of this section.
The largest memory overheads of Transformers typically come from the parameters themselves and the
KV cache. Therefore we can approximate the time required to generate a single token for each sequence
in the batch _𝐵_ during decoding as the time needed to load these two quantities from memory:


_Time to sample next token_ ≈ _[param size]_ [+] _[batch size]_ [×] _[cache size]_ _._ (5)

_memory bandwidth_


Here, _cache size_ refers to either the size of the KV cache at batch size 1 (for Transformers), or to the
size of the recurrent state at batch size 1 (for RNNs).


**Cache sizes** The difference in cache size relative to model parameters has important implications
for sampling efficiency. In recurrent and local attention blocks, parameter loading is the primary
bottleneck, (because the cache size is substantially smaller). In contrast, global attention’s KV cache
scales with the sequence length _𝑇_ and can be comparable to, or even exceed, the size of the model
parameters. This introduces considerable overhead when the sequence length _𝑇_ is large enough (as
shown in F.4). Consequently, an equally sized recurrent model can exhibit substantially lower latency
than a Transformer when _𝑇_ is large. Note however that as the model size grows the sequence length at
which we see latency benefits (where the KV cache size is comparable to parameter size) also increases.
It is important to note that, as well as improving latency, having a small recurrent state can also increase
the largest batch size that fits in memory on a single device, leading to higher throughput.


9


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models



140


120


100


80


60


40


20


0



|Col1|MQA<br>Griffin<br>Hawk|
|---|---|
|||
|||
|||
|||
|||
|||
|||


128 256 512 1024 2048 4096
# Tokens decoded


(a) Latency empty prefill



150


100


50


0



|Col1|Col2|MQA<br>Griffin<br>Hawk|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||


128 256 512 1024 2048 4096
# Tokens decoded


(b) Latency 4k prefill



Figure 4 | Latency of different 1B parameter models for a range of sequence lengths for (a) sampling
from an empty prefill and (b) sampling from a prefill of 4k tokens.


**5.2.** **Results**
Here, we look at inference results for models of size 1B parameters. For our baseline, we compare
against a MQA Transformer, which is significantly faster during inference than the standard MHA
Transformer often used in the literature. The models that we compare are: i) _MQA Transformer_, ii)
Hawk, and iii) Griffin. For comparing different models we report both latency and throughput.


**Latency** We compare the latency for models with a batch size of 16 with an empty prefill as well as
a prefill of 4096 tokens as seen in Figure 4. Hawk and Griffin achieve faster sampling latency than MQA
Transformers for long sequences. This is particularly noticeable as the sequence length and the prefill
length (which affect the size of the KV cache) are increased. Griffin achieves similar latency to Hawk,
demonstrating the excellent compatibility of linear recurrences and local attention.


**Throughput** We compare the maximum throughput (tokens/s) for the same models when sampling
512, 1024, 2048 and 4196 tokens following an empty prompt in Figure 1(b). We see that both Griffin
and Hawk achieve significantly higher throughput than the MQA Transformer baseline. This is partially
due to recurrent models having lower latency but also primarily occurs because Griffin and Hawk can
fit larger batch sizes than the MQA Transformer on a single device, since their cache size is smaller.
Hawk achieves higher throughputs than Griffin, since the size of the local attention cache eventually
becomes comparable to the size of the parameters when the batch size is large.

### **6. Long Context Modeling**


In this section, we explore the effectiveness of Hawk and Griffin to use longer contexts to improve their
next token prediction, and investigate their extrapolation capabilities during inference. Additionally,
we explore our models’ performance on tasks that require copying and retrieval capabilities, both for
models that are trained on such tasks, as well as when testing for these capabilities with our pre-trained
language models.


**6.1.** **Improving next token prediction with longer contexts**
We investigate the ability of Hawk and Griffin to improve their predictions with longer contexts. In
particular, we evaluate our trained models by measuring the loss on a held-out books dataset across
a range of sequence lengths. Using these long documents allows us to evaluate the ability of the models


10


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models



3.4


3.2


3.0


2.8


2.6


2.4



|Col1|Griffin<br>Hawk<br>MQA NoPE<br>MQA RoPE|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


128 256 512 1K 2K 4K 8K 16K 32K
Token position



3.4


3.2


3.0


2.8


2.6


2.4



|Col1|Griffin-2k<br>Griffin-8k<br>Hawk-2k<br>Hawk-8k|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||


128 256 512 1K 2K 4K 8K 16K 32K 65K131K
Token position



Figure 5 | Performance of various 1B parameter models on a held-out evaluation set of books. On the
left, the models have been trained with sequence length 2048, and on the right with sequence lengths
of respectively 2048 (2k) and 8192 (8k). Hawk and Griffin are able to extrapolate to significantly
longer sequences than the Transformer baselines, and further improve performance when trained
on longer sequences.


to extrapolate, i.e. the ability to accurately predict the next token given contexts that are longer than
those seen during training.


In Transformers, this ability to extrapolate is largely determined by the positional encoding used for the
attention layers (Kazemnejad et al., 2024). For recurrent models, it is instead dictated by the capacity
of the model to keep refining the representation stored in the recurrence state as the context becomes
longer. From the left plot of Figure 5, we observe that, up to some maximal length, both Hawk and
Griffin improve next token prediction given longer contexts, and they are overall able to extrapolate
to significantly longer sequences (at least 4x longer) than they were trained on. In particular, Griffin
extrapolates remarkably well even when using RoPE (Su et al., 2021) for the local attention layers.


The results so far evaluate models that have been trained on sequences of 2048 tokens. In order to assess
whether our models can also effectively learn from longer contexts, we train 1B parameter models on
sequences of 8192 (8k) tokens on MassiveText, and compare them to models trained on the same dataset
but on sequences of length 2048 (2k) tokens. We keep the total number of training tokens the same
across the models by reducing the batch size by a factor of 4 for the models trained on the sequence length
of 8192 (while keeping the number of training steps fixed). As illustrated in the right plot of Figure 5, we
find that Hawk-8k and Griffin-8k do achieve lower evaluation loss for sequences of length 8192 or larger,
compared to Hawk-2k and Griffin-2k. This indicates that Hawk and Griffin are able to learn to use longer
contexts during training. Interestingly, when evaluating at short sequence lengths, we find that Hawk-2k
and Griffin-2k perform slightly better than Hawk-8k and Griffin-8k. This suggests that the training sequence length should be carefully chosen according to the intended downstream use of the model.


**6.2.** **Copy and retrieval capabilities**
Recent work (Jelassi et al., 2024) has shown that Transformers can be significantly more efficient than
state space models (SSMs), a popular new family of RNNs, at learning synthetic tasks such as copying
the context or retrieving relevant tokens from the context. Additionally, Jelassi et al. (2024) showed
that pre-trained Transformers such as Pythia (Biderman et al., 2023) are much better at copying and
retrieval tasks at evaluation time compared to pre-trained SSM models such as Mamba (Gu and Dao,
2023). In this section, we investigate the efficiency of Griffin and Hawk in learning how to copy and
retrieve tokens from the context. Additionally, we evaluate pre-trained Hawk and Griffin models on
a phone number lookup task designed to test both copying and retrieval capabilities.


11


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models



1.00


0.75


0.50


0.25


0.00



|Col1|MQA<br>Hawk<br>Griffin|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||


0 25K 50K 75K 100K
Training Steps



|Col1|Col2|Col3|MQA<br>Hawk<br>Griffin<br>Train Length|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||


Eval Sequence Length



|Col1|Col2|ength|Col4|
|---|---|---|---|
||MQA<br>Hawk<br>Griffin<br>Train L|MQA<br>Hawk<br>Griffin<br>Train L|MQA<br>Hawk<br>Griffin<br>Train L|
|||||
|||||
|||||


0 1000 2000 3000
Eval Sequence Length



1.00


0.75


0.50


0.25


0.00



1.00


0.75


0.50


0.25


0.00



(a) Selective Copying Task



(b) Induction Heads Task



(c) Phonebook Lookup Task



Figure 6 | Exploring the copying and retrieval capabilities of Hawk and Griffin on three synthetic tasks.
Figures (a) and (b) show the performance of 5 layer deep models on a held out eval set when explicitly
trained on these tasks. Figure (c) shows the performance on a phone number lookup task when
evaluating our pre-trained 7B Hawk and Griffin models against our 6B MQA Transformer baseline.


**Training on synthetic tasks** To investigate the efficiency of learning how to copy and retrieve relevant
tokens from the context, we train on two synthetic tasks: Selective Copying and Induction Heads. To be
able to compare Transformers with Hawk and Griffin, we consider 5-block deep networks with model
dimension 64, totalling roughly 250K parameters, where Griffin uses a single local attention in the
middle of the network, in the third block.


 - **Selective copying task** : In this task, the model needs to learn to copy data tokens from a sequence
while ignoring noise tokens from the context. See Appendix H for more details on the setup for
this task. This task is inspired by Gu and Dao (2023), where the authors showed that Mamba
was able to solve this task better than previously proposed SSMs. We use a vocabulary size of
16, and train on sequences of length 1024, containing 16 data tokens (randomly sampled from
the vocabulary and at random locations), with the rest of the tokens set to the noise token. Griffin
uses a local attention window size of 512.


 - **Induction heads** : In this task, the model needs to learn to recall the token immediately following
a special token. This requires the model to learn the special token, and retrieve the token immediately following it in the context. If the model is able to learn the task, it should be able to extrapolate
to significantly longer sequences than it was trained for. We use a vocabulary size of 16 and train
on sequences of length 256 where the tokens are sampled randomly, and we randomly sample
the location of the special token in the sequence. Griffin uses a local attention window of size 128.


We show our results in Figure 6. On the Selective Copying task, we find that all 3 models are able to
solve the task perfectly. When comparing speed of learning on this task, we find Hawk to be significantly
slower than Transformers, similar to the observation made by Jelassi et al. (2024), where the authors
showed that Mamba was significantly slower to learn on similar tasks. Interestingly though, Griffin
shows almost no slowdown, effectively matching the speed of learning of Transformers, despite using
only a single local attention layer.


On the Induction Heads task, while all 3 models can solve the task perfectly up to the training sequence
length, our Transformer baseline is not able to extrapolate to longer sequences during evaluation. While
our MQA baseline uses RoPE, Gu and Dao (2023) had similar observation for Transformers with a range
of positional encodings. We find that Hawk is able to perfectly extrapolate on this task to evaluation
sequences several orders of magnitude longer than the training sequence length. Notably, Griffin, with
its local attention, also demonstrated exceptional ability to extrapolate on this task.


12


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


**Evaluating pre-trained models** We now evaluate whether copying and retrieval capabilities naturally
emerge in our pre-trained models. We consider our 7B Hawk and Griffin models and our 6B MQA Transformerbaseline, alltrainedon300BtokensontheMassiveTextdataset. Weconsiderthesamephonebook
lookup task introduced in Jelassi et al. (2024), where we provide to the model a synthetic phonebook containing names and numbers, and the model is asked to retrieve the correct phone number given a name.
The prompt to the model is a phonebook consisting of randomly sampled list of names and numbers of a
certain length, followed by two randomly sampled examples of the task, followed by a randomly sampled
name from the phonebook for which the model needs to retrieve the correct phone number.


From Figure 6(c), we see that while Hawk can do reasonably well on the task for very short phonebook
lengths, it fails to memorize and retrieve the correct phone number when the phonebook length grows,
similar to the observation made by Jelassi et al. (2024) on the Mamba model’s performance on this task.
This is not particularly surprising since Hawk uses a small fixed-size state. Our Transformer baseline can
almost perfectly solve this task up to the training sequence length, but fails to retrieve the correct phone
number for context lengths longer than the training sequence length. Interestingly, Griffin can perfectly
solve this task up to a context length that matches its local attention window size of 1024, in spite of
using only a single local attention layer. Once the context length is long enough such that the local
attention window does not cover the whole phonebook, performance starts to degrade. Griffin is also
able to extrapolate better to longer sequence lengths compared to Transformers. While the performance
of Griffin is promising for the ability of models with fixed-size state to solve copying and retrieval tasks,
our results suggest more work is needed to improve these capabilities for such models.

### **7. Related Works**


The Transformer architecture has become a more scalable alternative to RNNs. Transformers achieve
superior scalability through fully parallelized training, contrasting with the inherent limitations of
RNNs. Due to their sequential processing structure, classical RNNs suffer from slow training speeds
during both forward and backward propagation (Werbos, 1990). To mitigate this issue, researchers
have explored alternative RNN-based methods. Notable examples include Quasi-RNNs (Bradbury
et al., 2016), which combine convolutions and linear RNNs for greater parallelization, and the use of
input-based gating mechanisms to parallelize linear RNN training (Martin and Cundy, 2017).


State-space Models (SSMs) have recently emerged as a powerful tool for modeling long input sequences.
They demonstrated strong performance on tasks from the long-range arena benchmark (Tay et al., 2020),
and audio generation (Goel et al., 2022). SSMs successfully integrate concepts from classical state-space
models (Kalman, 1960) with those of RNNs. Their reliance on linear recurrences allows for efficient
hidden state computation, either through parallel scan operations or convolutions, resulting in training
speeds comparable to Transformer models. The S4 (Gu et al., 2021a) model proposed a sophisticated
parameterization called **normal plus low-rank** to diagonalize the recurrence computation. The S4D
parametrized the SSM directly with a diagonal state matrix and showed that it performed just as well
while being much simpler (Gu et al., 2022). S5 also diagonalized the recurrence, and showed that the
recurrence can be computed using the associative scan (Smith et al., 2022). The H3 model (Dao et al.,
2022b) generalizes the recurrent interpretation of linear attention (Katharopoulos et al., 2020). Hyena
(Poli et al., 2023) uses a similar architecture, but replaces the S4D layer with a global convolution kernel
parametrized by an MLP. RetNet (Sun et al., 2023) uses a simpler SSM design with a gating mechanism
which allows them to parallelize the computation using a variant of multi-head attention. Orvieto et al.
(2023b) systematically analyzed and ablated multiple modifications to standard RNNs. Their finding
showed that through better parameterization and initialization simplified linear RNNs (the LRU), perform just as well as other SSMs variants on various long-range tasks. RWKV (Peng et al., 2023) is a recent
RNN, shown to be competitive on language modeling tasks, based on another linear attention approximation inspired by the attention-free Transformer (Zhai et al., 2021). Concurrent to our work Gu and Dao


13


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


(2023) developed an SSM architecture called Mamba with an input dependant selection mechanism
and showed that it achieves performance comparable to Transformers with efficient inference. Several
extensions of Mamba have been proposed (Wang et al., 2024; Zhu et al., 2024) for different applications.
An input-dependent gating similar to Mamba was also proposed by Gateloop (Katsch, 2023).


Linear attention (Katharopoulos et al., 2020) offers a computationally efficient approximation of the
self-attention mechanism by linearizing the attention, which can be computed recurrently as a linear
RNN. While this approach significantly reduces computational cost compared to full attention, it often
comes with a trade-off in model performance. Flash Attention (Dao et al., 2022a) improves the training
speed of attention on GPUs by making efficient use of the memory hierarchy. Another approach to
reducing the computational cost of global attention, which is becoming increasingly more popular, is
using sparse-local attention (Child et al., 2019) or sliding window attention (Jiang et al., 2023).

### **8. Conclusion**


This work introduces Hawk; a recurrent model incorporating a novel gated linear recurrent layer, the
RG-LRU. We also introduce Griffin; a hybrid model which mixes the RG-LRU layer with local attention.
These models demonstrate exceptional language modeling performance across varying scales, with
held-out loss exhibiting power-law scaling as compute resources increase. Hawk exceeds the reported
performance of Mamba on downstream tasks when trained on half as many tokens, while Griffin slightly
exceeds the performance of Llama-2 when trained on over 6 times fewer tokens. Furthermore, we
empirically validate the inference-time advantages of Hawk and Griffin and observe reduced latency
and significantly increased throughput compared to our Transformer baselines. Lastly, Hawk and Griffin
exhibit the ability to extrapolate on longer sequences than they have been trained on and are capable of
efficiently learning to copy and retrieve data over long horizons. These findings strongly suggest that our
proposed models offer a powerful and efficient alternative to Transformers with global attention.

### **Acknowledgements**


We thank Adam Paszke, Sharad Vikram, Trevor Gale, Sebastian Borgeaud, George Scrivener, Raia
Hadsell, Oriol Vinyals, Toby Boyd, Zhifeng Chen, Chris Dyer, Kelvin Xu, Andriy Mnih for their guidance
and advice. We make use of the DeepMind Jax ecosystem (Bradbury et al., 2018) and especially thank
Andy Brock for building the internal framework we used for training and evaluating our models.

### **References**


J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,
S. Altman, S. Anadkat, et al. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023.


D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and
translate. _arXiv preprint arXiv:1409.0473_, 2014.


I. Beltagy, M. E. Peters, and A. Cohan. Longformer: The long-document transformer. _arXiv preprint_
_arXiv:2004.05150_, 2020.


S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit,
U. S. Prashanth, E. Raff, et al. Pythia: A suite for analyzing large language models across training
and scaling. In _International Conference on Machine Learning_, pages 2397–2430. PMLR, 2023.


J. Bradbury, S. Merity, C. Xiong, and R. Socher. Quasi-recurrent neural networks. _arXiv_ _preprint_
_arXiv:1611.01576_, 2016.


14


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy
programs, 2018. URL `[http://github.com/google/jax](http://github.com/google/jax)` .


T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry,
A. Askell, et al. Language models are few-shot learners. In _Advances in Neural Information Processing_
_Systems_, volume 33, pages 1877–1901, 2020.


R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers.
_arXiv preprint arXiv:1904.10509_, 2019.


J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks
on sequence modeling. _arXiv preprint arXiv:1412.3555_, 2014.


T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact
attention with io-awareness. In _Advances in Neural Information Processing Systems_, volume 35, pages
16344–16359, 2022a.


T. Dao, D. Y. Fu, K. K. Saab, A. W. Thomas, A. Rudra, and C. Ré. Hungry hungry hippos: Towards
language modeling with state space models. _arXiv preprint arXiv:2212.14052_, 2022b.


Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier. Language modeling with gated convolutional networks.
In _International Conference on Machine Learning_, pages 933–941. PMLR, 2017.


J. L. Elman. Finding structure in time. _Cognitive Science_, 14(2):179–211, 1990.


Gemini Team Google. Gemini: a family of highly capable multimodal models. _arXiv_ _preprint_
_arXiv:2312.11805_, 2023.


K. Goel, A. Gu, C. Donahue, and C. Ré. It’s raw! audio generation with state-space models. In
_International Conference on Machine Learning_, pages 7616–7633, 2022.


A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint_
_arXiv:2312.00752_, 2023.


A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Ré. Hippo: Recurrent memory with optimal polynomial
projections. In _Advances in Neural Information Processing Systems_, volume 33, pages 1474–1487, 2020.


A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with structured state spaces. _arXiv_
_preprint arXiv:2111.00396_, 2021a.


A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, and C. Ré. Combining recurrent, convolutional,
and continuous-time models with linear state space layers. In _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, volume 34, pages 572–585, 2021b.


A. Gu, A. Gupta, K. Goel, and C. Ré. On the parameterization and initialization of diagonal state space
models. _arXiv preprint arXiv:2206.11893_, 2022.


D. Hendrycks and K. Gimpel. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_, 2016.


S. Hochreiter and J. Schmidhuber. Long short-term memory. _Neural Computation_, 9(8):1735–1780,
1997.


J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A.
Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. _arXiv preprint_
_arXiv:2203.15556_, 2022.


15


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


S. Jelassi, D. Brandfonbrener, S. M. Kakade, and E. Malach. Repeat after me: Transformers are better
than state space models at copying. _arXiv preprint arXiv:2402.01032_, 2024.


A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel,
G. Lample, L. Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_, 2023.


N. Jouppi, G. Kurian, S. Li, P. Ma, R. Nagarajan, L. Nai, N. Patil, S. Subramanian, A. Swing, B. Towles,
et al. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware
support for embeddings. In _Proceedings of the 50th Annual International Symposium on Computer_
_Architecture_, pages 1–14, 2023.


N. P. Jouppi, D. H. Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. Ma, X. Ma,
etal. Tenlessonsfromthreegenerationsshapedgoogle’stpuv4i: Industrialproduct. In _2021ACM/IEEE_
_48th Annual International Symposium on Computer Architecture (ISCA)_, pages 1–14. IEEE, 2021.


R. E. Kalman. A new approach to linear filtering and prediction problems. _Journal of Basic Engineering_,
82, 1960.


J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and
D. Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_, 2020.


A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: Fast autoregressive
transformers with linear attention. In _International_ _Conference_ _on_ _Machine_ _Learning_, pages
5156–5165. PMLR, 2020.


T. Katsch. Gateloop: Fully data-controlled linear recurrence for sequence modeling. _arXiv preprint_
_arXiv:2311.01927_, 2023.


A. Kazemnejad, I. Padhi, K. Natesan Ramamurthy, P. Das, and S. Reddy. The impact of positional encoding
on length generalization in transformers. _Advances in Neural Information Processing Systems_, 36, 2024.


Y. LeCun, L. Bottou, G. B. Orr, and K.-R. Müller. Efficient backprop. In _Neural Networks: Tricks of the_
_Trade_, pages 9–50. Springer, 2002.


Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno,
A. Dal Lago, et al. Competition-level code generation with alphacode. _Science_, 378(6624):
1092–1097, 2022.


I. Loshchilov and F. Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_,
2017.


S. Markidis, S. W. Der Chien, E. Laure, I. B. Peng, and J. S. Vetter. Nvidia tensor core programmability,
performance & precision. In _2018 IEEE international parallel and distributed processing symposium_
_workshops (IPDPSW)_, pages 522–531. IEEE, 2018.


E. Martin and C. Cundy. Parallelizing linear recurrent neural nets over sequence length. _arXiv preprint_
_arXiv:1709.04057_, 2017.


H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur. Long range language modeling via gated state
spaces. _arXiv preprint arXiv:2206.13947_, 2022.


T. Mikolov, M. Karafiát, L. Burget, J. Cernocký, and S. Khudanpur. Recurrent neural network based
language model. In _INTERSPEECH 11th Annual Conference of the International Speech Communication_
_Association_, pages 1045–1048, 2010.


16


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand,
P. Kashinkunti, J. Bernauer, B. Catanzaro, et al. Efficient large-scale language model training on
gpu clusters using megatron-lm. In _Proceedings of the International Conference for High Performance_
_Computing, Networking, Storage and Analysis_, pages 1–15, 2021.


T. Norrie, N. Patil, D. H. Yoon, G. Kurian, S. Li, J. Laudon, C. Young, N. Jouppi, and D. Patterson. The
design process for Google’s training chips: TPUv2 and TPUv3. _IEEE Micro_, 41(2):56–63, 2021.


A. Orvieto, S. De, C. Gulcehre, R. Pascanu, and S. L. Smith. On the universality of linear recurrences
followed by nonlinear projections. _arXiv preprint arXiv:2307.11888_, 2023a.


A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De. Resurrecting recurrent
neural networks for long sequences. _arXiv preprint arXiv:2303.06349_, 2023b.


B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, H. Cao, X. Cheng, M. Chung, M. Grella, K. K.
GV, et al. Rwkv: Reinventing RNNs for the transformer era. _arXiv preprint arXiv:2305.13048_, 2023.


M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, and C. Ré. Hyena
hierarchy: Towards larger convolutional language models. _arXiv preprint arXiv:2302.10866_, 2023.


J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring,
S. Young, et al. Scaling language models: Methods, analysis & insights from training Gopher. _arXiv_
_preprint arXiv:2112.11446_, 2021.


S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion
parameter models. In _SC20: International Conference for High Performance Computing, Networking,_
_Storage and Analysis_, pages 1–16. IEEE, 2020.


N. Shazeer. Fast transformer decoding: One write-head is all you need. _arXiv preprint arXiv:1911.02150_,
2019.


N. Shazeer. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_, 2020.


M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-lm: Training multibillion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_, 2019.


H. T. Siegelmann and E. D. Sontag. Turing computability with neural nets. _Applied Mathematics Letters_,
4(6):77–80, 1991. ISSN 0893-9659.


J. T. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling.
_arXiv preprint arXiv:2208.04933_, 2022.


J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. Roformer: Enhanced transformer with rotary
position embedding. _arXiv preprint arXiv:2104.09864_, 2021.


Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor
to transformer for large language models. _arXiv preprint arXiv:2307.08621_, 2023.


I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence learning with neural networks. In _Advances_
_in Neural Information Processing Systems_, pages 3104–3112, 2014.


Y. Tay, M. Dehghani, S. Abnar, Y. Shen, D. Bahri, P. Pham, J. Rao, L. Yang, S. Ruder, and D. Metzler.
Long range arena: A benchmark for efficient transformers. _arXiv preprint arXiv:2011.04006_, 2020.


17


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal,
E. Hambro, F. Azhar, et al. LLama: Open and efficient foundation language models. _arXiv preprint_
_arXiv:2302.13971_, 2023.


A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin.
Attention is all you need. In _Advances in Neural Information Processing Systems_, volume 30, 2017.


J. Wang, T. Gangavarapu, J. N. Yan, and A. M. Rush. Mambabyte: Token-free selective state space
model. _arXiv preprint arXiv:2401.13660_, 2024.


P. J. Werbos. Backpropagation through time: what it does and how to do it. _Proceedings of the IEEE_,
78(10):1550–1560, 1990.


Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey,
et al. Google’s neural machine translation system: Bridging the gap between human and machine
translation. _arXiv preprint arXiv:1609.08144_, 2016.


R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T. Liu. On layer
normalization in the transformer architecture. In _International Conference on Machine Learning_,
pages 10524–10533. PMLR, 2020.


S. Zhai, W. Talbott, N. Srivastava, C. Huang, H. Goh, R. Zhang, and J. Susskind. An attention free
transformer. _arXiv preprint arXiv:2105.14103_, 2021.


B. Zhang and R. Sennrich. Root mean square layer normalization. _Advances in Neural Information_
_Processing Systems_, 32, 2019.


L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang. Vision mamba: Efficient visual representation
learning with bidirectional state space model. _arXiv preprint arXiv:2401.09417_, 2024.


18


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models

### **A. RG-LRU Recurrence Gate**


In Figure 7, we demonstrate the behavior of different gating mechanisms applied on the recurrent
weight _𝑎_ .



a=0.5



a=0.9



a=0.99



1.0


0.8


0.6


0.4


0.2


0.0





1.0


0.8


0.6


0.4


0.2


0.0





1.0


0.8


0.6


0.4


0.2


0.0





0.0 0.2 0.4 0.6 0.8 1.0
rt



0.0 0.2 0.4 0.6 0.8 1.0
rt



0.0 0.2 0.4 0.6 0.8 1.0
rt



Figure 7 | The behaviour of different gating mechanisms applied on the recurrent weight _𝑎_ (note that
in the Mamba’s notations this is − _𝐴_ ).


**Implementation** We implement our recurrence gate, as defined in Section 2.4, in a slightly different,
but mathematically equivalent form, for numerical stability. In particular, we compute the logarithm
of _𝑎𝑡_ and then we exponentiate it, instead of computing a sigmoid and then taking a power:


log _𝑎𝑡_ = log _𝑎_ _[𝑐𝑟][𝑡]_ = log _𝜎_ (Λ) _[𝑐𝑟][𝑡]_ =             - _𝑐_ softplus(Λ)⊙ _𝑟𝑡._ (6)


**Gate behaviour** Our gate is quite different than other standard gates in the literature. In particular,
most gating mechanisms, like the one used in Mamba and GRU, allow through the gate to interpolate
fully between the hidden state and the new observation. Ours on the other hand is biased towards
retaining information, and does not allow to fully discard the contribution of _ℎ𝑡_ −1 (this depends, however,
on the value of Λ). To demonstrate this, we analyze the relative weight of _𝑥𝑡_ compare to _ℎ𝑡_ −1 in the
output _𝑦𝑡_ . For a general recurrence we define this as:


_ℎ𝑡_ = _𝛼_ ( _𝑟𝑡_ ) _ℎ𝑡_ −1 + _𝛽_ ( _𝑟𝑡_ ) _𝑥𝑡._ (7)


~~√~~
For our model we have _𝛼_ ( _𝑟𝑡_ ) = _𝑎𝑡_ = _𝑎_ _[𝑐𝑟][𝑡]_ and _𝛽_ ( _𝑟𝑡_ ) = 1− _𝛼_ ( _𝑟𝑡_ ) [2] . For a standard GRU style gating we have

_𝛼_ ( _𝑟𝑡_ ) = 1 - _𝑟𝑡_ and _𝛽_ ( _𝑟𝑡_ ) = _𝑟𝑡_ . For Mamba, assuming in their notation _𝐵_ = 1 _,𝐶_ = 1, then _𝛼_ ( _𝑟𝑡_ ) = (1 - _𝑟𝑡_ ) [−] _[𝐴]_

and _𝛽_ ( _𝑟𝑡_ ) = (1− _𝛼_ )/ _𝐴_ . The behaviour of the different gating mechanisms is depicted in Figure 7, where
for clarity we have also included the update value of the LRU (Orvieto et al., 2023b), which has no
gating. As can be seen, the Mamba gating is almost identical to the GRU for values of _𝐴_ close to 1, with
minor deviations at smaller values. On the other hand, our gating mechanism performs a very different
non-linear interpolation between fully discarding the input _𝑥𝑡_ and the update of the LRU.

### **B. Complex-Gated Linear Recurrent Unit (CG-LRU)**


In Section 2.4 we have defined our recurrent layer, however it can be further extended to use complex
numbers. To achieve this we first parameterize a complex diagonal recurrence via ˜ _𝑎_ = _𝜎_ (Λ) _𝑒_ _[𝑖𝜃]_, where
_𝜃_ is a learnable parameter. In addition, we split the input _𝑥𝑡_ along its channel dimensions, and interpret


19


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


its first half as the real part of a complex vector, and the second part as the imaginary part of the same
complex vector:



_𝑥𝑡_ = - _𝑥𝑡_ [1]
_𝑥𝑡_ [2]




(8)



_𝑥_ ˜ _𝑡_ = _𝑥𝑡_ [1] [+] _[𝑖𝑥]_ _𝑡_ [2] _[.]_ (9)


With this we rewrite the equations for the LRU (see eq. 4) as:


_𝑟𝑡_ = _𝜎_ ( _𝑊𝑎𝑥𝑡_ + _𝑏𝑎_ ) _,_ _recurrence gate_ (10)

_𝑖𝑡_ = _𝜎_ ( _𝑊𝑥_ _𝑥𝑡_ + _𝑏𝑥_ ) _,_ _input gate_ (11)

_𝑎_ ˜ _𝑡_ = _𝑎_ ˜ _[𝑐𝑟][𝑡]_ _,_ (12)



√︃
_ℎ_ ˜ _𝑡_ = _𝑎_ ˜ _𝑡_ ⊙ _ℎ_ [˜] _𝑡_ −1 +



1−| _𝑎_ ˜ _𝑡_ | [2] ⊙( _𝑖𝑡_ ⊙ _𝑥_ ˜ _𝑡_ ) _._ (13)



We mark all complex variables with ˜· for clarity. Note that the number of dimensions of _𝑟𝑡,𝑖𝑡,𝑎_ ˜ _𝑡_ and _ℎ_ [˜] _𝑡_
are half of those of the real input _𝑥𝑡_ . Finally, to compute the output we stack the real and imaginary
part of _ℎ𝑡_ into a single vector _𝑦𝑡_ :




                           - Real( _ℎ_ [˜] _𝑡_ )
_𝑦𝑡_ =
Imaginary( _ℎ_ [˜] _𝑡_ )

### **C. Model Scale Hyper-Parameters**




(14)



In Table 2, we present the hyper-parameters of the models at different scales. These hyperparameters
are shared for all the model families that we explored in this paper.


Table 2 | Key model hyper-parameters considered for different model sizes. These hyperparameters
are shared across different architectures we tested.


**Model** **Model Width** **RNN Width** **Depth** **MLP Expansion** **Attention** **Training Tokens**
**Size** **(** _𝑫_ **)** **(** _𝑫𝑹𝑵𝑵_ **)** ( _𝑵_ ) **Factor (** _𝑴_ **)** **Heads** **(Optimal Scaling)**


100M 768 1024 12 3 6 1.9B
200M 1024 1536 12 3 8 3.9B
400M 1536 2048 12 3 12 7.8B
1.3B 2048 2560 24 3 16 25B
3B 3072 4096 24 3 24 60B
7B 4096 5632 32 3 32 132.5B
14B 5120 8192 40 3 40 300B

### **D. Efficient Linear Recurrences on Device**


The initial step in computational optimization lies in identifying the primary performance bottleneck
on the target hardware. For most accelerators, the key limiting factors are computational throughput
(FLOPs/s) and memory bandwidth between the high-bandwidth memory (HBM) and the fast vector
memory (VMEM). While factors like HBM capacity and host-device communication are relevant, techniques such as ZeRO sharding and pipelined data transfer offer practical mitigations. Modern accelerator
designs often prioritize a high FLOPs-to-byte ratio to accommodate workloads where computations
significantly outnumber memory transfers. We show the key specification of the TPU-v3 pod (two chips
per pod) in Table 3, which we use for all our experiments.


20


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


**Specification** **TPU-v3 Pod**


HBM capacity 32 GB
HBM bandwidth 900 GB/s
Peak MXU compute 123 TFLOPs/s (bfloat16)
Peak MXU FLOPs-to-byte-ratio 136
Peak VPU compute 3.8 TFLOPs/s
Peak VPU FLOPs-to-byte-ratio 4.2


Table 3 | Hardware specifications for a TPU-v3 pod.



60


50


40


30


20


10


0



|Linear(Pallas)<br>1.58x<br>Linear(Jax)<br>1.41x<br>Associative(bf16)<br>1.27x<br>Associative(f32)<br>1.00x 1.00x 1.00x<br>0.85x 0.81x 0.89x|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||


Sequence length


(a) Scan runtimes


|Col1|Col2|Col3|
|---|---|---|
||||
||||


|Col1|Col2|
|---|---|
|||


|Col1|Col2|Col3|
|---|---|---|
||||
||||



400m 1b 7b
Model size


(b) Hawk runtimes











Figure 8 | a) Runtimes of different implementations of the scan operation on a TPU-v3 at different
sequence lengths. The batch size of the input is fixed at 8 and the dimension of each token is 1024.
b) Relative runtimes of the Hawk model when using different implementations of the scan operation,
in reference to the one with the native Jax scan implementation.


**D.1.** **Matrix multiplication computation**
A typical matrix multiplication of a _𝐷_ × _𝐷_ matrix with a _𝐷_ × _𝑁_ matrix has 2 _𝑁𝐷_ [2] FLOPs and 2( _𝐷_ [2] +2 _𝑁𝐷_ )
bytes to transfer (both read and write) which translates to _𝐷𝑁𝐷_ + _𝑁_ [FLOPs/byte ratio. When] _[𝐷>> 𝑁]_ [and]
running on a TPU-v3 this implies that the dimension _𝑁_ must be at least 136 to saturate the device, in
which case the operation is “compute bound”, or otherwise most of the time will be spent on waiting
for memory transfers, in which case the operation is “memory bound”.


**D.2.** **Scan runtimes**
In Figure 8(a) we demonstrate that on a TPU-v3 our Pallas kernel achieves nearly x3 speed up compared
to the naive Jax implementation. In addition, the associative scan is significantly slower, even if fully run
in bfloat16 precision. Figure 8(b) demonstrates that these gains also translate to significant improvements of the overall training time per step of the full Hawk model even at the 7b scale. For completeness
we have also added the runtime of the associative scan, which can be up to 50% slower.

### **E. The Local Attention Window Size of Griffin**


Griffin uses both recurrent blocks as well as local attention layers in its temporal mixing blocks. For
all experiments previously shown using a training sequence length of 2048, we use a local attention
window size of 1024. We now investigate how the performance of different window sizes for the local
attention layer varies with the training sequence length.


We consider 400M parameter models trained on sequence lengths of 2048, 4096 and 8192 tokens,


21


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models



2.66

2.64

2.62

2.60

2.58

2.56

2.54

2.52



























2.50

### **F. Inference Speeds**


models.


22


|Col1|MQA Transformer<br>512 Griffin<br>1K<br>2K<br>1K<br>4K<br>2K 2K 8K1K<br>4K<br>1K<br>1K<br>2K<br>2K<br>4K|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|Tr<br>   0M<br>    ife<br>    h b<br>    tio<br>    len<br>      th<br>   bas<br>     mb<br>    att<br>    ell<br>     in<br>    wi<br> <br>    re<br>    t ou<br>   we<br>    an<br>    uen<br>    In p<br>    s o<br>    wi<br>    tio<br>** ou**<br>    ag<br>   N<br>   e li<br>**  ry**<br>   me<br>       to<br>     mor|Tr<br>   0M<br>    ife<br>    h b<br>    tio<br>    len<br>      th<br>   bas<br>     mb<br>    att<br>    ell<br>     in<br>    wi<br> <br>    re<br>    t ou<br>   we<br>    an<br>    uen<br>    In p<br>    s o<br>    wi<br>    tio<br>** ou**<br>    ag<br>   N<br>   e li<br>**  ry**<br>   me<br>       to<br>     mor|ain<br>    p<br>    re<br>     ar<br>    n<br>    gt<br>      e<br>   el<br>     er<br>    en<br>     M<br>      Fi<br>    nd<br>    ma<br>     tp<br>   ve<br>    d t<br>    ce<br>     ra<br>     f t<br>    nd<br>    n<br>** nd**<br>    e<br>    is<br>    ne<br>**   bo**<br>   nsi<br>        be<br>     y|Le<br>    ar<br>    nt<br>      in<br>     var<br>    h)<br>       plo<br>   in<br>      of<br>    ti<br>     QA<br>      gu<br>    ow<br>    rk<br>     er<br>   r,<br>     he<br>     le<br>     ct<br>      ra<br>    ow<br>     M<br>** ed**<br>     mo<br>     m<br>    ar<br>**   un**<br>   on<br>         co<br>      bo|ng<br>    am<br>     tr<br>       th<br>     ia<br>    . F<br>       t)<br>   es<br>       tr<br>    on<br>      T<br>      re<br>     s<br>    ab<br>     fo<br>    it i<br>      gl<br>     n<br>     ice<br>      ini<br>     si<br>     QA<br>** n**<br>     de<br>     em<br>     lay<br>**   d**<br>    (<br>         m<br>      un|th<br>    et<br>     ai<br>       e<br>     nt<br>ur<br>        fo<br>    ac<br>       ain<br>     w<br>      ra<br>  9 <br>     iz<br>    ly,<br>     rm<br>     s<br>      ob<br>     gt<br>     , t<br>      ng<br>     ze<br>      T<br>** es**<br>     ls<br>     o<br>     er<br>**   ed**<br>    us<br>         pu<br>      d|2K<br>T<br>    er Gri<br>     ning s<br>        plot.  <br>     s of th<br>therm<br>        r the<br>    ross a<br>       ing to<br>     indow<br>      nsfor<br>, wher<br>     e equa<br>     even<br>     s the<br>      worth<br>      al att<br>     h grow<br>      he har<br>       and i<br>     s less<br>      ransfo<br>** s**<br>      at dec<br>     ry bou<br>     s and<br>**   ness o**<br>    ually c<br>         te bou<br>       at dec|ra<br>     f<br>      eq<br>W<br>       e<br>o<br>          Gri<br>     ll t<br>        k<br>      si<br>      me<br> e<br>      l t<br>      w<br>       gl<br>       no<br>       en<br>      s<br>       d<br>        nf<br>       th<br>      rm<br>       o<br>      n<br>       se<br>**    f l**<br>     on<br>          n<br>        od|in L<br>     n<br>      ue<br>e n<br>        M<br>re,<br>          f<br>      ra<br>        en<br>      ze<br>      rs<br>  th<br>       o t<br>      he<br>       ob<br>       ti<br>       tio<br>       fu<br>       wa<br>        er<br>       an<br>      e<br>       de<br>      d. <br>       lf-<br>**     in**<br>     si<br>          d.<br>        e|en<br>      an<br>      nc<br> ot<br>        QA<br> w<br>          n<br>      in<br>        s f<br>      s.<br>       us<br>  e<br>        he<br>      n<br>       al<br>       ng<br>       n<br>       rth<br>       re<br>        en<br>        th<br>      rs,<br>        ti<br> In<br>       att<br>**     ea**<br>     sti<br>           At<br>         tim|gth<br>      d<br>      e l<br> ic<br>         T<br> e<br>           mo<br>      ing<br>         xe<br>       As<br>       in<br>   win<br>         tr<br>       us<br>        at<br>        th<br>        M<br>       er<br>        us<br>        ce<br>        e t<br>       as<br>        me<br> t<br>       en<br>**     r l**<br>     ng<br>            d<br>         e.|4<br>       MQ<br>       en<br> e t<br>         ra<br>  se<br>           d<br>       s<br>         d.<br>        b<br>       g l<br>   d<br>         ain<br>       in<br>        te<br>        at<br>        QA<br>       , i<br>        ed<br>         sp<br>         ra<br>        w<br>         is<br> he<br>       tio<br>**      ay**<br>      of<br>            ec<br>|K<br>       A Tra<br>       gths.<br>  hat a<br>         nsfor<br>  e that<br>           el out<br>       equen<br>          For e<br>        aselin<br>        ocal a<br>   ow si<br>         ing s<br>       g a fx<br>        ntion<br>         the p<br>         Tran<br>        t is lik<br>         will a<br>         eed. F<br>         ining<br>        ell as<br>          boun<br>  follo<br>       n) in<br>**      ers**<br>       batc<br>            ode ti|Tr<br>        ns<br>        Th<br>    gl<br>         me<br>    us<br>            pe<br>       ce<br>           ac<br>        es<br>         tte<br>    zes<br>          eq<br>         ed<br>         M<br>          er<br>         sfo<br>          el<br>          lso<br>          in<br>          se<br>          Gr<br>          de<br>  wi<br>         ou<br>       h_ 𝐵_<br>             me|ain<br>        fo<br>        e<br>    ob<br>         r (<br>    in<br>            rfo<br>        le<br>           h<br>        , w<br>         nt<br>     u<br>          ue<br>          w<br>         QA<br>          for<br>         rm<br>          y i<br>           h<br>          all<br>          qu<br>          if<br>          d<br>  ng<br>         r r<br>a<br>_ 𝑇_|Le<br>        rm<br>         win<br>    al<br>          wh<br>    g<br>            rm<br>        ng<br>            se<br>         e<br>         io<br>     se<br>          nc<br>          in<br>          T<br>          m<br>         er<br>           mp<br>           ea<br>          y,<br>          en<br>          n.<br>           by<br>   w<br>          ec<br>nd<br>=1|ng<br>        er<br>         d<br>     at<br>          er<br>     a f<br>            s<br>        th<br>            qu<br>          tra<br>         n l<br>     d a<br>          e le<br>          do<br>          ra<br>          an<br>          re<br>           or<br>           vil<br>           we<br>          ce<br> <br>            m<br>   e<br>          ur<br> se<br> a|th<br>         m<br>         ow<br>     ten<br>          e<br>      x<br>             all<br>        s.<br>            en<br>          in<br>          ay<br>      re<br>           n<br>          w<br>          ns<br>          ce<br>          du<br>           ta<br>           y d<br>            n<br>           le<br>            em<br>    wil<br>          re<br> qu<br> nd|8K<br>         o<br>          si<br>     ti<br>           th<br>      ed<br>              gl<br>            ce<br>           M<br>          er<br>       s<br>           gth<br>           si<br>          fo<br>           g<br>          ce<br>           nt<br>            et<br>            ote<br>           ng<br>            o<br>    l s<br>          nt<br> en<br>  if|del<br>          ze<br>     on<br>           e w<br>       lo<br>              ob<br>             le<br>           Q<br>          s<br>       ho<br>            a<br>           ze<br>          rm<br>           ap<br>          s<br>            to<br>            er<br>             th<br>           th<br>            ry<br>     h<br>           m<br> ce<br>   w|


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


**F.3.** **Estimating the memory boundedness of self-attention**
In the following, we calculate the ratio of memory accesses to arithmetic operations for the attention
computation for the _𝐿_ -th decode step, to show it is also memory-bound.


To simplify the following analysis, we assume that we start from an empty prompt (or equivalently
assume that the prefill contains 0 tokens).


When sampling auto-regressively from MHA or MQA, standard practice is to save the key and value
vectors in a Key-Value (KV) cache. For _𝐿_ tokens already sampled, the KV cache would therefore be of
size 2× _𝐿_ × _𝐻𝑘_ × _𝑑ℎ𝑒𝑎𝑑_ for each sequence in the batch, where _𝐻𝑘_ denotes the number of heads used for
the keys and values, and _𝑑ℎ𝑒𝑎𝑑_ denotes the dimension of the key and value vectors in each head.


For sampling the _𝐿_ -th token, once we calculate the query, key and value vectors corresponding to the
_𝐿_ -th token. The attention weights and the output of the attention layer are then computed using the _𝐿_ -th
key and value vectors in the KV cache. This requires _𝑂_ ( _𝐿𝐷_ ) operations overall and it requires loading
the _𝑂_ ( _𝐿_ × _𝐻𝑘_ × _𝑑ℎ𝑒𝑎𝑑_ ) sized KV cache from HBM, for each sequence in the minibatch. The size of the KV
cache, as well as the number of FLOPs, scales linearly with the batch size _𝐵_ .


For MHA, the number of heads for the key and values _𝐻𝑘_ is typically equal to the number of heads used
for the queries _𝐻_ . For MQA, a single head is used for keys and values, i.e., _𝐻𝑘_ = 1. Therefore for MQA,
the size of the KV cache is a factor of _𝐻𝑘_ smaller (i.e., of size 2× _𝐿_ × _𝑑ℎ𝑒𝑎𝑑_ ).

```
def attention_sampling(q, k, v):
 """ Auto-regressive sampling via attention.
 For MHA, h_k = h. For MQA, h_k = 1.
 Args:
  q : The q vector for current token of shape [b, h, k]
  k : The keys of the current + previous tokens [b, L, h_k, k]
  v : the values of the current + previous tokens [b, L, h_k, v]
 """
 logits = einsum("bhk,bLk->bhL", q, k) # O(bhLk)
 weights = softmax(logits)
 output = einsum("bhL,bLv->bhv", weights, v) # O(bhLv)
 return output

```

For a batch size of _𝐵_, the memory access to FLOPs ratio for the attention computation goes as
_𝑂_ ( _[𝐵]_ [×] _[𝐿]_ _𝐵_ [×] × _[𝐻]_ _𝐿_ _[𝑘]_ × [×] _𝐷_ _[𝑑][ℎ𝑒𝑎𝑑]_ ). For typical Transformer architectures, _𝐷_ = _𝐻_ × _𝑑ℎ𝑒𝑎𝑑_ and further _𝐻𝑘_ = _𝐻_ for MHA

and _𝐻𝑘_ = 1 for MQA. Therefore the memory access to flops ratio is _𝑂_ (1) for MHA and _𝑂_ (1/ _𝐻_ ) for MQA.
As explained in 3, in order to be compute bound on TPU-v3 a FLOPs-to-byte ratio of 136 is required,
and therefore both MHA and MQA would typically be memory bound. Nevertheless, MQA significantly
speeds up Transformer inference (when compared to MHA), since it lowers the memory boundedness
by a factor of _𝐻_ .


**F.4.** **Cache sizes**
In the following we do an analysis of the relative sizes of caches used in our recurrent and Transformers.
All caches sizes scale linearly with batch size and in the following we assume _𝐵_ = 1.


_**F.4.1.**_ _**The size of the KV cache**_
For attention, the KV cache has size 2 _𝑁𝑇ℎ𝑘𝑑ℎ𝑒𝑎𝑑_, where _𝑁_ denotes the number of attention layers (the
depth), _𝑇_ denotes the length of the sequence, _ℎ𝑘_ denotes the number of KV heads and _𝑑ℎ𝑒𝑎𝑑_ denotes
the head dimension. Throughout this work, _𝑑ℎ𝑒𝑎𝑑_ = 128. For MHA, _ℎ𝑘𝑑ℎ𝑒𝑎𝑑_ = _𝐷_, while for MQA, _ℎ𝑘_ = 1.
(We therefore expect MQA to be faster when decoding long sequences than MHA, since the size of the
KV cache is significantly smaller and less memory needs to be moved.)


23


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models


For either MHA or MQA the size of the KV cache can exceed the number of model parameters when
the sequence length _𝑇_ is large. We therefore expect to observe a transition from a ‘parameter bound’
regime when the sequence length is short, during which the decoding speed is dominated by the time
taken to load the model parameters on device, to a ‘cache bound’ regime for large sequences, where
the decoding speed is dominated by the time taken to load the KV cache.


_**F.4.2.**_ _**The size of the recurrent state**_
The recurrent state of a single RG-LRU layer has size _𝐷𝑅𝑁𝑁_, and the total state size for the entire Hawk
model is _𝑁𝐷𝑅𝑁𝑁_ ≈ 4 _𝐵𝑁𝐷_ /3. Unlike the KV cache, this state does not grow with sequence length and
is very small in comparison to parameter size. We therefore expect the decoding speed of RG-LRU to
be dominated by the time taken to load the model parameters on device at all sequence lengths.


A similar consideration applies to the size of the 1D convolution state size. For a temporal filter width
of size 4, the state has size (4−1) _𝐷𝑅𝑁𝑁_ = 3 _𝐷𝑅𝑁𝑁_ = 4 _𝐷_ for each recurrent block which is also substantially
smaller than parameter sizes.


_**F.4.3.**_ _**The local attention cache**_
A single local MQA layer has cache size upper bounded by 2 _𝑇𝑊𝑆𝑑ℎ𝑒𝑎𝑑_, where _𝑇𝑊𝑆_ denotes the local
attention window size. So long as _𝑇𝑊𝑆_ ≲ _𝐷_ [2] /( _𝐵𝑑ℎ𝑒𝑎𝑑_ ), the size of the local attention cache is also small
relative to the parameter count. We therefore expect the decoding speed of Griffin to be similar to the
decoding speed of the Hawk model.

### **G. ImprovingNextTokenPredictionwithLongerContexts: AdditionalResults**


Figure 10 shows an additional result demonstrating next token prediction performance at different
context lengths on a held out dataset of arXiv articles. We find that the results on this dataset are
qualitatively similar to the results shown in Figure 5.



3.50

3.25

3.00

2.75

2.50

2.25

2.00

1.75

1.50



|Griffin<br>Hawk<br>MQA NoPE<br>MQA RoPE|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||


128 256 512 1K 2K 4K 8K 16K 32K
Token position



|Griffin-2k<br>Griffin-8k<br>Hawk-2k<br>Hawk-8k|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


128 256 512 1K 2K 4K 8K 16K 32K 65K131K
Token position



3.50

3.25

3.00

2.75

2.50

2.25

2.00

1.75

1.50



Figure 10 | The evaluation performance of 1B parameter models across a range of sequence lengths
on held-out evaluation sets of ArXiv articles. On the left, we compare the performance of different
models trained with sequence length 2048, evaluated with a sequence length of up to 32,768. On the
right, we compare Griffin and Hawk when trained respectively on 2048 (2k) and 8192 (8k) sequence
lengths. Results are qualitatively similar to the evaluation on Books presented in Figure 5.

### **H. Additional Details of the Copy and Retrieval Tasks**


Figure 11 is an illustration of the Selective Copying and Induction Heads tasks.


In the Selective Copying task, the model needs to learn to copy data tokens (coloured tokens in Figure
11) from a sequence while ignoring noise tokens (white tokens in Figure 11). Crossed out tokens in
the output in Figure 6 denote tokens that are masked out in the loss.


24


Grifn: Mixing Gated Linear Recurrences with Local Attention for Efcient Language Models







(a) Selective Copying Task



(b) Induction Heads Task Task



Figure 11 | An illustration of the Selective Copying (left) and the Induction Heads tasks (right).


In the Induction Heads task, the model needs to learn to recall the token immediately following a special
token (black token in Figure 11). As before, crossed out tokens in the output denote tokens that are
masked out in the loss.


25


