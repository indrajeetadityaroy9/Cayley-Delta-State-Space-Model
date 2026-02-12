## Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

Tri Dao [∗][1] and Albert Gu [∗][2]


1
Department of Computer Science, Princeton University


2
Machine Learning Department, Carnegie Mellon University
tri@tridao.me, agu@cs.cmu.edu


**Abstract**


While Transformers have been the main architecture behind deep learning’s success in language modeling, state-space

models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale.

We show that these families of models are actually quite closely related, and develop a rich framework of theoretical

connections between SSMs and variants of attention, connected through various decompositions of a well-studied class

of structured _semiseparable_ _matrices_ . Our state space duality (SSD) framework allows us to design a new architecture
( **Mamba-2** ) whose core layer is an a refinement of Mamba’s selective SSM that is 2-8× faster, while continuing to be

competitive with Transformers on language modeling.

### **1 Introduction**


Transformers, in particular decoder-only models (e.g. GPT (Brown et al. 2020), Llama (Touvron, Lavril, et al. 2023)) which

process input sequences in a causal fashion, are one of the main drivers of modern deep learning’s success. Numer
ous approaches attempt to approximate the core attention layer to address its efficiency issues (Tay et al. 2022), such as

scaling quadratically in sequence length during training and requiring a cache of size linear in sequence length during

autoregressive generation. In parallel, a class of alternative sequence models, structured state-space models (SSMs), have

emerged with linear scaling in sequence length during training and constant state size during generation. They show

strong performance on long-range tasks (e.g. S4 (Gu, Goel, and Ré 2022)) and recently matched or beat Transformers on

language modeling (e.g. Mamba (Gu and Dao 2023)) at small to moderate scale. However, the development of SSMs have

appeared disjoint from the community’s collective effort to improve Transformers, such as understanding them theoreti
cally as well as optimizing them on modern hardware. As a result, it is more difficult to understand and experiment with

SSMs compared to Transformers, and it remains challenging to train SSMs as efficiently as Transformers from both an

algorithmic and systems perspective.


Our main goal is to develop a rich body of theoretical connections between structured SSMs and variants of attention. This

will allow us to transfer algorithmic and systems optimizations originally developed for Transformers to SSMs, towards

the goal of building foundation models that perform better than Transformers while scaling more efficiently in sequence

length. A milestone contribution in this direction was the **Linear Attention (LA)** framework (Katharopoulos et al. 2020),

which derived a connection between autoregressive attention and linear RNNs by showing the equivalence between “dual

forms” of quadratic kernelized attention and a particular linear recurrence. This duality allows new capabilities such as

the ability to have both efficient parallelizable training and efficient autoregressive inference. In the same spirit, this

paper provides multiple viewpoints connecting linear-complexity SSMs with quadratic-complexity forms to combine the

strengths of SSMs and attention. [1]


∗Alphabetical by last name.

1Technically speaking, these connections only relate to certain flavors of attention; the title of this paper is an homage to Katharopoulos et al. (2020)

which first showed that “Transformers are RNNs”.


1


**State** **Space** **Duality.** Our framework connecting structured SSMs and variants of attention, which we call **struc-**
**tured** **state** **space** **duality** (SSD), is made through the
abstractions of **structured** **matrices** : matrices with sub
quadratic parameters and multiplication complexity. We de
velop two broad frameworks for representing sequence mod
els, one as matrix transformations and one as tensor contrac
tions, which each reveal different perspectives of the duality.

Our technical contributions include:


- We show an equivalence between state space models and a

well-studied family of structured matrices called **semisep-**
**arable** **matrices** (Section 3). This connection is at the

heart our framework, revealing new properties and algo
rithms for SSMs. A central message of this paper is that

_different methods of computing state space models can be re-_
_framed as various matrix multiplication algorithms on struc-_
_tured matrices_ .


- We significantly improve the theory of linear atten
tion (Katharopoulos et al. 2020). We first provide an in
cisive proof of its recurrent form through the language of

tensor contractions, and then generalize it to a new family

of **structured masked attention (SMA)** (Section 4).


- We connect SSMs and SMA, showing that they have a large

intersection that are duals of each other, possessing both

SSM-like linear and attention-like quadratic forms (Sec
tion 5). We also prove that any kernel attention method

possessing a fast recurrent form must be an SSM.



**Efficient**
**Algorithms**


Sec. 6


**Structured**

**Matrices**



**Semiseparable** **Structured Masked**

**Matrices** Sec. 3 Sec. 4 **Attention (SMA)**



**Matrices**



**Structured Masked**



Sec. 3 Sec. 4



**State Space** Sec. 5
**Attention**
**Models (SSM)**



Sec. 5



**State Space**
**Duality (SSD)**



Sec. 7


**Mamba-2**


Figure 1: ( **Structured** **State-Space** **Duality** .) This paper

fleshes out the relationship between state space models

and attention through the bridge of structured matrices.



Beyond its intrinsic theoretical value, our framework opens up a broad set of directions for understanding and improving

sequence models.


**Efficient Algorithms.** First and most importantly, our framework exposes new efficient and easily-implementable algorithms for computing SSMs (Section 6). We introduce a new **SSD algorithm**, based on block decompositions of semisepa
rable matrices, that takes advantage of both the linear SSM recurrence and quadratic dual form, obtaining optimal tradeoffs

on all main efficiency axes (e.g. training and inference compute, memory usage, and ability to leverage matrix multipli
cation units on modern hardware). A dedicated implementation of SSD is 2 − 8× faster than the optimized selective scan
implementation of Mamba, while simultaneously allowing for much larger recurrent state sizes (8× the size of Mamba or

even higher, with minimal slowdown). SSD is highly competitive with optimized implementations of softmax attention

(FlashAttention-2 (Dao 2024)), crossing over at sequence length 2K and 6× faster at sequence length 16K.


**Architecture** **Design.** One major obstacle to adopting new architectures such as SSMs is the ecosystem tailored to

Transformers, such as hardware-efficient optimization and parallelism techniques for large-scale training. Our framework

allows using established conventions and techniques for attention to build a vocabulary of architecture design choices for

SSMs, and further improve them (Section 7). For example, we introduce the analog of heads from multi-head attention

(MHA) to SSMs. We show that the Mamba architecture is a **multi-input SSM (MIS)** that turns out to be analogous to
**multi-value attention (MVA)**, and compare other variants of Mamba with different head structures.


We also use these ideas to make slight modifications to the Mamba block, which allows tensor parallelism to be imple
mented (e.g. in the style of Megatron (Shoeybi et al. 2019)). The main ideas include introducing grouped-value attention

(GVA) head structure, and moving all data-dependent projections to occur in parallel at the beginning of the block.


The combination of the modified parallel Mamba block, together with using SSD as the inner SSM layer, results in the

**Mamba-2** architecture. We investigate Chinchilla scaling laws for Mamba-2 in the same setting as Mamba, finding that

it Pareto dominates Mamba and Transformer++ in both perplexity and wall-clock time. We additionally train a family of


2


Mamba-2 models at varying sizes on the Pile, showing that it matches or outperforms Mamba and open source Trans
formers on standard downstream evaluations. For example, Mamba-2 with 2.7B parameters trained on 300B tokens on

the Pile outperforms Mamba-2.8B, Pythia-2.8B and even Pythia-6.9B trained on the same dataset.


**Systems Optimizations.** The SSD framework connects SSMs and Transformers, allowing us to leverage a rich body of

work on systems optimizations developed for Transformers (Section 8).


- For example, Tensor Parallelism (TP) is an important model parallelism technique to train large Transformer models

by splitting each layer across GPUs on the same node. We design Mamba-2 to be TP-friendly, reducing the number of

synchronization point per block by half.


- For very long sequences whose activations do not fit on one device, sequence parallelism has been developed for the

attention blocks. We describe how to train SSMs in general and Mamba-2 in particular with sequence parallelism, by

passing the recurrent states between devices.


- For finetuning with examples of different lengths, for best efficiency, Transformer requires sophisticated techniques to

remove padding tokens and perform attention on variable length sequences. We show how Mamba-2 can be trained

with variable sequence lengths efficiently, requiring no padding tokens.


Section 9 empirically validates Mamba-2 on language modeling, training efficiency, and a difficult multi-query associative

recall task (Arora, Eyuboglu, Zhang, et al. 2024). Finally, in Section 10, we provide an extended related work and discuss

potential research directions opened up by our framework.


[Model code and pre-trained checkpoints are open-sourced at https://github.com/state-spaces/mamba.](https://github.com/state-spaces/mamba)

### **2 Background and Overview**


**2.1** **Structured State Space Models**


Structured state space sequence models (S4) are a recent class of sequence models for deep learning that are broadly related

to RNNs, CNNs, and classical state space models. They are inspired by a particular continuous system (1) that maps a
1-dimensional sequence _𝑥_ ∈ R [T] ↦→ _𝑦_ ∈ R [T] through an implicit latent state _ℎ_ ∈ R [(][T] _[,]_ [N][)] .


A general discrete form of structured SSMs takes the form of equation (1).



_ℎ𝑡_ = _𝐴ℎ𝑡_ −1 + _𝐵𝑥𝑡_ (1a)

_𝑦𝑡_ = _𝐶_ [⊤] _ℎ𝑡_ (1b)



_ℎ𝑡_ = _𝐴𝑡ℎ𝑡_ −1 + _𝐵𝑡𝑥𝑡_ (2a)

_𝑦𝑡_ = _𝐶𝑡_ [⊤] _[ℎ][𝑡]_ (2b)



where _𝐴_ ∈ R [(][N] _[,]_ [N][)] _, 𝐵_ ∈ R [(][N] _[,]_ [1][)] _,𝐶_ ∈ R [(][N] _[,]_ [1][)] . Structured SSMs are so named because the _𝐴_ matrix controlling the temporal
dynamics must be _structured_ in order to compute this sequence-to-sequence transformation efficiently enough to be used

in deep neural networks. The original structures introduced were diagonal plus low-rank (DPLR) (Gu, Goel, and Ré 2022)

and diagonal (Gu, Gupta, et al. 2022; Gupta, Gu, and Berant 2022; J. T. Smith, Warrington, and Linderman 2023), which

remains the most popular structure.


In this work, we use the term state space model (SSM) to refer to structured SSMs. There are many flavors of such

SSMs, with deep ties to several major paradigms of neural sequence models such as continuous-time, recurrent, and

convolutional models (Gu, Johnson, Goel, et al. 2021). We provide a brief overview below, and refer to prior work for

more context and details (Gu 2023; Gu and Dao 2023).


**Continuous-time Models.** The original structured SSMs originated as continuous-time maps on functions _𝑥_ ( _𝑡_ ) ∈ R ↦→
_𝑦_ ( _𝑡_ ) ∈ R, rather than operating directly on sequences. In the continuous-time perspective, in equation (1a) the matrices
( _𝐴, 𝐵_ ) are not directly learned but generated from underlying parameters ( _𝐴,_ [˚] _𝐵_ [˚] ), along with a parameterized step size Δ.
The “continuous parameters” (Δ _,_ _𝐴,_ [˚] _𝐵_ [˚] ) are converted to “discrete parameters” ( _𝐴, 𝐵_ ) through fixed formulas _𝐴_ = _𝑓𝐴_ (Δ _,_ _𝐴_ [˚] )
and _𝐵_ = _𝑓𝐵_ (Δ _,_ _𝐵_ [˚] ), where the pair ( _𝑓𝐴, 𝑓𝐵_ ) is called a _discretization rule_ .


**Remark 1.** _While our main models adopt the same parameterization and discretization step as prior work (see Gu and Dao_
_(2023)_ _for_ _details),_ _for_ _simplifying_ _exposition_ _and_ _notation_ _we_ _omit_ _it_ _in_ _the_ _rest_ _of_ _this_ _paper._ _We_ _note_ _that_ _prior_ _work_ _on_


3


_structured SSMs referred to the continuous parameters_ ( _𝐴,_ [˚] _𝐵_ [˚] ) _and discrete parameters_ ( _𝐴, 𝐵_ ) _as_ ( _𝐴, 𝐵_ ) _and_ ( _𝐴,_ [¯] _𝐵_ [¯] ) _instead; we_
_have changed notation to simplify the presentation and focus directly on the discrete parameters, which govern the main SSM_
_recurrence._


**Recurrent Models.** Equations (1) and (2) take the form of a recurrence which is linear in its input _𝑥_ . Structured SSMs

can therefore be viewed as types of recurrent neural networks (RNNs), where the linearity endows them with additional

properties and allows them to avoid the sequential computation of traditional RNNs. Conversely, despite this simplifi
cation, SSMs are still fully expressive as sequence transformations (in the sense of universal approximation) (Kaul 2020;

Orvieto et al. 2023; Shida Wang and Xue 2023).


**Convolutional Models.** When the SSM’s dynamics are constant through time as in equation (1), the model is called
**linear time-invariant (LTI)** . In this case, they are equivalent to convolutions. Thus, SSMs can also be viewed as types
of CNNs, but where (i) the convolution kernels are implicitly parameterized through the SSM parameters ( _𝐴, 𝐵,𝐶_ ) and (ii)

the convolution kernels are generally global instead of local. Conversely, through classical signal processing theory all

sufficiently well-behaved convolutions can be represented as SSMs.


Commonly, previous LTI SSMs would use the convolutional mode for efficient parallelizable training (where the whole

input sequence is seen ahead of time), and switched into recurrent mode (1) for efficient autoregressive inference (where

the inputs are seen one step at a time).


**Selective** **State** **Space** **Models.** The form (2) where the parameters ( _𝐴, 𝐵,𝐶_ ) can also vary in time was introduced in
Mamba as the **selective SSM** . Compared to the standard LTI formulation (1), this model can selectively choose to focus

on or ignore inputs at every timestep. It was shown to perform much better than LTI SSMs on information-dense data
such as language, especially as its state size N increases allowing for more information capacity. However, it can only

be computed in recurrent instead of convolutional mode, and requires a careful hardware-aware implementation to be

efficient. Even so, it is still less efficient than hardware-friendly models such as CNNs and Transformers because it does

not leverage matrix multiplication units, which modern accelerators such as GPUs and TPUs are specialized for.


While _time-invariant_ SSMs are closely related to continuous, recurrent, and convolutional sequence models, they are not
directly related to attention. In this paper, we show a deeper relationship between _selective_ SSMs and attention, and use
it to significantly improve the training speed of SSMs while simultaneously allowing for much larger state sizes N.


**Structured SSMs as Sequence Transformations.**


**Definition 2.1.** _We use the term_ _**sequence transformation**_ _to refer to a parameterized map on sequences 𝑌_ = _𝑓𝜃_ ( _𝑋_ ) _where_
_𝑋,𝑌_ ∈ R [(][T] _[,]_ [P][)] _and 𝜃_ _is an arbitrary collection of parameters._ T _represents the sequence or_ time _axis; subscripts index into the_
_first dimension, e.g. 𝑋𝑡,𝑌𝑡_ ∈ R [P] _._


Sequence transformations (e.g. SSMs, or self-attention) are the cornerstone of deep sequence models, where they are

incorporated into neural network architectures (e.g. Transformers). The SSM in (1) or (2) is a sequence transformation
with P = 1; it can be generalized to P _>_ 1 by simply broadcasting across this dimension (in other words, viewing the input
as P independent sequences and applying the SSM to each). One can think of P as a **head** **dimension**, which we will

elaborate on in Section 7.


**Definition** **2.2.** _We_ _define_ _the_ _**SSM**_ _**operator**_ SSM( _𝐴, 𝐵,𝐶_ ) = SSM( _𝐴_ 0: _𝑇_ _, 𝐵_ 0: _𝑇_ _,𝐶_ 0: _𝑇_ ) _as_ _the_ _sequence_ _transformation 𝑋_ ∈
R [(][T] _[,]_ [P][)] ↦→ _𝑌_ ∈ R [(][T] _[,]_ [P][)] _defined by equation_ (2) _._


In SSMs, the N dimension is a free parameter called the **state size** or state dimension. We also call it the **state expansion**
**factor**, because it expands the size of the input/output by a factor of _𝑁_, with implications for the computational efficiency

of these models.


Finally, we remark that many types of sequence transformations, such as attention, can be represented as a single matrix

multiplication across the sequence dimension.


**Definition 2.3.** _We call a sequence transformation 𝑌_ = _𝑓𝜃_ ( _𝑋_ ) _a_ _**matrix transformation**_ _if it can be written in the form_
_𝑌_ = _𝑀𝜃𝑋_ _where 𝑀_ _is a matrix depending on the parameters 𝜃_ _._ _We identify the sequence transformation with the matrix 𝑀,_
_and often drop the dependence on 𝜃_ _when clear from context._


4


**2.2** **Attention**


Attention broadly refers to a type of computation that assigns scores to every pair of positions in a sequence, allowing

each element to “attend” to the rest. By far the most common and important variant of attention is softmax self-attention,

which can be defined as


_𝑌_ = softmax( _𝑄𝐾_ [⊤] ) · _𝑉_


for _𝑄, 𝐾,𝑉_ ∈ R [(][T] _[,]_ [P][)] . The mechanism of pairwise comparisons (induced by materializing _𝑄𝐾_ [⊤] ) leads to the characteristic

quadratic training cost of attention.


Many variants of attention have been proposed, but all share the underlying core of these attention scores, with various

approximations (Tay et al. 2022). The most important variant for this work is **linear attention** (Katharopoulos et al. 2020).

Roughly speaking, this family of methods drops the softmax by folding it into a kernel feature map, and uses associativity
of matrix multiplication to rewrite ( _𝑄𝐾_ [⊤] ) - _𝑉_ = _𝑄_ - ( _𝐾_ [⊤] _𝑉_ ). Moreover, in the important case of causal (autoregressive)
attention, they show that when the causal mask is incorporated into the left-hand side as ( _𝐿_ - _𝑄𝐾_ [⊤] ) - _𝑉_, where _𝐿_ is the

lower-triangular 1’s matrix, then the right-hand side can be expanded as a recurrence. Several recent and concurrent works

such as RetNet (Y. Sun et al. 2023) and GateLoop (Katsch 2023) strengthen this to more general forms of _𝐿_ (Section 10). In

this work, our formulation of structured masked attention will strongly generalize these ideas.


**2.3** **Structured Matrices**


General matrices _𝑀_ ∈ R [(][T] _[,]_ [T][)] require T [2] parameters to represent and _𝑂_ (T [2] ) time to perform basic operations such as
matrix-vector multiplication. **Structured matrices** are those that


(i) can be represented in subquadratic (ideally linear) parameters through a compressed representation, and


(ii) have fast algorithms (most importantly matrix multiplication) by operating directly on this compressed represen
tation.


Perhaps the most canonical families of structured matrices are sparse and low-rank matrices. However, there exist many

other families, such as Toeplitz, Cauchy, Vandermonde, and butterfly matrices, which have all been used in machine

learning for efficient models (Dao, Gu, et al. 2019; D. Fu et al. 2024; Gu, Gupta, et al. 2022; Thomas et al. 2018). Structured

matrices are a powerful abstraction for efficient representations and algorithms. In this work, we will show that SSMs

are equivalent to another class of structured matrices that have not previously been used in deep learning, and use this

connection to derive efficient methods and algorithms.


**2.4** **Overview: Structured State Space Duality**


While this paper develops a much richer framework of connections between SSMs, attention, and structured matrices, we

provide a brief summary of the main method, which is actually quite self-contained and simple algorithmically.


**Recurrent (Linear) Form.** The state space dual (SSD) layer can be defined as a special case of the selective SSM (2).

The standard computation of an SSM as a recurrence (or parallel scan) can be applied, which has linear complexity in

sequence length. Compared to the version used in Mamba, SSD has two minor differences:


   - The structure on _𝐴_ is further simplified from diagonal to _scalar times identity_ structure. Each _𝐴𝑡_ can also be identified

with just a scalar in this case.


   - We use a larger head dimension P, compared to P = 1 used in Mamba. Typically P = {64 _,_ 128} is chosen which is

similar to conventions for modern Transformers.


Compared to the original selective SSM, these changes can be viewed as slightly decreasing the expressive power in

return for significant training efficiency improvements. In particular, our new algorithms will allow the use of matrix

multiplication units on modern accelerators.


5


**Dual (Quadratic) Form.** The dual form of SSD is a quadratic computation closely related to attention, defined as



( _𝐿_                       - _𝑄𝐾_ [⊤] ) · _𝑉_ _𝐿𝑖𝑗_ =


where _𝑎𝑖_ are input-dependent scalars bounded in [0 _,_ 1].




_𝑎𝑖_ × · · · × _𝑎_ _𝑗_ +1 _𝑖_ ≥ _𝑗_

0 _𝑖_ _<_ _𝑗_



Compared to standard softmax attention, there are two main differences


   - The softmax is dropped.


   - The attention matrix is multiplied elementwise-wise by an additional mask matrix _𝐿_ .


Both of these changes can be viewed as addressing problems in vanilla attention. For example, the softmax has been

recently observed to cause problems in attention scores, such as the “attention sink” phenomenon (Darcet et al. 2024;

Xiao et al. 2024). More importantly, the mask matrix _𝐿_ can be viewed as replacing the heuristic positional embeddings

of Transformers with a different _data-dependent positional mask_ that controls how much information is transfered across

time.


More broadly, this form is an instance of our **structured masked attention** generalization of linear attention, defined in

Section 4.


**Matrix Form and SSD Algorithm.** The various forms of SSD are connected through a unified matrix representation,
by showing that SSMs have a matrix transformation form _𝑌_ = _𝑀𝑋_ for a matrix _𝑀𝜃_ ∈ R [(][T] _[,]_ [T][)] that depends on _𝜃_ = ( _𝐴, 𝐵,𝐶_ ).

In particular, the dual form of SSD is equivalent to naive (quadratic-time) multiplication by the matrix _𝑀_, and the recurrent

form is a particular efficient (linear-time) algorithm that leverages the structure in _𝑀_ .


Going beyond these, _any_ algorithm for multiplication by _𝑀_ can be applied. Our proposed hardware-efficient SSD algo
rithm (Section 6) is a new structured matrix multiplication method that involves block decompositions of _𝑀_, which obtains

better efficiency tradeoffs than either the pure linear or quadratic forms. It is relatively simple and easy-to-implement

compared to general selective SSMs (Gu and Dao 2023); Listing 1 provides a complete implementation in a few lines of

code.


Figure 1 provides a simple roadmap of the relationships between the concepts presented in this paper.


**2.5** **Notation**


Throughout this paper, we prefer using precise notation that can be mapped to code.


**Matrices and Vectors.** We generally use lower case to denote vectors (i.e. tensors with a single axis) and upper case to

denote matrices (i.e. tensors with more than one axes). We do not bold matrices in this work. Sometimes, if a matrix is

tied or repeated along one axis (and hence can also be viewed as a vector), we may use either upper or lower case for it. [2]


- denotes scalar or matrix multiplication while ◦ denotes Hadamard (elementwise) multiplication.


**Indexing.** We use Python-style indexing, e.g. _𝑖_ : _𝑗_ refers to the range ( _𝑖,𝑖_ + 1 _, . . ., 𝑗_ - 1) when _𝑖_ _<_ _𝑗_ and ( _𝑖,𝑖_ - 1 _, . . ., 𝑗_ + 1)
when _𝑖_ _>_ _𝑗_ . For example, for any symbol _𝑣_ we let _𝑣_ _𝑗_ : _𝑖_ for _𝑗_ ≥ _𝑖_ denote the sequence ( _𝑣_ _𝑗, . . ., 𝑣𝑖_ +1). [ _𝑖_ ] is equivalent to
0 : _𝑖_ = (0 _, . . .,𝑖_ - 1). For shorthand, we also let _𝑣_ [×] _𝑗_ : _𝑖_ [denote the product] _[ 𝑣]_ _[𝑗]_ [× · · · ×] _[ 𝑣][𝑖]_ [+][1][.][3]


**Dimensions.** To distinguish from matrices and tensors, we often use capital letters in typewriter fonts (e.g. D _,_ N _,_ T) to
denote dimensions and tensor shapes. Instead of the traditional notation _𝑀_ ∈ R _[𝑇]_ [×] _[𝑇]_ we frequently use _𝑀_ ∈ R [(][T] _[,]_ [T][)] to

reflect tensor shapes in code.


**Tensor Contractions.** We will heavily rely on **tensor contraction** or **einsum** notation both for clarity and as a central

tool in stating and proving our results. We assume the reader to be familiar with this notation, which is commonly used


2In this work, this happens only with the _𝐴_ parameter of SSMs.
3In some contexts, it is always clear that the notation _𝑎𝑖_ : _𝑗_ or _𝐴𝑖_ : _𝑗_ means _𝑎𝑖_ ×: _𝑗_ [, and the superscript is omitted.]


6


in modern tensor libraries such as numpy. For example, we can use contract(MN _,_ NK → MK) to denote the matrix-matrix
multiplication operator, and in our notation contract(MN _,_ NK → MK)( _𝑋,𝑌_ ) (which is equivalent to _𝑋_ - _𝑌_ ) can be translated
to code as numpy _._ einsum( [′] mn _,_ nk → mk [′] _,_ X _,_ Y).


A large glossary of notation is included in Appendix A.

### **3 State Space Models are Structured Matrices**


This section explores different perspectives of the state space model as a sequence transformation, and outlines properties

and algorithms of such maps. The main results of this section are about the equivalence between state space models

and a family of structured matrices called semiseparable matrices, which imply new efficiency results (Theorems 3.5

and 3.7).


**3.1** **The Matrix Transformation Form of State Space Models**


Recall that our definition of an SSM is defined as a parameterized map defined through (2). Our theoretical framework
starts by simply writing this transformation as a matrix multiplication mapping the vectors _𝑥_ ∈ R [T] ↦→ _𝑦_ ∈ R [T] .


By definition, _ℎ_ 0 = _𝐵_ 0 _𝑥_ 0. By induction,


_ℎ𝑡_ = _𝐴𝑡_ _. . . 𝐴_ 1 _𝐵_ 0 _𝑥_ 0 + _𝐴𝑡_ _. . . 𝐴_ 2 _𝐵_ 1 _𝑥_ 1 + · · · + _𝐴𝑡𝐴𝑡_ −1 _𝐵𝑡_ −2 _𝑥𝑡_ −2 + _𝐴𝑡_ _𝐵𝑡_ −1 _𝑥𝑡_ −1 + _𝐵𝑡𝑥𝑡_



=



_𝑡_
∑︁

_𝐴𝑡_ [×] : _𝑠_ _[𝐵][𝑠][𝑥][𝑠]_ _[.]_

_𝑠_ =0



Multiplying by _𝐶𝑡_ to produce _𝑦𝑡_ and vectorizing the equation over _𝑡_ ∈[T], we derive the matrix transformation form of

SSMs.



(3)



_𝑦𝑡_ =



_𝑡_
∑︁

_𝐶𝑡_ [⊤] _[𝐴]_ _𝑡_ [×] : _𝑠_ _[𝐵][𝑠][𝑥][𝑠]_

_𝑠_ =0



_𝑦_ = SSM( _𝐴, 𝐵,𝐶_ )( _𝑥_ ) = _𝑀𝑥_

_𝑀𝑗𝑖_                      - _𝐶_ [⊤] _𝑗_ _[𝐴]_ _[𝑗]_ [· · ·] _[𝐴][𝑖]_ [+][1] _[𝐵][𝑖]_


**3.2** **Semiseparable Matrices**


_𝑀_ in equation (3) is a particular representation of a class of matrices known as semiseparable matrices. Semiseparable

matrices are a fundamental matrix structure. We first define these matrices and their properties.


**Definition 3.1.** _A (lower triangular) matrix 𝑀_ _is_ N _-semiseparable if every submatrix contained in the lower triangular portion_
_(i.e. on or below the diagonal) has rank at most_ N _. We call_ N _the_ order _or_ rank _of the semiseparable matrix._


Definition 3.1, and other forms of related “separable” structure (e.g. quasiseparable matrices and other definitions of

semiseparable matrices) are sometimes called **structured rank matrices** (or rank-structured matrices) because they are

characterized by rank conditions on their submatrices. Semiseparable matrices have many structured representations

including the hierarchical semiseparable (HSS), sequential semiseparable (SSS), and Bruhat forms (Pernet and Storjohann

2018). We will primarily use the SSS form.


**3.2.1** **The Sequentially Semiseparable (SSS) Representation**


**Definition 3.2.** _A lower triangular matrix 𝑀_ ∈ R [(][T] _[,]_ [T][)] _has a_ N _-_ _**sequentially semiseparable (SSS)**_ _representation if it can_
_be written in the form_
_𝑀𝑗𝑖_ = _𝐶_ [⊤] _𝑗_ _[𝐴]_ _[𝑗]_ [· · ·] _[𝐴][𝑖]_ [+][1] _[𝐵][𝑖]_ (4)


_for vectors 𝐵_ 0 _, . . ., 𝐵_ T−1 _,𝐶_ 0 _, . . .,𝐶_ T−1 ∈ R [N] _and matrices 𝐴_ 0 _, . . .,𝐴_ T−1 ∈ R [(][N] _[,]_ [N][)] _._


_We define the operator_ SSS _so that 𝑀_ = SSS( _𝐴_ 0:T _, 𝐵_ 0:T _,𝐶_ 0:T) _._


7


A fundamental result of semiseparable matrices is that they are exactly equivalent to matrices with SSS representations.

One direction can be deduced with a simple constructive proof.


**Lemma 3.3.** _An_ N _-SSS matrix 𝑀_ _with representation_ (4) _is_ N _-semiseparable._


_Proof._ Consider any off-diagonal block _𝑀𝑗_ : _𝑗_ [′] _,𝑖_ [′] : _𝑖_ where _𝑗_ [′] _>_ _𝑗_ ≥ _𝑖_ _> 𝑖_ [′] . This has an explicit rank-N factorization as



_𝐶_ [⊤] _. . ._ _𝐶_ [⊤]
_𝑗_ _[𝐴]_ [×] _𝑗_ : _𝑖_ [′] _[𝐵][𝑖]_ [′] _𝑗_ _[𝐴]_ [×] _𝑗_ : _𝑖_ −1 _[𝐵][𝑖]_ [−][1]
_..._ _..._
_𝐶_ [⊤] _. . ._ _𝐶_ [⊤]
_𝑗_ [′] −1 _[𝐴]_ [×] _𝑗_ [′] −1: _𝑖_ [′] _[𝐵][𝑖]_ [′] _𝑗_ [′] −1 _[𝐴]_ [×] _𝑗_ [′] −1: _𝑖_ −1 _[𝐵][𝑖]_ [−][1]













_𝐴_ [×] _𝑗_ : _𝑖_ −1 - _𝐴𝑖_ ×−1: _𝑖_ [′] _[𝐵][𝑖]_ [′] - · · _𝐴𝑖_ [×] −1: _𝑖_ −1 _[𝐵][𝑖]_ [−][1] - _._ (5)


                


=



_𝐶_ [⊤]
_𝑗_ _[𝐴]_ [×] _𝑗_ : _𝑗_
_..._
_𝐶_ [⊤]
_𝑗_ [′] −1 _[𝐴]_ [×] _𝑗_ [′] −1: _𝑗_





Equation (5) will be used extensively in deriving our fast algorithms for sequence models. The other direction is well
established in the literature on semiseparable matrices.


**Proposition 3.4.** _Every_ N _-semiseparable matrix has a_ N _-SSS representation._


Furthermore, note that although Definition 3.2 involves _𝑂_ (N [2] T) parameters for the representation (in particular to store the
_𝐴_ matrices), it can actually be compressed down to _𝑂_ (NT) parameters, which is asymptotically tight (Pernet, Signargout,

and Villard 2023). Therefore in the rest of this paper we will conflate the structured matrix class (Definition 3.1) and a

particular representation of it (Definition 3.2); we will always use this representation instead of other candidates. In turn
we will use N-SS to refer to an N-semiseparable matrix in SSS form.


Semiseparable matrices are a fundamental matrix structure and have many important properties. They are deeply related

to recurrences at large, and can be defined by multiple characterizations (e.g. Definitions 3.1 and 3.2) which reveal different

connections and efficient algorithms for them. We mention some of their other properties in Appendix C.1.


**Remark** **2.** _The_ _notion_ _of_ _semiseparability_ _is_ _very_ _broad_ _and_ _many_ _similar_ _but_ _subtlely_ _different_ _definitions_ _appear_ _in_ _the_
_literature; our definitions may differ slightly from other conventions. First, because we are primarily concerned with causal or_
_autoregressive settings in this paper, we have restricted the definition of semiseparability to the triangular case; Definition 3.1_
_more formally might be called_ (N _,_ 0) _-semiseparability by some authors. Some authors may also instead refer to it as a form of_
_quasiseparability (Eidelman and Gohberg 1999; Pernet 2016). See Vandebril et al. (2005) for a brief survey._


**3.2.2** **1-Semiseparable Matrices: the Scalar SSM Recurrence**


We will single out the special case of 1-SS matrices. Note that in this case, the _𝐶_ _𝑗_ and _𝐵𝑖_ are scalars, and can be factored

out of the SSS representation (4) (we also use lower-case to emphasize that the parameters are scalars in this case)


SSS( _𝑎,𝑏,𝑐_ ) = diag( _𝑐_ ) · _𝑀_              - diag( _𝑏_ ) where _𝑀𝑗𝑖_ = _𝑎_ [×] _𝑗_ : _𝑖_ _[.]_


Since diagonal matrices are easy to handle (e.g. multiplication by a diagonal matrix is the same as elementwise scalar

multiplication), we can ignore these terms. Thus our basic representation of a 1-SS matrix is _𝑀𝑗𝑖_ = _𝑎_ _𝑗_ : _𝑖_ or



_𝑀_ = 1SS( _𝑎_ 0: _𝑇_ ) 


1

_𝑎_ 1 1

_𝑎_ 2 _𝑎_ 1 _𝑎_ 2 1
_..._ _..._ _..._ _..._

_𝑎𝑇_ −1 _. . . 𝑎_ 1 _𝑎𝑇_ −1 _. . . 𝑎_ 2 _. . ._ _𝑎𝑇_ −1 1









_._ (6)



The importance of 1-SS matrices lies in their equivalence to the minimal form of a scalar recurrence - the case of a
degenerate SSM with state dimension N = 1 and no ( _𝐵,𝐶_ ) projections. Note that multiplication _𝑦_ = _𝑀𝑥_ can be computed

by the recurrence
_𝑦𝑡_ = _𝑎𝑡_ :0 _𝑥_ 0 + · · · + _𝑎𝑡_ : _𝑡𝑥𝑡_
= _𝑎𝑡_ ( _𝑎𝑡_ −1:0 _𝑥_ 0 + · · · + _𝑎𝑡_ −1: _𝑡_ −1 _𝑥𝑡_ −1) + _𝑎𝑡_ : _𝑡𝑥𝑡_ (7)
= _𝑎𝑡𝑦𝑡_ −1 + _𝑥𝑡_ _._


8


**Outputs**



𝑌



Sequence
Transformation
Matrix 𝑀



**P**



Sequence dim. **T**



**State Space Models** are **Semiseparable Matrix Transformations**



**Inputs**



𝑋



Figure 2: ( **State Space Models are Semiseparable Matrices** .) As sequence transformations, state space models can be
represented as a matrix transformation _𝑀_ ∈ R [(][T] _[,]_ [T][)] acting on the sequence dimension T, sharing the same matrix for each
channel in a head ( _Left_ ). This matrix is a semiseparable matrix ( _Right_ ), which is a rank-structured matrix where every
submatrix contained on-and-below the diagonal ( _Blue_ ) has rank at most N, equal to the SSM’s state dimension.


We thus also refer to matrix multiplication by 1-SS matrices as the **scalar SSM recurrence** or the cumprodsum (cumu
lative product sum; a generalization of cumulative product and cumulative sum) operator. As the fundamental form of

recurrence, multiplication by 1-SS matrices is important as a building block for our main algorithms.


We emphasize that one of the central themes of this paper is that _many_ _algorithms_ _on_ _sequence_ _models_ _can_ _be_ _reduced_
_to_ _structured_ _matrix_ _multiplication_ _algorithms_ . 1-SS matrices exemplify this connection: there are many fast algorithms
for computing the primitive scalar recurrence or cumprodsum operator, and all of them turn out to be equivalent to dif
ferent structured factorization of 1-SS matrices. We dedicate Appendix B to these algorithms for 1-SS matrix multiplica
tion.


**3.3** **State Space Models are Semiseparable Matrices**


Recall that our definition of an SSM is defined as a parameterized map defined through Definition 2.1. The connection

between SSMs and semiseparable matrices follows from simply writing this transformation as a matrix multiplication
mapping the vectors _𝑥_ ↦→ _𝑦_ ∈ R [T] .


Equation (3) directly establishes the link between state space models and the sequentially semiseparable representation,

which in turn are equivalent to semiseparable matrices in general (Lemma 3.3 and Proposition 3.4).


**Theorem 3.5.** _The state space model transformation 𝑦_ = SSM( _𝐴, 𝐵,𝐶_ )( _𝑥_ ) _with state size_ N _is identical to matrix multiplica-_
_tion by an_ N _-SS matrix in sequentially semiseparable representation 𝑦_ = SSS( _𝐴, 𝐵,𝐶_ ) · _𝑥._


In other words the sequence transformation operator SSM (Definition 2.2) coincides with the matrix construction operator SSS (Definition 3.2), and we use them interchangeably (or sometimes SS as shorthand). Furthermore—by a twist of

fate—structured state space models and sequentially semiseparable matrices have the same acronyms, underscoring their

equivalence! Conveniently we can use any of these acronyms SSM (state space model or semiseparable matrix), SSS (struc
tured state space or sequentially semiseparable), or SS (state space or semiseparable) interchangeably to unambiguously

refer to either concept. However, we will generally use the convention that SSM refers to state space model, SS refers to

semiseparable, and SSS refers to sequentially semiseparable.


Figure 2 illustrates the sequence transformation perspective of state space models as semiseparable matrices.


9


**3.4** **Computing State Space Models through Structured Matrix Algorithms**


The reason Theorem 3.5 is important is that it will allow us to _reduce_ _the_ _problem_ _of_ _efficient_ _computation_ _of_ _SSMs_ _(and_
_other_ _sequence_ _models)_ _into_ _efficient_ _algorithms_ _for_ _structured_ _matrix_ _multiplication_ . We briefly provide an overview and

defer our main new algorithm to Section 6, after showing the equivalence of SSMs to other sequence models in Sections 4

and 5.


As previously defined, semiseparable matrices (i.e. rank-structured matrices) are a classical type of structured matrix:


(i) They have compressed representations such as the SSS form which has only _𝑂_ (T) instead of _𝑂_ (T [2] ) parameters.


(ii) They have fast algorithms operating directly on the compressed representation.


Furthermore, the parameterization and matrix multiplication cost can be tight in the semiseparable order.


**Proposition 3.6** (Pernet, Signargout, and Villard (2023)) **.** _An_ N _-SS matrix of size_ T _can be represented in 𝑂_ (NT) _parameters_
_and has matrix-vector multiplication in time and space 𝑂_ (NT) _._


For example, 1-SS matrices illustrate the essence of this connection. The matrix _𝑀_ = 1SS( _𝑎_ ) is defined by exactly T − 1
parameters _𝑎_ 0:T−1 = _𝑎_ 1 _, . . .,𝑎_ T−1, and can be computed in _𝑂_ (T) time by following the scalar recurrence (7).


**3.4.1** **The Linear (Recurrent) Mode**


Proposition 3.6 can be easily seen in the case of diagonal structured SSMs (S4D (Gu, Gupta, et al. 2022)), simply by

leveraging the state space model formulation (2) and unrolling the recurrence. We provide the formal tensor-contraction
algorithm in (8), where the dimension S is equal to T [4] .


_𝑍_ = contract(SP _,_ SN → SPN)( _𝑋, 𝐵_ ) (S _,_ P _,_ N) (8a)

_𝐻_ = contract(TSN _,_ SPN → TPN)( _𝐿,𝑍_ ) (T _,_ P _,_ N) (8b)

_𝑌_ = contract(TN _,_ TPN → TP)( _𝐶, 𝐻_ ) (T _,_ P) (8c)


Here, _𝐿_ ∈ R [(][T] _[,]_ [T][)] is defined as 1SS( _𝐴_ ), or in other words _𝐿_ 0:T _,_ 0:T = 1SS( _𝐴_ 0:T) for _𝑖_ ∈[N]. This algorithm involves three steps

corresponding to (2):


(i) _expanding_ the input _𝑋_ by the input matrix _𝐵_ (8a),


(ii) unrolling independent scalar SSM recurrences (8b), and


(iii) _contracting_ the hidden state _𝐻_ by the output matrix _𝐶_ (8c).


Note that we have used the equivalence between scalar SSMs and 1-SS matrices in step (8b).


**Remark** **3.** _We_ _note_ _that_ (8) _is_ _a_ _special_ _case_ _of_ _the_ _Mamba_ _(S6)_ _model._ _however,_ _a_ _naive_ _implementation_ _is_ _slow_ _because_
_of the expanded tensors 𝑍_ _and 𝐻_ _of size_ (T _,_ P _,_ N) _; Gu and Dao (2023) introduced a hardware-aware implementation to avoid_
_materializing these tensors._


Surprisingly, Theorem 3.5 and Proposition 3.6 immediately imply that all SSMs have the same asymptotic efficiency as

algorithm (8).


**Theorem 3.7.** _Any state space model (Definition 2.2) of state size_ N _on sequence length_ T _can be computed in time 𝑂_ (TN) _(not_
_accounting for potential preprocessing)._


We note that this result is new to the structured SSM literature. In particular, given dense unstructured _𝐴𝑡_ matrices, the
total representation alone seems to be of size _𝑂_ (TN [2] ). Thus Theorem 3.7 states the non-trivial result that with a pre
processing step, even an unstructured SSM can be computed optimally efficiently, with upper bound matching the lower
bound _𝑂_ (TN) given by the size of _𝐵_ and _𝐶_ .


**Remark** **4.** _Theorem_ _3.7_ _is_ _perhaps_ _not_ _too_ _surprising_ _in_ _light_ _of_ _the_ _fact_ _that_ _almost_ _all_ _dense_ _matrices_ _over_ R [(][N] _[,]_ [N][)] _are_
_diagonalizable over_ C _, leading to the result that_ almost all _dense real SSMs are equivalent to a diagonal complex SSM. This_
_fact underlies the reason why diagonal SSMs are the most popular form of structured SSM (Gu, Gupta, et al. 2022; Gupta, Gu,_


4A different symbol is required for the contraction notation.


10


_and Berant 2022;_ _J. T. Smith, Warrington, and Linderman 2023)._ _However, Theorem 3.7 implies the much stronger result for_
all _real SSMs (not just the diagonalizable ones), as well as dense SSMs over other fields (including_ C _itself)._


In practice, efficiently computable SSMs still require additional structure on _𝐴_, particularly to avoid the expensive preprocessing step (which both has order N extra FLOPs and involves hardware-inefficient operations such as singular value

decompositions). These structures are the focus of past work on structured SSMs (e.g. S4(D) and Mamba) as well as our

new algorithms. In particular, when slightly stronger structure is imposed on _𝐴_, we will design very hardware-efficient

algorithms through block decompositions of the SSM matrix _𝑀_ = SSS( _𝐴, 𝐵,𝐶_ ) in Section 6.


**3.4.2** **The Quadratic (Naive) Mode**


We note that there is another way to compute an SSM exposed by our new matrix point of view. A naive computation of

the matrix SSM representation (3) involves simply materializing the sequence transformation matrix _𝑀_ = SSS( _𝐴, 𝐵,𝐶_ ).
This is a (T _,_ T) matrix, and therefore this naive algorithm will scale quadratically in sequence length. However, when the
sequence length T is short, this can actually be more efficient than the linear algorithm due to constant factors and the

hardware-friendliness of the computation pattern (e.g. leveraging matrix-matrix multiplications). In fact, for a particular

case of structured SSMs, this looks very similar to a quadratic attention computation (Section 5).


**3.4.3** **Summary**


Many sequence models are explicitly motivated or defined as matrix sequence transformations - most notably Trans
formers, where the matrix mixer is the attention matrix. On the other hand, RNNs and SSMs have not previously been

described in this way. By providing an explicit _matrix transformation_ form of state space models, we reveal new ways of

understanding and using them. From a computational perspective, any method of computing the forward pass of a state

space model can be viewed as a matrix multiplication algorithm on semiseparable matrices. The semiseparable matrix

perspective provides one lens into state space duality (SSD), where the dual modes respectively refer to a linear-time

semiseparable matrix multiplication algorithm and quadratic-time naive matrix multiplication.


Moreover, leveraging the rich structure of semiseparable matrices can lead to even better algorithms and more insights (e.g.

Section 6 and Appendix B). In Appendix C.1, we describe some additional properties of semiseparable matrices.

### **4 Structured Masked Attention: Generalizing Linear Attention** **with Structured Matrices**


In this section we revisit the linear attention framework from first principles. The main results in this section are a simple

tensor-contraction-based proof of linear attention (Proposition 4.1), and our generalized abstraction of structured masked

attention in Definition 4.2. We note that this section derives the main duality results from a different direction than state

space models and can be read completely independently of Section 3.


   - Section 4.1 sets up our framework for variants of attention, with a particular focus on kernel attention and masked

kernel attention.


   - Section 4.2 provides our first main attention result, a simple proof of linear attention through the lens of tensor

contractions.


   - Section 4.3 defines structured masked attention, our generalization of prior attention variants through structured

matrices.


11


**4.1** **The Attention Framework**


**4.1.1** **Attention**


The basic form of (single-head) attention is a map on three sequences of vectors ( _𝑄, 𝐾,𝑉_ ) ↦→ _𝑌_ .


_𝑄_ = input (T _,_ N)

_𝐾_ = input (S _,_ N)

_𝑉_ = input (S _,_ P)

_𝐺_ = _𝑄𝐾_ [⊤] (T _,_ S)


_𝑀_ = _𝑓_ ( _𝐺_ ) (T _,_ S)


_𝑌_ = _𝐺𝑉_ (T _,_ P)



(9)



We use “shape annotations” to indicate the dimensions of tensors, e.g. _𝑄_ ∈ R [(][T] _[,]_ [N][)] . In this general form, S and T represent
_source_ and _target_ sequence lengths, N represents the _feature dimension_, and P represents the _head dimension_ .


The most common variant of **softmax attention** uses a softmax activation _𝑓_ = softmax to normalize the rows of the _𝐺_

matrix.


**4.1.2** **Self-Attention**


Our treatment is motivated by the most important case of self-attention, where


(i) the source and target sequences are the same (i.e. S = T),


(ii) usually the feature and head dimensions are the same (i.e. N = P),


(iii) and _𝑄, 𝐾,𝑉_ are generated by linear projections on the same input vector ( _𝑄_ = _𝑊𝑄_  - _𝑋, 𝐾_ = _𝑊𝐾_  - _𝑋,𝑉_ = _𝑊𝑉_  - _𝑋_ ).


However, our presentation abstracts away these choices and begins from the _𝑄, 𝐾,𝑉_ matrices.


**Remark** **5.** _Our_ _focus_ _is_ _on_ _the_ _self-attention_ _case_ _with_ _equal_ _head_ _and_ _feature_ _dimensions_ _(i.e._ S = T _and_ N = P _),_ _which_
_should_ _be_ _used_ _as_ _the_ _running_ _example._ _We_ _define_ _the_ _general_ _formulation_ _of_ _attention_ _not_ _only_ _so_ _that_ _our_ _framework_
_captures_ _variants_ _such_ _as_ _cross-attention,_ _but_ _also_ _because_ _separating_ _the_ _notation_ _for_ _dimensions_ _(e.g._ S _and_ T _)_ _makes_ _the_
_contraction notation proofs of our main results in this section more clear._


**Remark 6.** _While attention is usually framed as an operation on these three inputs 𝑄, 𝐾,𝑉_ _which are viewed symmetrically,_
_the input and output dimensions in_ (9) _indicate otherwise._ _In particular, the feature dimension_ N _is not present in the output;_
_therefore_ _in_ _the_ _case_ _when_ S = T _(e.g._ _self-attention),_ _we_ _view 𝑉_ _as_ _the_ _main_ _input,_ _so_ _that_ (9) _defines_ _a_ _proper_ _sequence_
_transformation 𝑉_ ↦→ _𝑌_ _(Definition 2.1)._


**4.1.3** **Kernel Attention**


The step where the softmax function is applied to the Gram matrix _𝐺_ can be decomposed into two parts:


1. Exponentiating the _𝐺_ matrix.


2. Normalizing the _𝐺_ matrix on the S axis.


We can ignore the normalization term for now, as it amounts to simply passing in _𝑉_ = 1 and dividing (we revisit this in

Section 7.3). The exponentiation term can be viewed as a kernel transformation: there is an (infinite-dimensional) feature
map _𝜑_ such that exp( _𝑄𝐾_ [⊤] ) = _𝜑_ ( _𝑄_ ) _𝜑_ ( _𝐾_ ) [⊤] . By abstracting away the feature map into the definition of _𝑄_ and _𝐾_ itself

(i.e. define _𝑄, 𝐾_ as the post-transformed versions), we can ignore the softmax transformation, and assume that _𝑄, 𝐾_ are
arbitrarily generated by kernel feature maps and potentially N ≠ P.


Many instantiations of kernel attention have been proposed, including:


   - The original Linear Attention (Katharopoulos et al. 2020) defines the kernel feature map as an arbitrary pointwise

activation function, such as _𝑥_ ↦→ 1 + elu( _𝑥_ ).


   - Random Feature Attention (RFA) (H. Peng et al. 2021) chooses the kernel feature map to approximate softmax

attention (i.e. the exp feature map) using the random Fourier feature approximation of Gaussian kernels (Rahimi


12


and Recht 2007). This involves random projections (i.e. multiplying _𝑄_ and _𝐾_ by a random projection _𝑊_ and applying

the activation _𝑥_ ↦→(cos( _𝑥_ ) _,_ sin( _𝑥_ )).


   - Performer (Choromanski et al. 2021) proposes the fast attention via positive orthogonal random features (FAVOR+).

The positive random features (PRF) part chooses the kernel feature map to be a random projection followed by the
feature map _𝑥_ ↦→ 2 [−][1][/][2] (exp( _𝑥_ ) _,_ exp(− _𝑥_ )). This choice is motivated so that the kernel elements are positive-valued

and provably approximates the softmax attention. [It also proposes choosing the random projections in orthogonal

directions, which we do not consider.]


   - cosFormer (Qin, Weixuan Sun, et al. 2022) augment RFA with a cosine reweighting mechanism that incorpo
rates positional information to emphasize locality. This effectively passes _𝑄𝑡, 𝐾𝑡_ through the feature map _𝑥_ ↦→
( _𝑥_ cos( _𝜋𝑡_ /2 _𝑇_ ) _,_ sin( _𝜋𝑡_ /2 _𝑇_ )).


   - Linear Randomized Attention (Zheng, C. Wang, and Kong 2022) generalize RFA from the perspective of impor
tance sampling, and generalize it to provide better estimates of the full softmax kernel (rather than just the exp
transformed numerator).


Other related attention variants include Linformer (Sinong Wang et al. 2020) and Nyströformer (Xiong et al. 2021), which

both use low-rank approximations of the attention matrix _𝑀_ (and are thus compatible with equation (9)), through random

projections (Johnson-Lindenstrauss) and kernel approximation (the Nyström method) respectively.


**4.1.4** **Masked (Kernel) Attention**


Let _𝐿_ be a mask of shape (T _,_ S). Most commonly, in the _autoregressive_ self-attention case when S = T, _𝐿_ may be a lowertriangular matrix of 1’s representing a _causal mask_ . Besides enforcing causality, many other types of masks can be applied

- in particular various sparsity patterns such as banded, dilated, or block diagonal – which are motivated by reducing the

complexity of dense attention.


Masked attention is usually written in matrix notation as


_𝑦_ = ( _𝐿_ ◦( _𝑄𝐾_ [⊤] )) · _𝑉._ (10)


More precisely, with shape annotations and breaking this down into the precise sequence of computations:


_𝐺_ = _𝑄𝐾_ [⊤] (T _,_ S)


_𝑀_ = _𝐺_              - _𝐿_ (T _,_ S) (11)


_𝑌_ = _𝑀𝑉_ (T _,_ P)


Our improved derivation of attention variants in this section starts by noticing that this formula can be written as a _single_
_contraction_ :
_𝑌_ = contract(TN _,_ SN _,_ SP _,_ TS → TP)( _𝑄, 𝐾,𝑉, 𝐿_ ) (12)


and the algorithm in (11) can be reframed as computing (12) by a particular ordering of pairwise contractions


_𝐺_ = contract(TN _,_ SN → TS)( _𝑄, 𝐾_ ) (T _,_ S) (13a)

_𝑀_ = contract(TS _,_ TS → TS)( _𝐺, 𝐿_ ) (T _,_ S) (13b)

_𝑌_ = contract(TS _,_ SP → TP)( _𝑀,𝑉_ ) (T _,_ P) (13c)


**4.2** **Linear Attention**


Linear attention, and many other variants of efficient attention, is often motivated by changing the order of matrix associativity in the core attention computation ( _𝑄𝐾_ [⊤] ) _𝑉_ = _𝑄_ ( _𝐾_ [⊤] _𝑉_ ). However when the mask is added, the derivation is

somewhat less straightforward (for example, the original paper (Katharopoulos et al. 2020) and variants (Y. Sun et al. 2023)

state the formula without proof).


Roughly, the linear attention method claims that the following formula is equivalent to (10), which must be verified by

expanding the sum and tracking indices carefully.


_𝑌_ = _𝑄_                   - cumsum( _𝐾_ [⊤] _𝑉_ ) (14)


13


**Proposition 4.1** ((Katharopoulos et al. 2020)) **.** _Autoregressive kernel attention, i.e. masked kernel attention with the causal_
_mask, can be computed in 𝑂_ ( _𝑇_ ) _time by a recurrence taking constant time per step._


**4.2.1** **A Tensor Contraction Proof of Linear Attention**


We present a simple and rigorous derivation of linear attention that will also immediately reveal how to generalize it. The

main idea is to perform the contraction (12) in an alternate order. We avoid ambiguous matrix notation and work directly

with contraction notation:


_𝑍_ = contract(SP _,_ SN → SPN)( _𝑉, 𝐾_ ) (S _,_ P _,_ N) (15a)

_𝐻_ = contract(TS _,_ SPN → TPN)( _𝐿,𝑍_ ) (T _,_ P _,_ N) (15b)

_𝑌_ = contract(TN _,_ TPN → TP)( _𝑄, 𝐻_ ) (T _,_ P) (15c)


Intuitively, we interpret this contraction order as follows.


The first step (15a) performs an “expansion” into more features, by a factor of the feature dimension N. The third step

(15c) contracts the expanded feature dimension away. If _𝐾_ is viewed as the input (Remark 6), then _𝑉_ and _𝑄_ perform the

expansion and contraction, respectively.


The second step is the most critical, and explains the _linear_ part of linear attention. First notice that (15b) is just a direct
matrix multiplication by _𝐿_ (since the (P _,_ N) axes can be flattened). Also note that this is the only term that involves both
T and S axes, hence should have Ω(TS) complexity (i.e. quadratic in sequence length). However, when the mask _𝐿_ is

the standard causal attention mask (lower triangular 1’s), matrix-vector multiplication by _𝐿_ is identical to a feature-wise

cumulative sum







_𝑦_ =







1
_..._ _..._

1 _. . ._ 1



_𝑦_ 0 = _𝑥_ 0
_𝑥_ ⇐⇒ _._
_𝑦𝑡_ = _𝑦𝑡_ −1 + _𝑥𝑡_



**4.3** **Structured Masked Attention**


With the tensor contraction perspective of masked attention (15), we can immediately see that the crux of the original

linear attention is the fact that _matrix-vector multiplication by the causal mask is equivalent to the cumulative sum opera-_
_tor_ .


However, we observe that there is no reason the attention mask has to be all 1’s. All that is necessary for linear attention

to be fast is for _𝐿_ to be a _structured matrix_, which by definition are those that have fast matrix multiplication (Section 2.3).
In particular, we can use _any_ _mask_ _matrix_ _𝐿_ that has sub-quadratic (ideally linear) matrix-vector multiplication, which

would have the same complexity as standard linear attention by speeding up the bottleneck equation (15b).


**Definition** **4.2.** _**Structured**_ _**masked**_ _**attention**_ _**(SMA)**_ _(or_ _**structured**_ _**attention**_ _for_ _short)_ _is_ _defined_ _as_ _a_ function _on_
_queries/keys/values 𝑄, 𝐾,𝑉_ _as_ _well_ _as_ _any_ structured matrix _𝐿_ _(i.e._ _has_ _sub-quadratic_ _matrix_ _multiplication),_ _through_ _the_
_4-way tensor contraction_


_𝑌_ = contract(TN _,_ SN _,_ SP _,_ TS → TP)( _𝑄, 𝐾,𝑉, 𝐿_ ) _._


_The_ _SMA_ _**quadratic**_ _**mode**_ _**algorithm**_ _is_ _the_ _sequence_ _of_ _pairwise_ _contractions_ _defined_ _by_ (13) _,_ _which_ _corresponds_ _to_ _the_
_standard (masked) attention computation._


_The_ _SMA_ _**linear**_ _**mode**_ _**algorithm**_ _is_ _the_ _sequence_ _of_ _pairwise_ _contractions_ _defined_ _by_ (15) _,_ _where_ _step_ (15b) _is_ _optimized_
_through the subquadratic structured matrix multiplication._


We can instantiate structured masked attention to any given class of matrix structure. Some examples include (Fig
ure 3):


   - Linear attention uses a causal mask.


   - RetNet (Y. Sun et al. 2023) uses a decay mask _𝐿𝑖𝑗_ = _𝛾_ _[𝑖]_ [−] _[𝑗]_   - I[ _𝑗_ ≥ _𝑖_ ] for some decay factor _𝛾_ ∈[0 _,_ 1].


14


**Keys**
#### "



**Sequence Transformation**
#### Structured Mask $ Matrix %



**Causal Mask**


**Decay Mask**


**1-semiseparable**


**Toeplitz**


**Discrete Fourier**
**Transform**



**SSD**



**Queries**
#### !


#### =



**Linear Attention**


**Retentive Network**


**1-SS Structured Attention**



**Toeplitz Structured Attention**


**Fourier Structured Attention**



Figure 3: ( **Structured Masked Attention** .) SMA constructs a masked attention matrix _𝑀_ = _𝑄𝐾_ [⊤] - _𝐿_ for any structured
matrix _𝐿_, which defines a matrix sequence transformation _𝑌_ = _𝑀𝑉_ . All instances of SMA have a dual subquadratic form

induced by a different contraction ordering, combined with the efficient structured matrix multiplication by _𝐿_ . Previous

examples include Linear Attention (Katharopoulos et al. 2020) and RetNet (Y. Sun et al. 2023). Beyond SSD (1-semiseparable

SMA), the focus of this paper, many other potential instantiations of structured attention are possible.


   - The decay mask could be generalized to a Toeplitz matrix _𝐿𝑖𝑗_ = _𝛼𝑖_   - _𝑗_ for some learnable (or input-dependent) set of
parameters _𝛼_ ∈ R [T] . This can be interpreted as a form of relative positional encoding, reminiscent of other methods

such as AliBi (Press, N. Smith, and Lewis 2022) but multiplicative instead of additive.


   - Another variant could use a Fourier matrix _𝐿𝑖𝑗_ = _𝜔_ _[𝑖𝑗]_ [/][T] to encode positional structure a different way.


In Section 5, we consider semiseparable SMA, which defines our main SSD model.


**4.3.1** **Summary: The Dual Forms of Masked Attention**


Standard (masked kernel) attention is often conflated between a function and an algorithm. Separating this distinction

presents a clear way to understand different variants of attention.


   - We view **masked attention** as a particular _function_ (12).


   - The standard **quadratic attention** computation (13) can be viewed as an _algorithm_ to compute the function.


   - **Linear attention** (15) is an alternate algorithm to compute the same function.


Moreover, in this case


   - The masked attention function is simply a particular _contraction on four terms_ .


   - The quadratic and linear attention algorithms are simply _two different orders to perform the contractions_ .


It is known that contraction orderings can make large differences in computation complexity, leading to the quadratic vs.

linear split. Just as state space models are a transformation that can be computed in multiple ways, with dual quadratic

vs. linear forms (Section 3.4), linear attention has a similar duality that results from two contraction orders. In fact, these

turn out to be different perspectives on the same underlying duality, which we make explicit in Section 5.


15


### **5 State Space Duality**

In Sections 3 and 4, we defined structured state space models and structured attention, discussed their properties, and

showed that they both have a quadratic algorithm and a linear algorithm. This section connects them together. Our main

result is showing that a particular case of structured state space models coincides with a particular case of structured

attention, and that the linear-time SSM algorithm and quadratic-time kernel attention algorithm are dual forms of each

other.


   - Section 5.1 specializes state space models to scalar structure, where the naive quadratic computation can be seen as

an instance of kernel attention.


   - Section 5.2 specializes structured masked attention to semiseparable SMA, which characterizes masked attention

with efficient autoregression.


   - Section 5.3 summarizes the connection between structured masked attention and structured state space models,

termed structured state space duality.


**5.1** **Scalar-Identity Structured State Space Models**


In Section 3 we showed that state space models are equivalent to semiseparable matrix transformations, resulting in both

a linear recurrent form and quadratic naive form.


Recall that SSMs are defined by _𝑦_ = SSM( _𝐴, 𝐵,𝐶_ )( _𝑥_ ), and the matrix form of SSMs uses the SSS (sequentially semiseparable) representation _𝑀_ = SSS( _𝐴, 𝐵,𝐶_ ) where _𝑀𝑗𝑖_ = _𝐶_ [⊤] _𝑗_ _[𝐴]_ _[𝑗]_ [:] _[𝑖][𝐵][𝑖]_ [(equation (][3][)).]


Now let us consider the case where _𝐴_ _𝑗_ is simply a scalar; in other words, an instantiation of a structured SSM where the
_𝐴_ matrices are _extremely_ structured: _𝐴_ = _𝑎𝐼_ for scalar _𝑎_ and identity matrix _𝐼_ . Then we can rearrange


_𝑀𝑗𝑖_ = _𝐴_ _𝑗_ : _𝑖_                        - ( _𝐶_ [⊤] _𝑗_ _[𝐵][𝑖]_ [)] _[.]_


And this can be vectorized into


_𝐿_                      - 1SS( _𝑎_ )

_𝑀_ = _𝐿_ ◦( _𝐶𝐵_ [⊤] )


where _𝐵,𝐶_ ∈ R [(][T] _[,]_ [N][)] .


Using this formulation, the full output _𝑌_ = _𝑀𝑋_ is computed precisely as


_𝐺_ = contract(TN _,_ SN → TS)( _𝐶, 𝐵_ ) (T _,_ S)

_𝑀_ = contract(TS _,_ TS → TS)( _𝐺, 𝐿_ ) (T _,_ S) (16)

_𝑌_ = contract(TS _,_ SP → TP)( _𝑀,𝑋_ ) (T _,_ P)


where S = T. But this is exactly the same as original definition of masked kernel attention definition (13)!


Therefore, as alluded to in Section 3.4, _naively_ _computing_ _the_ _scalar_ _structured_ _SSM—by_ _materializing_ _the_ _semiseparable_
_matrix_ _𝑀_ _and_ _performing_ _quadratic_ _matrix-vector_ _multiplication—is_ _exactly_ _the_ _same_ _as_ _quadratic_ _masked_ _kernel_ _atten-_
_tion._


**5.2** **1-Semiseparable Structured Masked Attention**


Structured masked attention allows for the use of any structured mask _𝐿_ . When _𝐿_ is the causal mask, it is standard linear

attention. Note that the causal mask is _𝐿_ = SS(1 _𝑇_ ), i.e. the 1-SS mask is generated by _𝑎𝑡_ = 1 in definition (6). This motivates
generalizing _𝐿_ to the class of 1-semiseparable masks, or **1-semiseparable structured masked attention (1-SS SMA)**,
where the cumsum in linear attention’s recurrence is replaced by a more general recurrence – the scalar SSM scan, i.e.

1-semiseparable matrix multiplication (Section 3.2.2).


Finally, the most important reason we consider 1-semiseparable SMA is because the linear form for computing it is a

special case of diagonal state space model. The linear form of SMA is algorithm (15), where the bottleneck step (15b)


16


can be viewed as matrix multiplication by the 1-SS mask. In Section 3, we also wrote out the computation for a diagonal

SSM (8), where the bottleneck step (8b) is a scalar SSM recurrence which is equivalent to 1-SS multiplication. The only
difference is that (8b) has an extra N dimension in _𝐿_, because the matrix _𝐴_ is a diagonal matrix of size N. This N dimension

would disappear if all diagonal entries of _𝐴_ are the same, which results in Corollary 5.1.


**Corollary 5.1.** _1-SS SMA (masked attention with 1-semiseparable structured matrices 𝐿)_ (15) _is a special case of a diagonal_
_SSM_ (8) _where the diagonal matrix is a scalar multiple of the identity._


While Corollary 5.1 says that 1-SS SMA has an efficient recurrent form, we can also show a converse result that charac
terizes which instances of SMA has efficient autoregression.


**Theorem** **5.2.** _For_ _any_ _instantiation_ _of_ _structured_ _masked_ _attention_ _(Definition_ _4.2)_ _that_ _is_ _an_ _autoregressive_ _process_ _with_
_bounded order, the structured mask 𝐿_ _must be a semiseparable matrix._


In other words, efficient autoregressive attention is general _semiseparable SMA_ . Theorem 5.2 is proved in Appendix C.2.


**Remark 7.** _While 1-semiseparable SMA is a special case of a state space model, general semiseparable SMA is strictly more_
_expressive than 1-SS SMA, and cannot be described by a standard SSM. However, the semiseparable multiplication by 𝐿_ _and_
_the linear form of SMA (equation_ (15a) _) each involve an expansion and contraction step, and can be absorbed into a similar_
_instance of 1-SS SMA with a single (larger) expansion._


In summary, 1-semiseparable structured attention is the most important case of SMA, because it is:


   - a natural generalization of linear attention with an input-dependent recurrence.


   - the simplest case of general semiseparable attention, which is equivalent to efficient autoregressive attention.


   - a special case of a diagonal state space model.


**5.3** **Structured State-Space Duality (SSD)**


To summarize our results:


   - Structured state-space models (Section 3) are a model usually defined through a linear-time recurrence. However,

by expanding the matrix formulation characterizing its linear sequence-to-sequence transformation, one can derive

a quadratic form.


   - Attention variants (Section 4) are a model defined through quadratic-time pairwise interactions. However, by view
ing it as a four-way tensor contraction and reducing in a different order, one can derive a linear form.


   - A natural special case of each one   - more precisely, state space models with scalar-identity structure on the _𝐴_

matrices, and structured masked attention with 1-semiseparable structure on its _𝐿_ mask – are duals of each other

with the exact same linear and quadratic forms.


Figure 4 summarizes the duality between these two representations.


An extended related work and discussion (Section 10) describes the relationship between SSD and general SSMs / attention

in more detail.

### **6 A Hardware-Efficient Algorithm for SSD Models**


The benefits of developing the theoretical SSD framework between SSMs, attention, and structured matrices lies in using

the connections to improve the models and algorithms. In this section, we show how various algorithms for computing

SSD models efficiently can be derived from various algorithms for computing structured matrix multiplication.


Our main computational result is an algorithm for computing SSD models that combines both the linear (recurrent) mode

and quadratic (attention) mode. This algorithm is as computation efficient as SSMs (linear scaling in sequence length) and

as hardware-friendly as attention (primarily uses matrix multiplications).


**Theorem 6.1.** _Consider an SSD model with state expansion factor_ N _and head dimension_ P = N _._ _There exists an algorithm_
_for computing the model on any input 𝑋_ ∈ R [(][T] _[,]_ [P][)] _which only requires 𝑂_ (TN [2] ) _training FLOPs, 𝑂_ (TN) _inference FLOPs, 𝑂_ (N [2] )
_inference memory, and whose work is dominated by matrix multiplications._


17


Structured State Space Model Structured Masked Attention


_𝐶_ (contraction matrix) _𝑄_ (queries)

_𝐵_ (expansion matrix) _𝐾_ (keys)

_𝑋_ (input sequence) _𝑉_ (values)

_𝐴𝑗_ : _𝑖_ (state matrix) _𝐿𝑗𝑖_ (mask)
N (state expansion dim.) N (kernel feature dim.)


_𝐻_ (hidden states (8b))
SMA linear dual (15)
= _𝐿_  - _𝑋𝐵_ (linear mode)


_𝐺_ (Gram matrix (13a))
SSM quadratic dual (16)
= _𝑄_             - _𝐾_ [⊤] (quadratic mode)






















|Structured State Space Model (SSM)<br>S4 Diagonal State Space Model<br>DSS Scalar-Identity SSM<br>S4D RetNet GateLoop<br>S5 TransNormer<br>S6 Linear Attention<br>1-Semiseparable SMA|Col2|Col3|Structured<br>State Space<br>Duality (SSD)|
|---|---|---|---|
|**Structured State Space Model (SSM)**<br>**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS<br>S4|**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS|**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS|**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS|
|**Structured State Space Model (SSM)**<br>**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS<br>S4|**Diagonal State Space Model**<br>**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>S4D<br>S5<br>S6<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop<br>DSS|**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop|**Scalar-Identity SSM**<br>**1-Semiseparable SMA**<br>Linear Attention<br>RetNet<br>TransNormer<br>GateLoop|
|||||



Figure 4: ( **Structured** **State** **Space** **Duality** .) State space duality describes the close relationship between state space
models and masked attention. ( _Left_ ) General SSMs and SMA both possess linear and quadratic forms, with direct analogs
in notation. ( _Right_ ) SSMs and SMA intersect at a large class of _state space dual models_ (SSD) which capture many sequence

models as special cases.


Note that all of these bounds are tight, because a state space model with state expansion N operating on a head size of
N has total state size N [2] (yielding the lower bounds for training and inference FLOPs of _𝑂_ (TN [2] ) and _𝑂_ (N [2] ) respectively).
Furthermore the input _𝑋_ itself has TN elements, yielding the memory lower bound.


The main idea behind Theorem 6.1 is once again viewing the problem of computing a state space model as a semiseparable

matrix multiplication, but leveraging its structure in a new way. Instead of computing the whole matrix in either recurrent

or attention mode, we perform a _block decomposition_ of the matrix. The diagonal blocks can be computed using the dual

attention mode, which can be efficiently done with matrix multiplications, while the off-diagonal blocks can be factored

by the rank-structure of semiseparable matrices and reduced to a smaller recurrence. We highlight that Listing 1 provides

a self-contained implementation of the SSD algorithm. Compared to the general selective SSM of Gu and Dao (2023),

this implementation is much simpler, and relatively efficient even in native PyTorch without requiring special low-level

kernels.



To begin, we partition the matrix _𝑀_ into a [T]




[T] [T]

Q [×] Q



To begin, we partition the matrix _𝑀_ into a Q [×] Q [grid of submatrices of size][ Q][ ×][ Q][, for some block size][ Q][.] [Note that the]

off-diagonal blocks are low-rank by the defining property of semiseparable matrices (Definition 3.1). [5]



_𝑀_ [(][1] _[,]_ [0][)] _𝑀_ [(][1] _[,]_ [1][)]

(Block Decomposition) _𝑀_ =

_..._ _..._ _..._
_𝑀_ [(][T][/][Q][−][1] _[,]_ [0][)] _𝑀_ [(][T][/][Q][−][1] _[,]_ [1][)] _. . ._ _𝑀_ [(][T][/][Q][−][1] _[,]_ [T][/][Q][−][1][)]

 

(Diagonal Block) _𝑀_ [(] _[𝑗,𝑗]_ [)] = SSM( _𝐴_ _𝑗_ Q:( _𝑗_ +1)Q _, 𝐵_ _𝑗_ Q:( _𝑗_ +1)Q _,𝐶_ _𝑗_ Q:( _𝑗_ +1)Q)



_𝑀_ [(][0] _[,]_ [0][)]



(Block Decomposition) _𝑀_ =







_𝑀_ [(][1] _[,]_ [0][)] _𝑀_ [(][1] _[,]_ [1][)]
_..._ _..._ _..._
_𝑀_ [(][T][/][Q][−][1] _[,]_ [0][)] _𝑀_ [(][T][/][Q][−][1] _[,]_ [1][)] _. . ._ _𝑀_ [(][T][/][Q][−][1] _[,]_ [T][/][Q][−][1][)]







⊤



(Low-Rank Block) _𝑀_ [(] _[𝑗,𝑖]_ [)] =



_𝐶_ [⊤]
_𝑗_ Q _[𝐴]_ _[𝑗]_ [Q][:] _[𝑗]_ [Q][−][1]
_..._
_𝐶_ [⊤]
( _𝑗_ +1)Q−1 _[𝐴]_ [(] _[𝑗]_ [+][1][)][Q][−][1:] _[𝑗]_ [Q][−][1]





_𝐶_ [⊤]
_𝑗_ Q _[𝐴]_ _[𝑗]_ [Q][:] _[𝑗]_ [Q][−][1]
_..._
_𝐶_ [⊤]
( _𝑗_ +1)Q−1 _[𝐴]_ [(] _[𝑗]_ [+][1][)][Q][−][1:] _[𝑗]_ [Q][−][1]







_𝐴_ _𝑗_ Q−1:( _𝑖_ +1)Q−1



_𝐵_ [⊤]
_𝑖_ Q _[𝐴]_ [(] _[𝑖]_ [+][1][)][Q][−][1:] _[𝑖]_ [Q]
_..._
_𝐵_ [⊤]
( _𝑖_ +1)Q−1 _[𝐴]_ [(] _[𝑖]_ [+][1][)][Q][−][1:][(] _[𝑖]_ [+][1][)][Q][−][1]





This is easiest illustrated through an example, e.g. for T = 9 and decomposing into chunks of length Q = 3. The shaded


5Note that the block decomposition is valid even with partitions of varying size, e.g. if Q ∤ T, but we assume even divisibility for simplicity.


18


cells are low-rank factorizations of the off-diagonal blocks of the semiseparable matrix.



_𝑀_ =


=



_𝐶_ 0 [⊤] _[𝐴]_ [0:0] _[𝐵]_ [0]
_𝐶_ 1 [⊤] _[𝐴]_ [1:0] _[𝐵]_ [0] _𝐶_ 1 [⊤] _[𝐴]_ [1:1] _[𝐵]_ [1]

 _𝐶_ 2 [⊤] _[𝐴]_ [2:0] _[𝐵]_ [0] _𝐶_ 2 [⊤] _[𝐴]_ [2:1] _[𝐵]_ [1] _𝐶_ 2 [⊤] _[𝐴]_ [2:2] _[𝐵]_ [2]
 _𝐶_ 3 [⊤] _[𝐴]_ [3:0] _[𝐵]_ [0] _𝐶_ 3 [⊤] _[𝐴]_ [3:1] _[𝐵]_ [1] _𝐶_ 3 [⊤] _[𝐴]_ [3:2] _[𝐵]_ [2] _𝐶_ 3 [⊤] _[𝐴]_ [3:3] _[𝐵]_ [3]

_𝐶_ 4 [⊤] _[𝐴]_ [4:0] _[𝐵]_ [0] _𝐶_ 4 [⊤] _[𝐴]_ [4:1] _[𝐵]_ [1] _𝐶_ 4 [⊤] _[𝐴]_ [4:2] _[𝐵]_ [2] _𝐶_ 4 [⊤] _[𝐴]_ [4:3] _[𝐵]_ [3] _𝐶_ 4 [⊤] _[𝐴]_ [4:4] _[𝐵]_ [4]



_𝐶_ 5 [⊤] _[𝐴]_ [5:0] _[𝐵]_ [0] _𝐶_ 5 [⊤] _[𝐴]_ [5:1] _[𝐵]_ [1] _𝐶_ 5 [⊤] _[𝐴]_ [5:2] _[𝐵]_ [2] _𝐶_ 5 [⊤] _[𝐴]_ [5:3] _[𝐵]_ [3] _𝐶_ 5 [⊤] _[𝐴]_ [5:4] _[𝐵]_ [4] _𝐶_ 5 [⊤] _[𝐴]_ [5:5] _[𝐵]_ [5]

 _𝐶_ 6 [⊤] _[𝐴]_ [6:0] _[𝐵]_ [0] _𝐶_ 6 [⊤] _[𝐴]_ [6:1] _[𝐵]_ [1] _𝐶_ 6 [⊤] _[𝐴]_ [6:2] _[𝐵]_ [2] _𝐶_ 6 [⊤] _[𝐴]_ [6:3] _[𝐵]_ [3] _𝐶_ 6 [⊤] _[𝐴]_ [6:4] _[𝐵]_ [4] _𝐶_ 6 [⊤] _[𝐴]_ [6:5] _[𝐵]_ [5] _𝐶_ 6 [⊤] _[𝐴]_ [6:6] _[𝐵]_ [6]

_𝐶_ 7 [⊤] _[𝐴]_ [7:0] _[𝐵]_ [0] _𝐶_ 7 [⊤] _[𝐴]_ [7:1] _[𝐵]_ [1] _𝐶_ 7 [⊤] _[𝐴]_ [7:2] _[𝐵]_ [2] _𝐶_ 7 [⊤] _[𝐴]_ [7:3] _[𝐵]_ [3] _𝐶_ 7 [⊤] _[𝐴]_ [7:4] _[𝐵]_ [4] _𝐶_ 7 [⊤] _[𝐴]_ [7:5] _[𝐵]_ [5] _𝐶_ 7 [⊤] _[𝐴]_ [7:6] _[𝐵]_ [6] _𝐶_ 7 [⊤] _[𝐴]_ [7:7] _[𝐵]_ [7]
_𝐶_ 8 [⊤] _[𝐴]_ [8:0] _[𝐵]_ [0] _𝐶_ 8 [⊤] _[𝐴]_ [8:1] _[𝐵]_ [1] _𝐶_ 8 [⊤] _[𝐴]_ [8:2] _[𝐵]_ [2] _𝐶_ 8 [⊤] _[𝐴]_ [8:3] _[𝐵]_ [3] _𝐶_ 8 [⊤] _[𝐴]_ [8:4] _[𝐵]_ [4] _𝐶_ 8 [⊤] _[𝐴]_ [8:5] _[𝐵]_ [5] _𝐶_ 8 [⊤] _[𝐴]_ [8:6] _[𝐵]_ [6] _𝐶_ 8 [⊤] _[𝐴]_ [8:7] _[𝐵]_ [7] _𝐶_ 8 [⊤] _[𝐴]_ [8:8] _[𝐵]_ [8]

























































From here we can reduce the problem into these two parts. These can also be interpreted as dividing the output of a

“chunk” _𝑦_ _𝑗_ Q:( _𝑗_ +1)Q into two components: the effect of inputs within the chunk _𝑥_ _𝑗_ Q:( _𝑗_ +1)Q, and the effect of inputs before the

chunk _𝑥_ 0: _𝑗_ Q.


**6.1** **Diagonal Blocks**


The diagonal blocks are easy to handle, because they are simply self-similar problems of a smaller size. The _𝑗_ -th block
represents computing the answer SSM( _𝐴𝑅, 𝐵𝑅,𝐶𝑅_ )( _𝑥𝑅_ ) for the range _𝑅_ = _𝑗_ Q : ( _𝑗_ + 1)Q = ( _𝑗_ Q _, 𝑗_ Q + 1 _, . . ., 𝑗_ Q + Q − 1).
The key is that this block can be computed using any desired method. In particular, for small chunk lengths Q, this

problem is computed more efficiently using the dual quadratic SMA form. Additionally, the chunks can be computed in

parallel.


These subproblems can be interpreted as: what is the output per chunk _supposing that the initial state (to the chunk) is_ 0.

In other words for chunk _𝑗_, this computes the correct outputs taking into account only the chunk inputs _𝑥_ _𝑗_ Q:( _𝑗_ +1)Q.


**6.2** **Low-Rank Blocks**


The low-rank factorizations consist of 3 terms, and there are correspondingly three pieces of the computation. In this

factorization, we will use the terminology



_𝐵_ 0 [⊤] _[𝐴]_ [2:0]

- The terms like _𝐵_ 1 [⊤] _[𝐴]_ [2:1]

_𝐵_ 2 [⊤] _[𝐴]_ [2:2]





_𝐵_ 0 [⊤] _[𝐴]_ [2:0]
_𝐵_ 1 [⊤] _[𝐴]_ [2:1]
_𝐵_ 2 [⊤] _[𝐴]_ [2:2]



⊤







are called the right factors or _𝐵_ -block-factors.




- The terms like _𝐴_ 5:2 are called the center factors or _𝐴_ -block-factors.



_𝐶_ 6 [⊤] _[𝐴]_ [6:5]

- The terms like _𝐶_ 7 [⊤] _[𝐴]_ [7:5]

_𝐶_ 8 [⊤] _[𝐴]_ [8:5]

 



are called the left factors or _𝐶_ -block-factors.


19


**Semiseparable Matrix**
Block Decomposition



𝑀



**Outputs** 𝑌


**States** 𝐻


**Inputs** 𝑋



Diagonal Block: Input → Output

Low-Rank Block: Input → State

Low-Rank Block: State → State

Low-Rank Block: State → Output



Figure 5: ( **SSD** **Algorithm** .) By using the matrix transformation viewpoint of state space models to write them as

semiseparable matrices (Section 3), we develop a more hardware-efficient computation of the SSD model through a block
decomposition matrix multiplication algorithm. The matrix multiplication also has an interpretation as a state space

model, where blocks represent chunking the input and output sequence. Diagonal blocks represent intra-chunk compu
tations and the off-diagonal blocks represent inter-chunk computations, factored through the SSM’s hidden state.


**Right Factors.** This step computes the multiplication by the right _𝐵_ -block-factors of the low-rank factorization. Note
that for each chunk, this is a (N _,_ Q) by (Q _,_ P) matrix multiplication, where N is the state dimension and _𝑃_ is the head
dimension. The result is a (N _,_ P) tensor for each chunk, which has the same dimensionality as the expanded hidden state

_ℎ_ .


This can be interpreted as: what is the final state per chunk _supposing_ _that_ _the_ _initial_ _state_ _(to_ _the_ _chunk)_ _is_ 0. In other
words this computes _ℎ_ _𝑗_ Q+Q−1 assuming that _𝑥_ 0: _𝑗_ Q = 0.


**Center Factors.** This step computes the effect of the center _𝐴_ -block-factors terms in the low-rank factorization. In the
previous step, the final states per chunk have total shape (T/Q _,_ N _,_ P). This is now multiplied by a 1-SS matrix generated by
_𝐴_ [×]
2Q−1:Q−1 _[,𝐴]_ 3 [×] Q−1:2Q−1 _[, . . .,𝐴]_ T [×] −1:T−Q−1 [.]


This step can be computed by any algorithm for computing 1-SS multiplication (also known as the scalar SSM scan or
cumprodsum operator).


This can be interpreted as: what is the actual final state per chunk _taking into account all previous inputs_ ; in other words,

this computes the true hidden state _ℎ_ _𝑗_ Q taking into account all of _𝑥_ 0:( _𝑗_ +1)Q.


**Left Factors.** This step computes the multiplication by the left _𝐶_ -block-factors of the low-rank factorization. For each
chunk, this can be represented by a matrix multiplication contract(QN _,_ NP → QP).


This can be interpreted as: what is the output per chunk _taking into account the correct initial state ℎ_ _𝑗_ Q−1 _, and supposing_
_the inputs 𝑥_ _𝑗_ Q:( _𝑗_ +1)Q _are_ 0. In other words for chunk _𝑗_, this computes the correct outputs taking into account only the prior
inputs _𝑥_ 0: _𝑗_ Q.


**6.3** **Computational Cost**


We define the notation BMM(B _,_ M _,_ N _,_ K) to define a batched matrix multiplication contract(MK _,_ KN → MN) with batch dimension B. From this notation we can infer three aspects of the efficiency:


   - _Computation cost_ : total of _𝑂_ (BMNK) FLOPs.


   - _Memory cost:_ total of _𝑂_ (B(MK + KN + MN)) space.


20


**Listing 1** Full PyTorch example of the state space dual (SSD) model.

**def** segsum(x):

"""Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,

which is equivalent to a scalar SSM."""
T = x.size(-1)
x_cumsum = torch.cumsum(x, dim=-1)
x_segsum = x_cumsum[..., :, **None** ]   - x_cumsum[..., **None**, :]
mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
**return** x_segsum


**def** ssd(X, A, B, C, block_len=64, initial_states= **None** ):

"""
Arguments:

X: (batch, length, n_heads, d_head)
A: (batch, length, n_heads)
B: (batch, length, n_heads, d_state)
C: (batch, length, n_heads, d_state)
Return:

Y: (batch, length, n_heads, d_head)
"""
**assert** X.dtype == A.dtype == B.dtype == C.dtype
**assert** X.shape[1] % block_len == 0


# Rearrange into blocks/chunks
X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) **for** x **in** (X, A, B, C)]


A = rearrange(A, "b c l h -> b h c l")
A_cumsum = torch.cumsum(A, dim=-1)


# 1. Compute the output for each intra-chunk (diagonal blocks)
L = torch.exp(segsum(A))
Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)


# 2. Compute the state for each intra-chunk
# (right term of low-rank factorization of off-diagonal blocks; B terms)
decay_states = torch.exp((A_cumsum[:, :, :, -1:]   - A_cumsum))
states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)


# 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
# (middle term of factorization of off-diag blocks; A terms)
**if** initial_states **is** **None** :
initial_states = torch.zeros_like(states[:, :1])
states = torch.cat([initial_states, states], dim=1)
decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
states, final_state = new_states[:, :-1], new_states[:, -1]


# 4. Compute state -> output conversion per chunk
# (left term of low-rank factorization of off-diagonal blocks; C terms)
state_decay_out = torch.exp(A_cumsum)
Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)


# Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
**return** Y, final_state


   - _Parallelization:_ larger M _,_ N _,_ K terms can leverage specialized matrix multiplication units on modern accelerators.


**Center Blocks.** The cost of the quadratic SMA computation consists of three steps (equation (16)):


   - Computing the kernel matrix _𝐶_ [⊤] _𝐵_, which has cost BMM(T/Q _,_ Q _,_ Q _,_ N).


   - Multiplying by the mask matrix, which is an elementwise operation on tensors of shape (T/Q _,_ Q _,_ Q).


   - Multiplying by the _𝑋_ values, which has cost BMM(T/Q _,_ Q _,_ P _,_ N)


21


**Low-Rank Blocks: Right Factors.** This step is a single matrix multiplication with cost BMM(T/Q _,_ N _,_ P _,_ Q).


**Low-Rank** **Blocks:** **Center** **Factors.** This step is a scalar SSM scan (or 1-SS multiplication) of length T/Q on (N _,_ P)
independent channels. The work of this scan is TNP/Q, which is negligible compared to the other factors.


Note that because of the blocking which reduces the length of the sequence from T to T/Q, this scan has Q times smaller cost

than a pure SSM scan (e.g. the selective scan of Mamba). Thus we observe that on most problem lengths, other algorithms

(Appendix B) may be more efficient or much easier to implement without a significant slowdown. For example, a naive
implementation of this via 1-SS matrix multiplication has cost BMM(1 _,_ T/Q _,_ NP _,_ T/Q), which is much easier to implement

and can be more efficient than a naive recurrence/scan implementation.


**Low-Rank Blocks: Left Factors.** This step is a single matrix multiplication with cost BMM(T/Q _,_ Q _,_ P _,_ N).


**Total Cost.** If we set N = P = Q (in other words the state dimension, head dimension, and chunk length are equal), then
all BMM terms above become BMM(T/N _,_ N _,_ N _,_ N). The computational chacteristics of this are:


   - Total FLOP count of _𝑂_ (TN [2] ).


   - Total memory of _𝑂_ (TN).


   - The work _consists primarily of matrix multiplications_ on matrices of shape (N _,_ N).


Notice that the memory consumption is tight; the inputs and outputs _𝑥,𝑦_ have shape (T _,_ P) = (T _,_ N). Meanwhile the
flop count reflects an extra factor of N, which is cost incurred by the autoregressive state size and is common to all

models.


Aside from the matmuls, there is a scalar SSM scan on NP = N [2] features and sequence length T/Q. This has cost _𝑂_ (T/QN [2] )
FLOPs and _𝑂_ (log(T/Q)) depth. Although it does not use matrix multiplications, it is still parallelizable and the total work

done is negligible compared to the other steps; this has a negligible cost in our GPU implementation.


**Comparison to Pure SSM and Attention Models.** Quadratic attention is also very hardware efficient by only leveraging matrix multiplications, but has T [2] _𝑁_ total FLOPs. Its slower computation speed at both training and inference can

directly be seen as a consequence of having a larger state size – standard attention has a state size scaling with sequence
length T because it caches its history and does not compress its state.


Linear SSMs have TNP = TN [2] total FLOPs, which is the same as SSD. However, a naive implementation requires a state

expansion (15a) that materializes extra memory, and a scalar operation (15b) that does not leverage matrix multiplica
tions.


Attention SSM **SSD**


State size T N N
Training FLOPs T [2] N TN [2] TN [2]

Inference FLOPs TN N [2] N [2]


(Naive) memory T [2] TN [2] TN
Matrix multiplication ✓ ✓


We note that many other matrix decompositions are possible (for example, see Appendix B for a compendium of algorithms

for 1-SS multiplication through different structured matrix decompositions) which may lead to more algorithms for SSDs

that could be better for other specialized settings. Even more broadly, we note that semiseparable matrices have a rich

literature and many more representations besides the SSS form that we use (Definition 3.2), and even more efficient

algorithms may be possible.

### **7 The Mamba-2 Architecture**


By connecting SSMs and attention, the SSD framework allows us to develop a shared vocabulary and library of techniques

for both. In this section we discuss some examples of understanding and modifying SSD layers using ideas originally


22


|X<br>Y<br>SSM<br>A XB C<br>! !<br>Conv|Col2|
|---|---|
|X<br>B C<br>~~X~~<br>!<br>!<br>**Conv**<br>**SSM**<br>A<br>Y||
|||


**Sequential Mamba Block**


|N<br>X<br>Y<br>SSM<br>A XB C<br>! !<br>Conv|Col2|
|---|---|
|B C<br>X<br>~~X~~<br>!<br>**Conv**<br>**SSM**<br>A<br>**N**<br>Y<br>!||
|||



**Parallel Mamba Block**





Linear projection


Sequence transformation

Nonlinearity (activation,
normalization, multiplication)



Figure 6: ( **Mamba-2** **Architecture** .) The Mamba-2 block simplifies the Mamba block by removing sequential linear

projections; the SSM parameters _𝐴, 𝐵,𝐶_ are produced at the beginning of the block instead of as a function of the SSM input

_𝑋_ . An additional normalization layer is added as in NormFormer (Shleifer, Weston, and Ott 2021), improving stability.

The _𝐵_ and _𝐶_ projections only have a single head shared across the _𝑋_ heads, analogous to multi-value attention (MVA).


developed for Transformers. We discuss several design choices, resulting in the Mamba-2 architecture. These axes of

variation are ablated in Section 9.4.


**7.1** **Block Design**


We first discuss modifications to the neural network block that are independent of the inner sequence mixing layer (i.e.

outside the core SSD layer).


**Parallel Parameter Projections.** Mamba-1 was motivated by an SSM-centric point of view where the selective SSM
layer is viewed as a map from _𝑋_ ↦→ _𝑌_ . The SSM parameters _𝐴, 𝐵,𝐶_ are viewed as subsidiary and are functions of the SSM
input _𝑋_ . Thus the linear projections defining ( _𝐴, 𝐵,𝐶_ ) occur after the initial linear projection to create _𝑋_ .


In Mamba-2, the SSD layer is viewed as a map from _𝐴,𝑋, 𝐵,𝐶_ ↦→ _𝑌_ . It therefore makes sense to produce _𝐴,𝑋, 𝐵,𝐶_ in parallel

with a single projection at the beginning of the block. Note the analogy to standard attention architectures, where _𝑋, 𝐵,𝐶_

correspond to the _𝑄, 𝐾,𝑉_ projections that are created in parallel.


Note that adopting parallel projections for the _𝐴, 𝐵,𝐶,𝑋_ inputs to the SSM slightly reduces parameters and more impor
tantly is more amenable to tensor parallelism for larger models, by using standard Megatron sharding patterns (Shoeybi

et al. 2019)).


**Extra Normalization.** In preliminary experiments, we found that instabilities were prone to arising in larger models.

We were able to alleviate this by adding an extra normalization layer (e.g. LayerNorm, GroupNorm, or RMSNorm) to the

block right before the final output projection. This usage of a normalization is most directly related to the NormFormer

architecture (Shleifer, Weston, and Ott 2021), which also added normalization layers at the end of the MLP and MHA

blocks.


We also note that this change is similar to other recent models related to Mamba-2 that were derived from a linear attention

viewpoint. The original linear attention formulation normalizes by a denominator term that emulates the normalization

of the softmax function in standard attention. TransNormerLLM (Qin, Dong Li, et al. 2023) and RetNet (Y. Sun et al. 2023)

find that this normalization is unstable and add an extra LayerNorm or GroupNorm after the linear attention layer. Our

extra normalization layer differs slightly from these, occuring after the multiplicative gate branch instead of before.


23


**7.2** **Multihead Patterns for Sequence Transformations**


Recall that SSMs are defined as a sequence transformation (Definition 2.1) where:


   - _𝐴, 𝐵,𝐶_ parameters have a state dimension N.


   - They define a sequence transformation R [T] → R [T], which for example can be represented as a matrix _𝑀_ ∈ R [(][T] _[,]_ [T][)] .


   - This transformation operates over an input sequence _𝑋_ ∈ R [(][T] _[,]_ [P][)], independently over the P axis.


One can view this as defining one _head_ of the sequence transformation.


**Definition 7.1** (Multihead patterns) **.** _A multihead sequence transformation consists of_ H _independent heads, for a total model_
_dimension of_ D = d ___ model _. The parameters may be tied across heads, leading to a_ _**head pattern**_ _._


The state size N and head dimension P are analogous to the _𝑄𝐾_ head dimension and _𝑉_ head dimension of attention, respec
tively. Just as in modern Transformer architectures (Chowdhery et al. 2023; Touvron, Lavril, et al. 2023), in Mamba-2 we
generally choose these to be constants around 64 or 128; when the model dimension D increases, we increase the number of
heads while keeping the head dimensions N and P fixed. In order to describe how to do this, we can transfer and generalize

ideas from multihead attention to define similar patterns for SSMs, or any general sequence transformation.



**Multi-head SSM**


(Multi-head Attn.)


_𝑋_ (T _,_ H _,_ P)


_𝐴_ (T _,_ H)


_𝐵_ (T _,_ H _,_ N)


_𝐶_ (T _,_ H _,_ N)



(17)



**Multi-contract SSM**


(Multi-query Attn.)


_𝑋_ (T _,_ 1 _,_ P)


_𝐴_ (T _,_ H)


_𝐵_ (T _,_ 1 _,_ N)


_𝐶_ (T _,_ H _,_ N)



(18)



**Multi-expand SSM**


(Multi-key Attn.)


_𝑋_ (T _,_ 1 _,_ P)


_𝐴_ (T _,_ H)


_𝐵_ (T _,_ H _,_ N)


_𝐶_ (T _,_ 1 _,_ N)



(19)



**Multi-input SSM**


(Multi-value Attn.)


_𝑋_ (T _,_ H _,_ P)


_𝐴_ (T _,_ H)


_𝐵_ (T _,_ 1 _,_ N)


_𝐶_ (T _,_ 1 _,_ N)



(20)



**Multihead** **SSM** **(MHS)** **/** **Multihead** **Attention** **(MHA)** **Pattern.** The classic MHA pattern assumes that the head
dimension P divides the model dimension D. The number of heads is defined as H = D/P. Then, H copies of the core sequence
transformation are created by creating H independent copies of each parameter. Note that while the MHA pattern was

first described for the attention sequence transformation, it can be applied to anything compatible with Definition 2.1. For

example, a multi-head SSD layer would accept inputs with shapes according to equation (17) where the SSD algorithm is
broadcasted over the H = n_heads dimension.


**Multi-contract SSM (MCS) / Multi-query Attention (MQA) Pattern.** Multi-query attention (Shazeer 2019) is a clever

optimization for attention that can dramatically improve the speed of autoregressive inference, which relies on caching

the _𝐾_ and _𝑉_ tensors. This technique simply avoids giving _𝐾_ and _𝑉_ the extra head dimension, or in other words broadcasts

a single head of ( _𝐾,𝑉_ ) across all the heads of _𝑄_ .


Using the state space duality, we can define an equivalent SSM version of MQA as equation (18). Here, _𝑋_ and _𝐵_ (the SSM
analogs of attention’s _𝑉_ and _𝐾_ ) are shared across the H heads. We also call this the _multi-contract SSM (MCS)_ head pattern,

because the _𝐶_ parameter which controls the SSM state contraction has independent copies per head.


We can similarly define a multi-key attention (MKA) or _multi-expand SSM (MES)_ head pattern, where _𝐵_ (which controls

the SSM expansion) is independent per head while _𝐶_ and _𝑋_ are shared across heads.


**Multi-input SSM (MIS) / Multi-value Attention (MVA) Pattern.** While MQA makes sense for attention because of

its KV cache, it is not the natural choice for SSMs. In Mamba, instead, _𝑋_ is viewed as the main input to the SSM, and

therefore _𝐵_ and _𝐶_ are parameters that are shared across the input channels. We define a new multi-value attention (MVA)

of _multi-input_ _SSM_ _(MIS)_ pattern in equation (20), which can again be applied to any sequence transformation such as

SSD.


Armed with this vocabulary, we can characterize the original Mamba architecture more precisely.


**Proposition 7.2.** _The selective SSM (S6) layer of the Mamba architecture (Gu and Dao 2023) can be viewed as having_


24


   - _Head dimension 𝑃_ = 1 _: every channel has independent SSM dynamics 𝐴._


   - Multi-input SSM _(MIS) or_ multi-value attention _(MVA) head structure: the 𝐵,𝐶_ _matrices (corresponding to 𝐾,𝑄_ _in the_
_attention duality) are shared across all channels of the input 𝑋_ _(corresponding to 𝑉_ _in attention)._


We can also ablate these head pattern variants when applied to SSD (Section 9.4.3). Interestingly, despite being controlled

in parameter counts and total state dimension, there is a noticeable difference in downstream performance. We empirically

find that the MVA pattern as originally used in Mamba performs best.


**Grouped** **Head** **Patterns.** The ideas of multi-query attention can be extended to _grouped-query_ _attention_ (Ainslie et
al. 2023): instead of 1 K and V head, one can create G independent K and V heads, where 1 _<_ G and G divides H. This

is motivated both by bridging the performance difference between multi-query and multi-head attention, and enabling
more efficient tensor parallelism by setting G to be a multiple of the number of shards (Section 8).


Similarly, the multi-input SSM head pattern used in Mamba-2 can be easily extended to **grouped-input** **SSM** **(GIS)**,
or synonymously **grouped-value** **attention** **(GVA)** . The generalization is straightforward and we omit the details for

simplicity.


**7.3** **Other SSD Extensions from Linear Attention**


We describe here an example of architectural modifications to SSD motivated by linear attention. We ablate these in

Section 9.4.3 as a form of negative result, finding that they do not significantly improve performance enough to adopt

them as default settings. Nonetheless, these illustrate how the vast literature on attention can be incorporated to define

variants of SSD. We treat the choice of kernel feature map as a hyperparameter in the Mamba-2 architecture, and expect

other simple modifications inspired by attention to be possible as well.


**Kernel Attention Approximations to Softmax Attention.** Many variants of linear attention or kernel attention are
motivated by viewing the attention scores softmax( _𝑄𝐾_ [⊤] ) as composed of


1. An exponential kernel _𝑍_ = exp( _𝑄𝐾_ [⊤] ), which can be approximated by _𝑍_ = _𝜓_ ( _𝑄_ ) _𝜓_ ( _𝐾_ ) [⊤] for some kernel feature

map.


2. Normalizing the kernel so that rows sum to 1 via _𝑀_ = _𝐺_ / _𝐺_ 11 [⊤], where the division happens elementwise and 1 is

the all 1’s vector.


**Exponential Kernel Feature Maps.** In Mamba-2, we incorporate a flexible kernel feature map, and apply it to the _𝐵_

and _𝐶_ branches (corresponding to the _𝐾_ and _𝑉_ branches in attention). The feature map can also be optionally applied to

the _𝑋_ ( _𝑉_ ) branch, for simplicity and symmetry. This is represented in Figure 6 by an arbitrary nonlinearity. By default,

we simply choose _𝜓_ to be an elementwise Swish / SiLU function (Hendrycks and Gimpel 2016; Ramachandran, Zoph, and

Le 2017). We explore other options in the ablations in Section 9.4.3, including feature maps used by Linear Attention,

Performer, Random Feature Attention, and cosFormer (Section 4.1.3).


**Incorporating a Normalization (Denominator) Term.** To find the denominator term, we simply have to compute
_𝑀_ 1. But recall that the final output of the model is just _𝑌_ = _𝑀𝑋_ (equation (16)). So the normalization terms can be found
simply by augmenting _𝑋_ with an extra column 1, resulting in a tensor of shape (T _,_ P + 1).


Note that in this case, the kernel feature map _𝜓_ must be positive so that the sum is positive.

### **8 Systems Optimization for SSMs**


We describe several systems optimizations for SSMs, in particular the Mamba-2 architecture, for large-scale efficient

training and inference. In particular, we focus on tensor parallel and sequence parallel for large-scale training, as a well

variable-length sequences for efficient finetuning and inference.


25


**8.1** **Tensor Parallel**


Tensor parallelism (TP) (Shoeybi et al. 2019) is a model parallelism technique that splits each layer (e.g., attention, MLP)

to run on multiple accelerators such as GPUs. This technique is widely used to train most large models (Brown et al.

2020; Chowdhery et al. 2023; Touvron, Lavril, et al. 2023; Touvron, L. Martin, et al. 2023) on GPU clusters where each

node typically has 4-8 GPUs with fast networking such as NVLink. TP was originally developed for the Transformer

architecture, and it is not straight-forward to adapt it other architecture. We first show the challenge of using TP with the

Mamba architecture, and the show how the Mamba-2 architecture is designed to make TP efficient.


Recall the Mamba architecture, with a single input _𝑢_ ∈ R _[𝐿]_ [×] _[𝑑]_ (no batching for simplicity), input projection matrices
_𝑊_ [(] _[𝑥]_ [)] _,𝑊_ [(] _[𝑧]_ [)] ∈ R _[𝑑]_ [×] _[𝑒𝑑]_ where _𝑒_ is the expansion factor (typically 2), and output projection matrix _𝑊_ [(] _[𝑜]_ [)] ∈ R _[𝑒𝑑]_ [×] _[𝑑]_ :


_𝑥_ = _𝑢𝑊_ [(] _[𝑥]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑒𝑑]_

_𝑧_ = _𝑢𝑊_ [(] _[𝑧]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑒𝑑]_


_𝑥𝑐_ = conv1d( _𝑥_ ) ∈ R _[𝐿]_ [×] _[𝑒𝑑]_ (depthwise, independent along _𝑑_ )


Δ _, 𝐵,𝐶_ = low-rank projection( _𝑥𝑐_ )

_𝑦_ = _𝑆𝑆𝑀𝐴,𝐵,𝐶,_ Δ ( _𝑥𝑐_ ) ∈ R _[𝐿]_ [×] _[𝑒𝑑]_ (independent along _𝑑_ )


_𝑦𝑔_ = _𝑦_                - _𝜙_ ( _𝑧_ ) (gating, e.g., with _𝜙_ being SiLU)

out = _𝑦𝑔𝑊_ [(] _[𝑜]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑑]_ _._



With TP, suppose that we want to split the computation along 2 GPUs. It is easy to split the input projection matrices
_𝑊_ [(] _[𝑥]_ [)] and _𝑊_ [(] _[𝑧]_ [)] into two partitions each of size _𝑑_ × _[𝑒𝑑]_ 2 [.] [Then each GPU would hold half of] _[ 𝑥][𝑐]_ [of size] _[ 𝐿]_ [×] _[𝑒𝑑]_ 2 [.] [However,]

we see that since Δ _, 𝐵,𝐶_ are functions are _𝑥𝑐_, so we would need an extra all-reduce between the GPUs to get the whole
of _𝑥𝑐_ before computing Δ _, 𝐵,𝐶_ . After that the two GPUs can compute the SSM in parallel since they are independent
along _𝑑_ . At the end, we can split the output projection matrices _𝑊_ [(] _[𝑜]_ [)] into two partitions each of size _[𝑒𝑑]_

2 [×] _[ 𝑑]_ [, and do an]

all-reduce at the end. Compared to Transformers, we would incur two all-reduces instead of one, doubling the time spent

in communication. For large-scale Transformers training, communication might already take a significant fraction of time

(e.g. 10-20%), and doubling communication would make Mamba not as efficient for large-scale training.


With Mamba-2, our goal is to have only one all-reduce per block, similar to attention or MLP blocks in Transformers.
As a result, we have the projection to get Δ _, 𝐵,𝐶_ directly from _𝑢_ instead of from _𝑥𝑐_, allowing us to split these projection
matrices. This implies that we have different sets of Δ _, 𝐵,𝐶_ on different GPUs, which is equivalent to having several
“groups” of Δ _, 𝐵,𝐶_ on a larger “logical GPU”. Moreover, we use GroupNorm within each block, with number of groups

divisible by the TP degree, so that the GPUs in a TP group do not have a communicate within the block:


_𝑥_ = _𝑢𝑊_ [(] _[𝑥]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑒𝑑]_

_𝑧_ = _𝑢𝑊_ [(] _[𝑧]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑒𝑑]_


Δ _, 𝐵,𝐶_ = projection( _𝑢_ ) (one or more groups of Δ _, 𝐵,𝐶_ per GPU)


_𝑥𝑐_ = conv1d( _𝑥_ ) ∈ R _[𝐿]_ [×] _[𝑒𝑑]_ (depthwise, independent along _𝑑_ )

_𝑦_ = _𝑆𝑆𝑀𝐴,𝐵,𝐶,_ Δ ( _𝑥𝑐_ ) ∈ R _[𝐿]_ [×] _[𝑒𝑑]_ (independent along _𝑑_ )


_𝑦𝑔_ = _𝑦_            - _𝜙_ ( _𝑧_ ) (gating, e.g., with _𝜙_ being SiLU)


_𝑦𝑛_ = groupnorm( _𝑦𝑔_ ) (number of groups divisible by degree of tensor parallel)

out = _𝑦𝑔𝑊_ [(] _[𝑜]_ [)⊤] ∈ R _[𝐿]_ [×] _[𝑑]_ _._


We see that we only need to split the input projection matrices, and the output projection matrices, and only need to do

all-reduce at the end of the block. This is similar to the design of TP for attention and MLP layers. In particular, if we
have TP degree 2, we would split _𝑊_ [(] _[𝑥]_ [)] = [ _𝑊_ 1 [(] _[𝑥]_ [)] _,𝑊_ 2 [(] _[𝑥]_ [)] ] with _𝑊𝑖_ [(] _[𝑥]_ [)] ∈ R _[𝑑]_ [×] _[𝑒𝑑]_ [/][2], _𝑊_ [(] _[𝑧]_ [)] = [ _𝑊_ 1 [(] _[𝑧]_ [)] _,𝑊_ 2 [(] _[𝑧]_ [)] ] with _𝑊𝑖_ [(] _[𝑧]_ [)] ∈ R _[𝑑]_ [×] _[𝑒𝑑]_ [/][2],


26


**Outputs** "


**States** #


**Inputs** !















|Col1|Col2|Col3|Col4|All-reduce|
|---|---|---|---|---|
|X<br>A<br>**G**<br>Y<br>&<br>&|X<br>A<br>**G**<br>Y<br>&<br>&|X<br>A<br>**G**<br>Y<br>&<br>&|B C<br>**N**<br>!<br>(')|B C<br>**N**<br>!<br>(')|
|X<br>A<br>**G**<br>Y<br>&<br>&|||||
|X<br>A<br>**G**<br>Y<br>&<br>&||X|X|X|
|X<br>A<br>**G**<br>Y<br>&<br>&||X|!<br>(#)<br>&!<br>(%)|!<br>(#)<br>&!<br>(%)|
||||||


|&<br>G<br>Y<br>A X<br>&|Col2|Col3|(')<br>&<br>N<br>B C|
|---|---|---|---|
|X<br>A<br>**G**<br>Y<br>&<br>&||||
|X<br>A<br>**G**<br>Y<br>&<br>&||X|X|
|X<br>A<br>**G**<br>Y<br>&<br>&||X|&<br>(#)<br>&&<br>(%)|
|X<br>A<br>**G**<br>Y<br>&<br>&||X||


Figure 7: ( **Parallelism with the Mamba-2 Block** .) ( _Left_ : **Tensor Parallelism** ) We split the input projection matrices
_𝑊_ [(] _[𝑥]_ [)] _,𝑊_ [(] _[𝑧]_ [)] and the output projection matrix _𝑊_ [(] _[𝑜]_ [)] . Each SSM head ( _𝐴, 𝐵,𝐶,𝑋_ ) ↦→ _𝑌_ lives on a single device. Choosing

GroupNorm for the final normalization layer avoids extra communication. We need one all-reduce per layer, just like the

MLP or attention blocks in a Transformer. ( _Right_ : **Sequence/Context** **Parallelism** ) Analogous to the SSD algorithm,

with multiple devices, we can split along the sequence dimension. Each device computes the state of its sequence, then

pass that state to the next GPU.







with _𝑊𝑖_ [(] _[𝑜]_ [)] ∈ R _[𝑒𝑑]_ [/][2][×] _[𝑑]_ . For _𝑖_ = 1 _,_ 2, the TP Mamba-2 layer can be written as:


⊤ _𝐿_ × _𝑒𝑑_ /2
_𝑥_ [(] _[𝑖]_ [)] = _𝑢𝑊𝑖_ [(] _[𝑥]_ [)] ∈ R

⊤ _𝐿_ × _𝑒𝑑_ /2
_𝑧_ [(] _[𝑖]_ [)] = _𝑢𝑊𝑖_ [(] _[𝑧]_ [)] ∈ R



and _𝑊_ [(] _[𝑜]_ [)] =




_𝑊_ [(] _[𝑜]_ [)]
1
_𝑊_ [(] _[𝑜]_ [)]
2



Δ [(] _[𝑖]_ [)] _, 𝐵_ [(] _[𝑖]_ [)] _,𝐶_ [(] _[𝑖]_ [)] = projection( _𝑢_ ) (one or more groups of Δ _, 𝐵,𝐶_ per GPU)

_𝑥𝑐_ [(] _[𝑖]_ [)] = conv1d( _𝑥_ [(] _[𝑖]_ [)] ) ∈ R _[𝐿]_ [×] _[𝑒𝑑]_ [/][2]

_𝑦_ [(] _[𝑖]_ [)] = _𝑆𝑆𝑀𝐴,𝐵,𝐶,_ Δ ( _𝑥𝑐_ [(] _[𝑖]_ [)] [)] [∈] [R] _[𝐿]_ [×] _[𝑒𝑑]_ [/][2]

_𝑦𝑔_ [(] _[𝑖]_ [)] = _𝑦_ [(] _[𝑖]_ [)]            - _𝜙_ ( _𝑧_ [(] _[𝑖]_ [)] )

_𝑦𝑛_ [(] _[𝑖]_ [)] = groupnorm( _𝑦𝑔_ [(] _[𝑖]_ [)] [)] (number of groups divisible by degree of tensor parallel)

⊤ _𝐿_ × _𝑑_ /2
out [(] _[𝑖]_ [)] = _𝑦𝑔_ [(] _[𝑖]_ [)] _[𝑊]_ _𝑖_ [(] _[𝑜]_ [)] ∈ R


∑︁
out = out [(] _[𝑖]_ [)] _._ (summing outputs from all GPUs with an all-reduce)


_𝑖_


We illustrate tensor parallel with Mamba-2 in Figure 7 ( _Left_ ).


**8.2** **Sequence Parallelism**


For very long sequences, we might need to split the input and activation to different GPUs along the sequence length

dimension. There are two main techniques:


1. Sequence parallelism (SP) for the residual and normalization operations: first proposed by Korthikanti et al. (2023),

this technique decomposes the all-reduce in TP as reduce-scatter and all-gather. Noticing that the residual and

normalization operations are repeated on the same input for all GPUs in the same TP group, SP splits the activations

along the sequence length dimension by performing: reduce-scatter, residual and normalization, then all-gather.


Since the Mamba-2 architecture uses the same residual and normalization structure, SP applies without modification.


2. Sequence parallelism for the token-mixing operations (attention or SSM), also known as “context parallelism” (CP).

Several techniques have been developed for attention layer (e.g., Ring attention (Liu, Yan, et al. 2024; Liu, Zaharia,


27


Sequence Length: 512


32 64 128 256
Model dimension



Sequence Length: 1024


32 64 128 256
Model dimension



1.00


0.75


0.50


0.25


0.00



Sequence Length: 256


32 64 128 256
Model dimension



Attention
Based
Mamba (N=16)
Mamba-2 (N=16)
Mamba-2 (N=64)
Mamba-2 (N=256)



Figure 8: ( **Multi-Query** **Associative** **Recall** **(MQAR)** ). Associative recall tasks are challenging for SSMs, which must

memorize all relevant information into their recurrent state. The SSD layer combined with improved architecture allows

for much larger state sizes in Mamba-2, which performs significantly better than Mamba-1 and even vanilla attention.


and Abbeel 2023)), with sophisticated load-balancing technique (Brandon et al. 2023). The difficulty with sequence

parallelism in attention is that we can split queries and keys into block, but each query block needs to interact with

key blocks, leading to communication bandwidth quadratic in the number of workers.


With SSMs, we can split the sequence in a simple manner: each worker takes an initial state, compute the SSM

with respect to their inputs, return the final state, and pass that final state to the next worker. The communication

bandwidth is linear in the number of workers. This decomposition is exactly the same as the block-decomposition

in the SSD algorithm (Figure 5) to split into blocks / chunks. We illustrate this context parallelism in Figure 7 ( _Right_ ).


**8.3** **Variable Length**


While pretraining often uses the same sequence lengths for the batch, during finetuning or inference, the model might

need to process different input sequences of different lengths. One naive way to handle this case is to right-pad all

sequences in the batch to the maximum length, but this can be inefficient if sequences are wildly different lengths. For

transformers, sophisticated techniques have been develop to avoid padding and do load-balancing between GPUs (Zeng

et al. 2022; Y. Zhai et al. 2023), or packing multiple sequences in the same batch and adjust the attention mask (Ding

et al. 2024; Pouransari et al. 2024). With SSMs and Mamba in particular, we can handle variable sequence lengths by

simply treating the whole batch as one long sequence, and avoid passing the states between individual sequences. This is

equivalent to simply setting _𝐴𝑡_ = 0 for tokens _𝑡_ at the end of one sequence to prevent it from passing information to the
token _𝑡_ + 1, which belongs to a different sequence.

### **9 Empirical Validation**


We empirically evaluate Mamba-2 on synthetic recall tasks that have been challenging for recurrent models (Section 9.1),

and standard language modeling pre-training and downstream evaluations (Section 9.2). We validate that our SSD algo
rithm is much more efficient than Mamba-1 (Section 9.3) and comparable to optimized attention for moderate sequence

lengths. Finally, we ablate various design choices in the Mamba-2 architecture (Section 9.4).


**9.1** **Synthetics: Associative Recall**


Synthetic associative recall tasks have been popular for testing the ability of language models to look up information in

their context. Broadly, they involve feeding autoregressive models pairs of key-value associations, and then prompting the

model to produce the correct completion upon being shown a previously-seen key. The **multi-query associative recall**
**(MQAR)** task is a particular formulation of this task that requires the model to memorize multiple associations (Arora,

Eyuboglu, Timalsina, et al. 2024). The original Mamba paper reported results on related synthetic tasks, in particular

Selective Copying (Gu and Dao 2023) and Induction Heads (Olsson et al. 2022), which can be seen as easier associative

recall tasks. The MQAR task is also closely related to “phonebook look-up” tasks which has been shown to be challenging

for recurrent models such as SSMs, due to their finite state capacity (De et al. 2024; Jelassi et al. 2024).


28


10 [1]


9 �10 [0]


8 �10 [0]


7 �10 [0]


6 �10 [0]



Scaling Laws on The Pile (Sequence Length 8192)

|Transformer++<br>Mamba<br>Mamba-2|Col2|Col3|
|---|---|---|
||||



FLOPs (log scale)



Figure 9: ( **Scaling Laws** .) Models of size ≈ 125 _𝑀_ to ≈ 1 _._ 3 _𝐵_ parameters, trained on the Pile. Mamba-2 matches or exceeds

the performance of Mamba as well as a strong “Transformer++” recipe. Compared to our Transformer baseline, Mamba-2

is Pareto dominant on performance (perplexity), theoretical FLOPs, and actual wall-clock time.


Table 1: ( **Zero-shot Evaluations** .) Best results for each size in bold, second best unlined. We compare against open source LMs with

various tokenizers, trained for up to 300B tokens. Pile refers to the validation split, comparing only against models trained on the same

dataset and tokenizer (GPT-NeoX-20B). For each model size, Mamba-2 outperforms Mamba, and generally matches Pythia at twice the

model size. Full results in Table 10.


Model Token. Pile LAMBADA LAMBADA HellaSwag PIQA Arc-E Arc-C WinoGrande OpenbookQA Average

ppl ↓ ppl ↓ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑ acc ↑


Pythia-1B NeoX 7 _._ 82 7 _._ 92 56 _._ 1 47 _._ 2 70 _._ 7 57 _._ 0 27 _._ 1 53 _._ 5 31 _._ 4 49 _._ 0

Mamba-790M NeoX 7 _._ 33 6 _._ 02 62 _._ 7 55 _._ 1 72 _._ 1 61 _._ 2 29 _._ 5 56 _._ 1 34 _._ 2 53 _._ 0
**Mamba-2-780M** NeoX 7 _._ 26 5 _._ 86 61 _._ 7 54 _._ 9 72 _._ 0 61 _._ 0 28 _._ 5 60 _._ 2 36 _._ 2 53 _._ 5


Hybrid H3-1.3B GPT2 - 11 _._ 25 49 _._ 6 52 _._ 6 71 _._ 3 59 _._ 2 28 _._ 1 56 _._ 9 34 _._ 4 50 _._ 3

Pythia-1.4B NeoX 7 _._ 51 6 _._ 08 61 _._ 7 52 _._ 1 71 _._ 0 60 _._ 5 28 _._ 5 57 _._ 2 30 _._ 8 51 _._ 7

RWKV4-1.5B NeoX 7 _._ 70 7 _._ 04 56 _._ 4 52 _._ 5 72 _._ 4 60 _._ 5 29 _._ 4 54 _._ 6 34 _._ 0 51 _._ 4

Mamba-1.4B NeoX 6 _._ 80 5 _._ 04 65 _._ 0 59 _._ 1 74 _._ 2 65 _._ 5 32.8 61 _._ 5 36.4 56 _._ 4
**Mamba-2-1.3B** NeoX 6 _._ 66 5 _._ 02 65 _._ 7 59 _._ 9 73 _._ 2 64 _._ 3 33 _._ 3 60 _._ 9 37 _._ 8 56 _._ 4


Hybrid H3-2.7B GPT2 - 7 _._ 92 55 _._ 7 59 _._ 7 73 _._ 3 65 _._ 6 32 _._ 3 61 _._ 4 33 _._ 6 54 _._ 5

Pythia-2.8B NeoX 6 _._ 73 5 _._ 04 64 _._ 7 59 _._ 3 74 _._ 0 64 _._ 1 32 _._ 9 59 _._ 7 35 _._ 2 55 _._ 7

RWKV4-3B NeoX 7 _._ 00 5 _._ 24 63 _._ 9 59 _._ 6 73 _._ 7 67 _._ 8 33 _._ 1 59 _._ 6 37 _._ 0 56 _._ 4

Mamba-2.8B NeoX 6 _._ 22 4 _._ 23 69 _._ 2 66 _._ 1 75 _._ 2 69 _._ 7 36 _._ 3 63 _._ 5 39 _._ 6 59 _._ 9
**Mamba-2-2.7B** NeoX 6 _._ 09 4 _._ 10 69 _._ 7 66 _._ 6 76 _._ 4 69 _._ 6 36 _._ 4 64 _._ 0 38 _._ 8 60 _._ 2



SSD vs Scan time (A100 80GB PCIe)


48 16 32 64 128 256
State dim



1000


100


10


1


0.1



SSD, Scan, Convolution vs Attention time (A100 80GB PCIe)


512 1k 2k 4k 8k 16k 32k 64k 128k 256k 512k
Sequence length



7


6


5


4


3


2


1


0



Figure 10: ( **Efficiency Benchmarks** .) ( _Left_ ) Our SSD is 2 − 8× faster than a Mamba fused scan for large state expansion
( _𝑁_ = 64) and faster than FlashAttention-2 for sequence length 2k and above. ( _Right_ ) Sequence length 4K: Increasing state

expansion slows down the Mamba optimized scan implementation linearly. SSD can handle much larger state expansion

factors without much slowdown.


29


We compare on a challenging version of the MQAR setup from (Arora, Eyuboglu, Zhang, et al. 2024), using a harder task,

longer sequences, and smaller models. Our baselines include standard multi-head softmax attention as well as the Based

architecture which combines convolutions, local attention, and a linear attention variant.


Results are shown in Figure 8. While Mamba-1 struggles on this task, Mamba-2 performs well across all settings. Surprisingly, it is significantly better than Mamba-1 even when the state sizes are controlled (N = 16). (We are not sure which

aspect of the architecture is the predominant factor, which remains a question to explore in future work.) Additionally,
this task validates the importance of state size: increasing from N = 16 to N = 64 and N = 256 consistently improves

performance on MQAR, as the larger state allows more information (key-value pairs) to be memorized.


**9.2** **Language Modeling**


Following standard protocols in LLMs, we train and evaluate the Mamba-2 architecture on standard autoregressive lan
guage modeling against other architectures. We compare both pretraining metrics (perplexity) and zero-shot evaluations.

The model sizes (depth and width) follow GPT3 specifications, from 125m to 2.7B. We use the Pile dataset (L. Gao, Bider
man, et al. 2020), and follow the training recipe described in Brown et al. (2020). This follows the same setup as reported

in Mamba (Gu and Dao 2023); training details are in Appendix D.


**9.2.1** **Scaling Laws**


For baselines, we compare against both Mamba and its Transformer++ recipe (Gu and Dao 2023), which is based on the

PaLM and LLaMa architectures (e.g. rotary embedding, SwiGLU MLP, RMSNorm instead of LayerNorm, no linear bias, and

higher learning rates). As Mamba has already demonstrated that it outperforms the standard Transformer architecture

(GPT3 architecture) as well as recent subquadratic architectures (H3 (Dao, D. Y. Fu, et al. 2023), Hyena (Poli et al. 2023),

RWKV-4 (B. Peng, Alcaide, et al. 2023), RetNet (Y. Sun et al. 2023)), we omit those in the plot for clarity (see Gu and Dao

(2023) for comparisons).


Figure 9 shows scaling laws under the standard Chinchilla (Hoffmann et al. 2022) protocol, on models from ≈ 125 _𝑀_ to
≈ 1 _._ 3 _𝐵_ parameters.


**9.2.2** **Downstream Evaluations**


Table 1 shows the performance of Mamba-2 on a range of popular downstream zero-shot evaluation tasks, compared

to the most well-known open source models at these sizes, most importantly Pythia (Biderman et al. 2023) which were

trained with the same tokenizer, dataset, and training length (300B tokens) as our models.


**9.2.3** **Hybrid Models: Combining SSD Layer with MLP and Attention**


Recent and concurrent work (Dao, D. Y. Fu, et al. 2023; De et al. 2024; Glorioso et al. 2024; Lieber et al. 2024) suggests that a

hybrid architecture with both SSM layers and attention layers could improve the model quality over that of a Transformer,

or a pure SSM (e.g., Mamba) model, especially for in-context learning. We explore the different ways that SSD layers can

be combined with attention and MLP to understand the benefits of each. Empirically we find that having around 10% of

the total number of layers being attention performs best. Combining SSD layers, attention layers, and MLP also works

better than either pure Transformer++ or Mamba-2.


**SSD** **and** **Attention** We find that SSD and attention layers are complementary: by themselves (e.g. in the Mamba-2

architecture vs. Transformer++) their performance (measured by perplexity) is nearly the same, but a mixture of SSD

and attention layers outperforms the pure Mamba-2 or Transformer++ architecture. We show some results (Table 2)

for the 350M model (48 layers) trained to 7B tokens on the Pile with the GPT-2 tokenizer (same number of parameters,

same hyperparameters, same training and validation set). Adding in just a few attention layers already yields notable

improvement and strikes the best balance between quality and efficiency. We hypothesize that the SSM layers function

well as a general sequence-to-sequence mapping, and attention layers act as a retrieval mechanism to quickly refer to

previous tokens in the sequence instead of forcing the model to compress all the context to its memory (SSM states).


30


