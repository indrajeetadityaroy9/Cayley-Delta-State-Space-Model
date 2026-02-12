## **Attention when you need**

**Lokesh Boominathan**
Neuroscience Institute
Carnegie Mellon University, Pittsburgh, PA 15213
```
              lboomina@andrew.cmu.edu

```

**Yizhou Chen**
Neuroscience Institute
Carnegie Mellon University, Pittsburgh, PA 15213
```
              yizhouc3@andrew.cmu.edu

```

**Matthew McGinley**
Neuroscience
Baylor College of Medicine, Houston, TX 77030
```
                mmcginle@bcm.edu

```

**Xaq Pitkow**
Neuroscience Institute and Department of Machine Learning
Carnegie Mellon University, Pittsburgh, PA 15213
```
                 xaq@cmu.edu

```

**Abstract**


Being attentive to task-relevant features can improve task performance, but paying
attention comes with its own metabolic cost. Therefore, strategic allocation of
attention is crucial in performing the task efficiently. This work aims to understand
this strategy. Recently, de Gee _et al._ [1] conducted experiments involving mice
performing an auditory sustained attention-value task. This task required the mice
to exert attention to identify whether a high-order acoustic feature was present
amid the noise. By varying the trial duration and reward magnitude, the task
allows us to investigate how an agent should strategically deploy their attention
to maximize their benefits and minimize their costs. In our work, we develop a
reinforcement learning-based normative model of the mice to understand how it
balances attention cost against its benefits. The model is such that at each moment
the mice can choose between two levels of attention and decide when to take
costly actions that could obtain rewards. Our model suggests that efficient use of
attentional resources involves alternating blocks of high attention with blocks of
low attention. In the extreme case where the agent disregards sensory input during
low attention states, we see that high attention is used rhythmically. Our model
provides evidence about how one should deploy attention as a function of task
utility, signal statistics, and how attention affects sensory evidence.


**1** **Introduction**


It is well known that paying attention to relevant features in a task aids in achieving better task
performance [2–4]. However, attention, like other brain computations, may come at its own costs. For
example, studies demonstrate that humans find tasks requiring sustained attention over long periods


Preprint. Under review.


to be fatiguing, leading to performance variations and an overall decline as time on task increases

[5–9]. Recently, de Gee _et al._ [1] conducted a study on mice performing an attention-driven auditory
feature-based detection task. In order to succeed in the task, the mice need to pay high levels of
attention to detect the emergence of coherent time-frequency motion in random cloud of tones. It is
computationally expensive to filter sound from a particular source [10], so by varying the amount of
sugar water reward this experimental setup provides an opportunity to explore the trade-off between
attention cost and its benefits.


In our work, we build a simple normative model of the mouse, where at every time instant in the trial,
the mouse can choose to pay low or high attention and to lick or not. The normative model’s objective
is to maximize task performance while minimizing attentional costs. We use reinforcement learning
(RL) to optimize for this objective, and study the patterns of how attention could be economically
deployed across a session. For example, we see that in order to use energy resources frugally, blocks
of high attention need to be interspersed between blocks of low attention. Interestingly, we see
that when an inattentive state completely ignores sensory information, the optimal strategy is to
deploy high attention rhythmically. Similar such rhythmic patterns have been previously observed
in literature for visual attention tasks [11–14], suggesting that our framework could potentially
provide a normative rationale for this strategic behavior. Based on our framework, we predict how
neural/physiological correlates of attention change as we vary different parameters, including the task
utility, the signal duration in the trial, and how attention affects sensory reliability.


**2** **Related Work**


**Studies on attention.** There are several dimensions of attention studied in the neuroscience community [15, 16]. In our work, we focus on one such class of attention known as sustained attention,
also known as vigilance, which is the subject’s preparedness to detect infrequent and unpredictable
signals over extended periods [17]. However, despite the term "sustained" in the terminology, there is
evidence in literature suggesting that attention fluctuates between optimal and sub-optimal states over
time, due to factors such as arousal, mind wandering, and competing behavioral demands.[18, 19, 12].
Our work explores one such factor, frugal use of cognitively demanding attentional resource, especially how it shapes the deployment of heightened attention within trials, during a session with
varying utility of attention.


Recently, de Gee _et_ _al._ [1] conducted a study on a large cohort of mice performing an auditory
feature-based sustained-attention task. The study investigated how the mice modulated its pupillinked arousal for different values of task utility. They found the pre-trial pupil sizes for high task
utility trials to be close to the optimal mid-level size. Their analysis also includes fitting the data using
drift-diffusion models to understand performance changes with task utility. In our work, we study the
same task structure but we take an alternate route of building a normative model to understand how
attention might be modulated during the trial.


**Modeling approaches.** Balkenius _et al._ [10] posited that attention should be viewed as an action
rather than solely a sensory process. Several works have adopted this perspective and built sophisticated optimal control models with attention as an action choice, especially using RL, to determine
where [20–22] and how often [23] to attend. The most relevant among them is [24], which examined
the trade-off between the benefits of high-quality observations in decision-making and the attentional
costs involved. They developed RL agents to solve an abstract model of a sustained attention task
with a fixed trial duration and stochastic signal duration. In their study, they investigated how the
agents’ strategies evolved with changes in task parameters, focusing on the average trends across
trials for agents’ beliefs and attention choices. In our work, we address a different task setup where
the signal duration is fixed, but the trial duration is variable and potentially long. Our analysis focuses
on how attention is distributed within each trial, leading to new insights as shown in Section 5.


**3** **Background**


Recently, de Gee _et al._ [1] conducted a study on a large cohort of mice (88 mice, 1983 sessions)
performing an auditory feature-based sustained-attention task with intermittently shifting reward
magnitude. To succeed in the sustained-attention task, the mice need to pay attention to identify
whether there is a signal amid noise. However, analyzing a sound source for long periods of time


2


A Noise Signal Inter-trial-interval (ITI)



.5


0





B





Hit
-> reward





Freq. 96
(kHz)1.5



ITI Noise Signal ITI Noise Signal ITI Noise ITI



1 s



Noise



Signal



12 μl
2 μl





Utility



2


1



Hits
FAs



RTs



10 min.



6 4 2 5 3 1
Low blocks     High blocks



Figure 1: ( **A** ) Experimental setup figure modified with permission, from de Gee _et al._ [1]. Mice are
subjected to noise and signal phases with inter-trial intervals (ITIs). Hits during the signal phase
result in rewards, while false alarms during the noise phase lead to time-outs. The blocks alternate
between low (2 _µ_ l) and high (12 _µ_ l) sugar water rewards. ( **B** ) Behavioral data showing an increase in
the fraction of hits and false alarms (FAs) and a decrease in reaction times (RTs) as the subjective
value of the food reward increases across blocks.


is behaviorally taxing and hence might not be affordable to sustain across a session. Since this
experiment was done with predictable variations in reward size, a computational model to solve for
this task could help us understand how the mice might be balancing the attentional cost associated
with every instance it pays high attention to the task, against the benefits of the resulting high quality
observations in achieving the trial’s task utility.


Below, we describe the experimental approach used in [1], and the mice’s performance trends in the
experiment that we aim to emulate using our model.


**3.1** **Experimental approach**


The study was done with 88 mice performing a total of 1983 sessions. As shown in Fig 1A, each
experimental session includes multiples blocks, and each block includes multiple trials. The details
of a single trial, block, and session are as follows.


_Single trial._ Each trial follows a structured sequence. Initially, a variable duration of noise, depicted
by a random cloud of tones, is introduced. This duration is randomly selected from an exponential
distribution with a mean of 5 seconds. If the mouse licks during this noise phase, the trial promptly
concludes, imposing a 14-second time-out penalty on the mouse. However, should the mouse abstain
from licking throughout this noise period, a distinct signal phase ensues. This signal phase persists for
a fixed duration of 3 seconds, presenting coherent time-frequency motion (pitch sweeps) embedded
within the random tonal array. Should the mouse lick during this signal phase, it is rewarded with
sugar water. The trial concludes either when the mouse licks at any phase or after the signal phase
elapses before the mouse licks. A trial is counted as a miss if the mouse does not lick during the trial.
If the mouse licks in the noise/signal phase, then it is considered as a false alarm (FA)/hit trial.


A _single_ _block_ is a set of 60 trials with inter-trial intervals featuring pink noise, which is highly
discernible from the intra-trial sound stimuli. The quantity of sugar water administered as reward
remains fixed across the 60 trials within each block. Each _single session_ comprises seven blocks with
rewards alternating between low (2 _µL_ ) and high (12 _µL_ ) quantities of 10% sucrose solution (sugar
water), beginning with low. As in [1], we focus our analysis on the last six blocks as the mice spent
the first block in each session gradually engaging in the task.


**3.2** **Mice’s behavioral trends**


The authors hypothesized that high sugar water reward blocks are interpreted by the animal as having
a higher value for expending attention than low sugar water reward blocks. In addition, even when
blocks have the same objective sugar water reward, the subjective value of food reward diminishes
due to fatigue or satiety as the session progresses. According to these notions, the subjective rewards


3


world
states









actions


beliefs















Figure 2: Formal task structure. ( **A** ) In this POMDP, the agent constructs beliefs _bt_ that reflect
inferences about the latent world state using observations _ot_ controlled by the ‘attend’ action _at_ . The
agent uses these beliefs to plan ‘lick’ actions that may obtain rewards. ( **B** ) Specific world states and
transitions in the finite state machine modeling this task. We choose _p_ = 0 _._ 024 so signal begins
after an exponentially distributed time interval with mean of 5 seconds. The signal lasts throughout
a sequence of _N_ = 25 states incremented each time step, lasting a total of 3 seconds. Lick actions
produce rewards when taken during the signal period, and otherwise lead to a false alarm and penalty.


in each block should increase in the following order: 6, 4, 2, 5, 3, 1 (see Fig 1A). On comparing
mouse behavior in this order, the authors showed that as the subjective food reward increases, the hits
and false alarms also increase. Furthermore, the time from signal onset to the time of licking in hit
trials, called reaction time (RT), also significantly decreased with increase in subjective reward. This
is shown in Fig 1B. In the rest of the paper, the term _food reward_ refers to the subjective food reward.


**4** **Method**


Our goal is to create an agent model of a mouse that emulates the behavioral trends described in
Section 3.2. We then use this model to investigate how attention is economically deployed. We
make the following approximations of the actual experiment to keep the problem tractable. We
approximate the continuous-time experiment with discrete time steps with a temporal resolution
of 120 milliseconds. This resolution is assumed to be fine enough to make qualitative predictions
for physiological/neural attention correlates. We assume the mouse’s decision-making is based on
optimizing its objective for a single trial as compared to entire block/session. For simplicity, we
assume that at each time step, our agent decides whether to lick and whether to pay low or high
attention for the next time step. The level of attention chosen determines the quality of observation in
the following time step. That is, the observation at a time step is sampled stochastically depending
on if the trial is in the signal or noise phase, with the degree of stochasticity being influenced by the
selected level of attention. Higher attention is considered computationally expensive, but generates
less stochastic observations. The observations are then used to inform its decision at the next time
step, and so on until the end of the trial. The overall objective of the agent is to succeed in the task
while minimizing the cost of attention allocation. Based on these assumptions, we formulate the
experiment as an agent solving a partially observable Markov decision process (POMDP; Fig 2A)

[25]. Below, we formally describe the elements of the PODMP.


_States._ The latent world dynamics follow a finite state machine in discrete time (Fig 2B), with one
probabilistic transition but mostly deterministic transitions. Our baseline model includes a single
_start_ state, one _noise_ state (indexed as state-0), 25 _signal_ states (indexed from state-1 to state-25 that
serve as a clock), and one _terminal_ state. A trial begins at the fully observable start state, which then
transitions deterministically to the partially observable noise state in the next time step. The fully
observable start state informs the agent that the trial is beginning, analogous to how the end of the
easily recognizable pink noise during the inter-trial interval informs the mice in the experiment that a
new trial is starting. Once in the noise state, if the agent chose to lick, then the agent transitions to
the terminal state signifying the premature end of trial. However, if the agent chose not to lick, it
either remains in the noise state with a probability of 0.976 or transitions to the signal state-1 with a
probability of 0.024 at the next time step. This setup approximates the signal onset time, which in
the actual experiment is drawn from an exponential distribution with a mean of 5 seconds. In the
experiment, the signal then persists for exactly 3 seconds, which we approximate in discrete time as a
sequence of deterministic transitions. Once in a signal state, if the agent does not lick, it transitions
deterministically to the next signal state until reaching the terminal state. However, if the agent licks
during any signal state, it transitions deterministically to the terminal state.


4


_Actions._ At each time step, the agent has two actions: to lick or not, and to pay low or high attention
in the next time step. The decision to lick determines the agent’s next state, while the attention choice
affects the quality of observation it receives in the next time step.


_Observations._ We assume the agent knows the start state and terminal state with absolute certainty. For
other states, we assume that observations are binary numbers generated from a Bernoulli distribution,
with probability determined by the attention level chosen in the previous step and the current latent
state. Let _p_ low and _p_ high denote the probabilities of correctly observing the latent state for low and
high attention, respectively, and _p_ high _> p_ low _≥_ 0 _._ 5. In the noise state, observations are drawn from
either Bernoulli(1 _−_ _p_ low) or Bernoulli(1 _−_ _p_ high), depending on the prior attention choice. In the
signal state, observations are drawn from Bernoulli( _p_ low) or Bernoulli( _p_ high) similarly.


_Rewards._ If the agent licks while in the noise state, it incurs a penalty known as a _false alarm (FA)_
_cost_, reflecting the 14-second time-out in the experiment. Each instance the agent chooses to allocate
high attention comes with an associated _attention cost_ . If the agent licks during any signal state, its
reward is calculated as the _food reward_ divided by the time taken until the lick. The _food reward_ here
is the subjective value of sugar water dispensed upon successful completion of the trial. Normalizing
the food reward by the time taken to lick promotes faster reaction times, mirroring observations from
the actual experiment. This normalization conceptually aligns with the mouse’s natural inclination to
acquire sugar water more swiftly. Consequently, the total reward accumulated in a trial comprises the
cumulative attention cost and FA cost, or the normalized food reward, contingent on when the agent
licks.


_Beliefs._ The agent’s decision at any time point depends on what it believes about the current world
state. For instance, if the agent strongly believes that the signal phase has started, it should lick
soon to avoid missing the reward. Conversely, if the agent strongly believes it is in the noise phase,
to avoid the false alarm cost it should wait to lick until it more likely has transitioned out of this
phase. Since the actual states during the noise and signal phases are hidden from the agent, it bases
its choices on a probability distribution over the states, which represents the likelihood of being in
a particular state given all the observations up to that point. This distribution is called the agent’s
_belief_ . Mathematically, this would be a vector with a dimension equaling the total number of states,
each element in the vector holding the probability of being in the corresponding state. As we are
dealing with a POMDP, the belief at any time point can be calculated exactly based on the current
observation, the belief at the previous time step, and the actions chosen at the previous time [26].
Fig 3 shows an example of how the beliefs evolve with observations and chosen actions in a trial.


The agent’s behavior is determined by how it chooses actions based on its beliefs. This mapping from
agent’s beliefs to its actions is known as the policy. Mathematically, the policy is the distribution
from which the agent samples its actions, conditioned on its belief. Solving the POMDP essentially
involves learning the policy that maximizes the average reward acquired over trials. We achieve
this by framing the POMDP as a Markov Decision Process (MDP) in the space of beliefs and then
optimizing the policy using the Proximal Policy Optimization (PPO) algorithm [27], as implemented
in Stable-Baselines3 [28].The reinforcement learning training and simulation analysis were conducted
on a Linux server with 28 CPU cores. The code will soon be available on GitHub. In the following
section, we discuss the results obtained from our trained agents.


**5** **Results**


We analyze our agent to gain insights into the animals’ attention strategies during a trial and how
these strategies vary with task utility. First, we present a single trial simulation to illustrate the agent’s
behavior. We then demonstrate that the trained agent’s behavior aligns with observed experimental
trends. Next, we examine how the agent’s action choices change based on its belief in being in the
signal phase and the task utility. Finally, using our model, we make empirical predictions for when
the next high attention instance would be deployed, conditioned on the agent’s current belief in being
in the signal phase.


**5.1** **Agent’s performance changes in the same direction as mice when food reward increases**


Figure 3A illustrates how the agent interacts with the simulated trial. Throughout the trial, the agent
maintains an internal belief representing the likelihood of being in different states. At the start of the


5


Lick





Figure 3: Illustration of the agent’s interaction with the simulated trial. ( **A** ) The top row indicates if
the trial is in noise (white) or signal phase (black). The second row indicates the agent’s observations,
the hatched patterns indicating those obtained through low attention. The color indicates if the agent
received a 0 (white) or a 1 (black) as the observation. The third row shows the agent’s likelihood
over the hidden states given the history, called the belief vector. Lick action is denoted by the small
orange triangle. ( **B** ) Grey curve shows the progression of the agent’s belief in being in the signal
phase (signal belief) with time. The blue and orange curves show the consequent attention and lick
choice probabilities. White and black dots indicate 0 and 1 observations respectively. During low
attention instances, the change in signal belief is more gradual (slow ramping shown). In contrast,
high attention driven observations lead to steep upward/downward jumps in signal beliefs depending
on if observations favor signal or noise respectively.


trial (immediately following the _start_ state), the agent is completely certain it is in the noise state,
resulting in a probability value of 1 for the noise state in the belief vector (first element). As the trial
progresses, the belief is redistributed at each time step based on new observations, action choices,
and known state dynamics. Belief updates are more prominent during high-attention instances than
during low-attention instances because high-attention observations are more reliable, giving them
greater influence on the belief update. The agent’s actions, such as whether to lick and whether to
pay high attention at the next time step, are determined based on the current belief.


Similar to the experimental data (Figs 1B), our simulations with the trained agent demonstrate that as
the food reward increases (Fig 4A), the probabilities of hit and false alarm (FA) trials also increase.
Additionally, consistent with the experimental data, the agent’s reaction time decreases as the food
reward increases (Fig 4B). In an ablation study presented in Appendix Fig6, we examined the scenario
without any FA cost. Without this cost, the agent is strongly incentivized to lick in almost every trial,
which is inconsistent with the actual data. Therefore, including the FA cost in the model’s objective
function is crucial.


**5.2** **Attention and lick propensities increase with signal belief and food reward**


Now that we have an agent that conforms to the trends observed in the experimental data, we can use
our model to explore how attention operates during a trial in greater detail. As illustrated in Fig 4C,
the average number of high-attention time instances per trial increases as the food reward rises. This
trend occurs because higher food rewards justify the expenditure of additional attention resources to
enhance signal discriminability, thereby aiding in the attainment of those greater rewards.


At each time step, the agent selects its actions based on a policy distribution influenced by its current
belief. To understand the agent’s behavior, we can examine how this action distribution changes with
varying belief it is in the signal phase. This belief is quantified as the sum of probabilities in the
belief vector for signal nodes 1 to 25, referred to as the agent’s _signal belief_ . Fig 3B is an example
demonstration how the signal belief and the different action probabilities change during a trial based
on the agent’s observations.


For this analysis, we conduct multiple trial simulations, logging the agent’s beliefs and the corresponding probabilities of licking and paying attention as dictated by its policy. We group beliefs with
similar signal beliefs together and average the attention and lick probabilities within each group to
determine the average probabilities for each signal belief bin. When plotting these averages for a fixed
food reward value, we observe that as the signal belief increases, the average attention probability also
rises gradually (Fig 4D). This indicates that the agent is less inclined to expend attention on checking


6


A Performance C













1



Performance Attention distribution







0



Reward





low high





B





Lick





F





Reaction time distribution



D

1



Lick prob















0







Figure 4: Trained agent shows ( **A** ) an increase in the fraction of hits and false alarms, and ( **B** ) a
decrease in reaction times, as the food reward increases (similar to experimental data Fig 1B). ( **C** )
The agent pays more attention in a trial as food reward increases. ( **D** ) The agent’s policy distribution
is plotted as a function of the agent’s belief in being in the signal phase. The attention choice
probability gradually increases with signal belief increasing from low to high values. Whereas, the
lick choice probability stays flat at 0 until a certain signal belief, after which it steeply increases.
Change in ( **E** ) attention and ( **F** ) lick choice probabilities for increasing food rewards. Change in the
( **G** ) autocorrelation of attention sequences for varying food rewards and the corresponding raster plot
of attention times. ( **H** ) The autocorrelation and raster plot for changes in signal duration.


for signals when it believes it is likely in the noise phase. Regarding the average lick probability, it
remains at zero for low to medium signal beliefs, then increases sharply for higher signal beliefs.
Comparing attention and lick probabilities, we find the agent is much more cautious about licking
than attending when signal beliefs are low. This caution stems from the significant consequences of a
false lick, which incurs a high false alarm cost and prematurely ends the trial, losing opportunities.


When we fix the signal belief value and examine agents trained with different food rewards, we
notice that higher food rewards lead to higher average attention probabilities (Fig 4E). This aligns
with the idea that greater rewards justify increased attention. A similar trend is seen in average lick
probabilities, with higher food rewards making the agent more willing to risk licking (Fig 4F).


In summary, the agent’s likelihood to both attend and lick increases with higher food rewards and
stronger signal beliefs. However, it is important to note that this analysis, based on signal beliefs, is
somewhat simplistic, as the exact policy distribution depends not only on the signal belief but also on
its distribution across the signal states.


**5.3** **Since attention is costly, only use it when necessary**


When attention is costly, we see that a temporal structure emerges in how the agent allocates its
attention. For pedagogical purpose, we first analyze the simpler scenario where low attention does
not yield informative observations. We then extend our discussion to the more general case where
both low and high attention provide informative observations, but with different levels of information.


Initially, we examine the scenario where the agent switches between disengaging ( _p_ low = 0 _._ 5) and
focusing ( _p_ high _>_ 0 _._ 5) during a trial. In Figs 4G and 4H, we display the attention time points
across multiple trials for various food rewards and signal durations, along with their corresponding
autocorrelation plots. For fixed food reward and signal duration, we observe that the initial high
attention time point remains consistent across all trials. This consistency arises because _p_ low = 0 _._ 5
yields observations that carry no information about the underlying latent state, thus not contributing
to belief updates. However, each high attention instance provides an observation used to update the
belief, influencing subsequent action choices. Even during high attention instances, the observation
is randomly drawn (0 _._ 5 _< p_ high _≤_ 1), leading to two possible belief updates even if the latent state
remains unchanged. This randomness, along with stimulus onset randomness, compounds over time,
resulting in significant variability in trial dynamics after the first high attention instance.


7


C Minimum wait time











Minimum wait time Total high attn times













50


25


0





Figure 5: ( **A** ) Auto-correlation of attention sequences as we increase the low attention certainty
from 0 _._ 5. We see that the side peaks start to diminish for increasing low attention certainty. ( **B** )
The distribution of wait times as a function of the signal belief and varying food reward sizes. The
corresponding minimum wait times are shown in ( **C** ). ( **D** ) The number of high-attention instances in
a trial decrease with increasing low-attention certainty.


Despite this variability, we notice a general strategy of deploying blocks of high attention that are
equally spaced. We speculate that this pattern arises from the following principle: as shown in Fig 3B,
the agent initially waits a certain duration before paying the first high attention. During this waiting
period, the signal belief gradually increases, driven solely by the prior over latent state dynamics.
Once the signal belief reaches a sufficient value, the agent pays high attention to check for the signal’s
presence, as the attention cost is justified by the higher chance of detecting the signal. The agent
continues paying high attention until the signal belief either drops below the minimum required for
attention or becomes high enough to trigger a response (e.g., licking). Consecutive high attention
instances appear as blocks in the raster plot, with block sizes varying based on the sequence of
observations during high attention (right panels of Figs 4G and 4H).


If the agent’s signal belief drops just below the minimum required for attention, it waits again until
the belief accumulates sufficiently. Due to the memory-less property of the Markov process, the time
required for the signal belief to increase from one value to another remains consistent, regardless
of history. As a result, the wait times between high attention blocks are approximately the same
(shown in Appendix Fig 7). This is evident as we compute the autocorrelation of the attention
sequences over multiple simulations (left panels of Figs 4G and 4H). We see evenly spaced peaks
in the autocorrelation indicative of the rhythmic pattern in which attention is deployed. We observe
that increasing the food reward size decreases the spacing between the peaks, implying shorter wait
duration and higher attention frequency (Fig 4G). This is because a higher food reward lowers the
minimum signal belief needed to justify high attention, reducing the time required for belief updates.
Conversely, decreasing signal duration increases attention frequency, as a lower minimum signal
belief is needed to avoid missing short-duration signals, resulting in shorter wait periods (Fig 4H). In
summary, when the agent alternates between disengaging ( _p_ low = 0 _._ 5) and engaging ( _p_ high _>_ 0 _._ 5)
during trials, we observe rhythmic attention patterns with attention frequency dependent on task
parameters.


Next, we explore a more general scenario where the agent can adjust its focus, such that 0 _._ 5 _≤_
_p_ low _<_ _p_ high _<_ 1. Previously, with _p_ low = 0 _._ 5, belief updates during low attention were driven
solely by the agent’s prior over latent state dynamics. As low attention certainty increases from 0.5,
observations during low attention become more informative, significantly influencing belief updates.
Consequently, variations in belief updates occur even if underlying latent states remain unchanged,
making the time for signal belief to increase to the minimum required for attention random rather
than fixed. As a result, autocorrelation alone cannot capture this temporal structure, as evident from
the diminishing peaks in Fig 5A with increasing low attention certainty.


For any given signal belief, less than the minimum signal belief needed for attention, there are several
possible observation sequences that can result in moving from current signal belief to that required
for attention. The wait time for that signal belief is the time it takes to go from the current belief to
the minimum required for attention. Due to the stochasticity in observations, this wait time would be
a distribution with a non-zero probability starting at a time that corresponds to the minimum number
of consecutive signal-favoring observations that it needs to move from current signal belief to that
required for attention. When the signal belief increases, this minimum wait time decreases, pushing
the distribution a little closer to 0 each time. This trend continues for increasing signal belief until
when signal belief equals the minimum signal belief, after which the wait time stays at a fixed 0 since


8


it’s now worth paying attention right away. This is shown in Figs 5B, 5C for different values of food
reward sizes and signal beliefs. With increasing food reward, the minimum signal belief for attention
also reduces as the potential returns for incurring attention cost increases. As a decrease in minimum
signal belief for attention implies a decrease in wait time, we see that the wait time distribution shifts
downwards for all signal beliefs with increasing food rewards. We also see that the cusp at which
the minimum wait time for attention goes to 0 also shifts leftwards with increasing food reward, as
this cusp is indeed the minimum signal belief value. Finally, we see for that larger increments low
attention certainty, the number of high attention instances in a trial decreases (see Fig 5D). This
occurs because, with fixed high attention certainty, the relative gain in information between the two
attention levels diminishes as low attention certainty increases. The lower the relative information
gain, the less justified it becomes to incur the attention cost.


**6** **Discussion**


_Temporal structure in correlates of attention._ Our theory provides predictions on how neural and
physiological correlates of attention could change with food reward. We predict the change in the
overall number of high attention instances in a trial, a clear rhythmic pattern of attention instances
when disengagement is an option, and, for the more general case, a distribution of attention as a
function of the animal’s signal belief (probed via its neural correlates).


Interestingly, several studies found similar rhythmic structure in the way attention is deployed. For
example, in visual attention tasks, VanRullen _et al._ [11] developed models based on psychometric
curves to show attention is deployed periodically even when attending to a single target. Busch
and VanRullen [12] showed electrophysiological evidence for attention facilitating perception in a
rhythmic manner. Helfrich _et al._ [13] showed neural evidence in support of functional architecture
of top-down attention being intrinsically rhythmic. Using electroencephalography measurements,
Merholz _et_ _al._ [14] showed that the frequency of periodic attention increased with increasing
attentional demand of the task. In the auditory front, Zalta _et al._ [29] examined trials of individuals
tasked with discerning whether a pure tone stimulus occurred on-beat or off-beat. They found
the performance to be the highest for a specific beat frequency, characterizing it to be the natural
occurring sampling rate of periodic attention. Shen and Alain [30] showed that temporal attention can
be allocated at a particular time for short-term consolidation of auditory information. We hypothesize
that the observed attention patterns could be a consequence of attention cost constraints, and could
benefit from our approach in understanding the underlying principle.


_Ramping Vs Jumping._ Neural signals in the parietal cortex reflect the integration of sensory evidence

[31]. Mean activity ramps up over time, with a slope that depends on the strength of the sensory
evidence. This lead to the expectation that individual neurons encode the current estimates about
decision variables, though activity on single trials is noisy making this computation hard to see.
Later studies argued instead that neurons were not ramping on individual trials, but jumping from
a low to high firing state [32], and that only on average did the rate ramp up. Our modeling efforts
suggest a third, intermediate possibility: neural activity could ramp up in bursts, corresponding to
the fluctuating strength of evidence as attention fluctuates (Fig 3B). These fluctuations should also
exhibit correlations between other proxies for attention, such as pupil dilation [1] or other gain-related
signals in the brain. We plan to test this prediction in future analyses.


_Limitations and future work._ Our observation model and controlled attention unrealistically both
allow only binary values. A continuous model for observation and attention would provide a closer
approximation of the animal’s sensory processing and computations. This might also expand or
contract the task parameter space where rhythmic attention patterns arise. Our model also allows for
fast attentional changes, but a switching cost for transitions between low and high attention states
could better reflect the actual cognitive costs experienced by the animals [24]. Our analysis used the
signal belief as a summary statistic to gain a broad understanding of the animal’s decision-making
process. Future work could delve deeper into the full belief, including uncertainty about timing. This
detailed analysis could address questions such as whether animals increase their attention when they
think they might be approaching the end of the trial.


_Broader impacts._ Understanding the neural foundations of thought would have major impacts on
human life, through neurology, brain-computer interfaces, artificial intelligence, and social communi

9


cation. Our work aims to refine the theoretical foundations for such understanding. Thoughtless or
naive application of these scientific advances could lead to unanticipated consequences and increase


**7** **Conclusion**


We proposed a normative attention model that maximizes task performance when there is a cost for
paying attention. We developed this model for a task inspired by a recent study on an attention-based
auditory foraging task, solved it using reinforcement learning, and analyzed the properties of the
resultant solution. Interestingly, our model suggests that to use attention economically, blocks of
low attention must be interspersed between blocks of high attention. This aligns with the rhythmic
patterns of attention inferred in visual attention literature and provides a normative explanation for
apparent signal neglect (attention lapses) in de Gee et al. Our results suggest principled, quantitative
predictions and encourage further investigation into neural and behavioral correlates of rhythmic
attention.


**Acknowledgments.** This work was supported by the Air Force Office of Scientific Research (AFOSR)
under award number FA9550-21-1-0422.


**References**


[1] JW de Gee, Z Mridha, M Hudson, Y Shi, H Ramsaywak, S Smith, N Karediya, M Thompson,
K Jaspe, H Jiang, et al. Strategic stabilization of arousal boosts sustained attention. _bioRxiv_,
pages 2022–03, 2022.


[2] Marlene R Cohen and John HR Maunsell. A neuronal population measure of attention predicts
behavioral performance on individual trials. _Journal of Neuroscience_, 30(45):15241–15253,
2010.


[3] Marlene R Cohen and John HR Maunsell. When attention wanders: how uncontrolled fluctuations in attention affect performance. _Journal of Neuroscience_, 31(44):15802–15806, 2011.


[4] Jan Willem de Gee, Olympia Colizoli, Niels A Kloosterman, Tomas Knapen, Sander Nieuwenhuis, and Tobias H Donner. Dynamic modulation of decision biases by brainstem arousal
systems. _elife_, 6:e23232, 2017.


[5] David L Gilden and Stephanie Gray Wilson. On the nature of streaks in signal detection.
_Cognitive psychology_, 28(1):17–64, 1995.


[6] Joel S Warm, Raja Parasuraman, and Gerald Matthews. Vigilance requires hard mental work
and is stressful. _Human factors_, 50(3):433–441, 2008.


[7] Francesca C Fortenbaugh, Joseph DeGutis, and Michael Esterman. Recent theoretical, neural,
and clinical advances in sustained attention research. _Annals_ _of_ _the_ _New_ _York_ _Academy_ _of_
_Sciences_, 1396(1):70–91, 2017.


[8] Norman H Mackworth. The breakdown of vigilance during prolonged visual search. _Quarterly_
_journal of experimental psychology_, 1(1):6–21, 1948.


[9] Donald Eric Broadbent. Decision and stress. 1971.


[10] Christian Balkenius and Nils Hulth. Attention as selection-for-action: a scheme for active
perception. In _1999_ _Third_ _European_ _Workshop_ _on_ _Advanced_ _Mobile_ _Robots_ _(Eurobot’99)._
_Proceedings (Cat. No. 99EX355)_, pages 113–119. IEEE, 1999.


[11] Rufin VanRullen, Thomas Carlson, and Patrick Cavanagh. The blinking spotlight of attention.
_Proceedings of the National Academy of Sciences_, 104(49):19204–19209, 2007.


[12] Niko A Busch and Rufin VanRullen. Spontaneous eeg oscillations reveal periodic sampling
of visual attention. _Proceedings of the National Academy of Sciences_, 107(37):16048–16053,
2010.


10


[13] Randolph F Helfrich, Ian C Fiebelkorn, Sara M Szczepanski, Jack J Lin, Josef Parvizi, Robert T
Knight, and Sabine Kastner. Neural mechanisms of sustained attention are rhythmic. _Neuron_,
99(4):854–865, 2018.


[14] Garance Merholz, Laetitia Grabot, Rufin VanRullen, and Laura Dugué. Periodic attention
operates faster during more complex visual search. _Scientific Reports_, 12(1):6688, 2022.


[15] William Ocasio. Attention to attention. _Organization science_, 22(5):1286–1296, 2011.


[16] Grace W Lindsay. Attention in psychology, neuroscience, and machine learning. _Frontiers in_
_computational neuroscience_, 14:516985, 2020.


[17] Martin Sarter, Ben Givens, and John P Bruno. The cognitive neuroscience of sustained attention:
where top-down meets bottom-up. _Brain research reviews_, 35(2):146–160, 2001.


[18] Michael Esterman, Monica D Rosenberg, and Sarah K Noonan. Intrinsic fluctuations in
sustained attention and distractor processing. _Journal of Neuroscience_, 34(5):1724–1730, 2014.


[19] Michael Esterman and David Rothlein. Models of sustained attention. _Current_ _opinion_ _in_
_psychology_, 29:174–180, 2019.


[20] Cesar Bandera, Francisco J Vico, José Manuel Bravo, Mance E Harmon, and Leemon C Baird.
Residual q-learning applied to visual attention. In _ICML_, pages 20–27, 1996.


[21] Lucas Paletta, Gerald Fritz, and Christin Seifert. Q-learning of sequential attention for visual
object recognition from informative local descriptors. In _Proceedings of the 22nd international_
_conference on Machine learning_, pages 649–656, 2005.


[22] Silviu Minut and Sridhar Mahadevan. A reinforcement learning model of selective visual
attention. In _Proceedings of the fifth international conference on Autonomous agents_, pages
457–464, 2001.


[23] Dario Di Nocera, Alberto Finzi, Silvia Rossi, and Mariacarla Staffa. Attentional action selection
using reinforcement learning. In _From_ _Animals_ _to_ _Animats_ _12:_ _12th_ _International_ _Confer-_
_ence on Simulation of Adaptive Behavior, SAB 2012, Odense, Denmark, August 27-30, 2012._
_Proceedings 12_, pages 371–380. Springer, 2012.


[24] Sahiti Chebolu, Peter Dayan, and Kevin Lloyd. Vigilance, arousal, and acetylcholine: Optimal
control of attention in a simple detection task. _PLOS Computational Biology_, 18(10):e1010642,
2022.


[25] Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in
partially observable stochastic domains. _Artificial intelligence_, 101(1-2):99–134, 1998.


[26] Zhengwei Wu, Minhae Kwon, Saurabh Daptardar, Paul Schrater, and Xaq Pitkow. Rational
thoughts in neural codes. _Proceedings of the National Academy of Sciences_, 117(47):29311–
29320, 2020.


[27] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms. _arXiv preprint arXiv:1707.06347_, 2017.


[28] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah
Dormann. Stable-baselines3: Reliable reinforcement learning implementations. _Journal of_
_Machine Learning Research_, 22(268):1–8, 2021.


[29] Arnaud Zalta, Spase Petkoski, and Benjamin Morillon. Natural rhythms of periodic temporal
attention. _Nature communications_, 11(1):1051, 2020.


[30] Dawei Shen and Claude Alain. Temporal attention facilitates short-term consolidation during a
rapid serial auditory presentation task. _Experimental brain research_, 215:285–292, 2011.


[31] Alexander C Huk and Michael N Shadlen. Neural activity in macaque parietal cortex reflects
temporal integration of visual motion signals during perceptual decision making. _Journal of_
_Neuroscience_, 25(45):10420–10436, 2005.


11


[32] Kenneth W Latimer, Jacob L Yates, Miriam LR Meister, Alexander C Huk, and Jonathan W
Pillow. Single-trial spike trains in parietal cortex reveal discrete steps during decision-making.
_Science_, 349(6244):184–187, 2015.


**A** **Appendix**


**Model performance without FA cost:** We see that in the absence of a false alarm (FA) cost, with
increasing food reward, the chances of a FA decreases, chances of a hit increases, and probability
of misses is almost negligible (Fig 6). This is because in the absence of FA cost, there is a strong
incentive to lick at some point in every trial, leading to a negligible probability for missing a trial.
The decrease in FA and increase in hit with increasing food reward can be attributed to a higher
permissible budget for paying attention. However, this is not the trend we observe in the experimental
data. Therefore, inclusion of a sufficiently large FA cost is imperative for the model.



Low
reward


High
reward




#### Performance no false alarm cost



food reward


Figure 6: Agent performance without false
alarm cost.



0 10 20
Wait time



Figure 7: Agent wait time distribution shows
rhythmic attention pattern.



12


