Published as a conference paper at ICLR 2020

## Symplectic Recurrent Neural Networks


**Zhengdao** **Chen** _[a,c]_ **,** **Jianyu** **Zhang** _[b,c]_ **,**
**Martin** **Arjovsky** _[a]_ **,** **Léon** **Bottou** _[c,a]_

_a_ New York University, New York, USA
_b_ Tianjin University, Tianjin, China
_c_ Facebook AI Research, New York, USA


Abstract


We propose Symplectic Recurrent Neural Networks (SRNNs) as learning
algorithms that capture the dynamics of physical systems from observed
trajectories. An SRNN models the Hamiltonian function of the system by a
neural network and furthermore leverages symplectic integration, multiplestep training and initial state optimization to address the challenging numerical issues associated with Hamiltonian systems. We show SRNNs succeed reliably on complex and noisy Hamiltonian systems. We also show how
to augment the SRNN integration scheme in order to handle stiff dynamical
systems such as bouncing billiards.


1 Introduction


Can machines learn physical laws from data? A recent paper (Greydanus et al., 2019),
Hamiltonian Neural Networks (HNN), proposes to do so representing the Hamiltonian function _H_ ( _q, p_ ) as a multilayer neural network. The partial derivatives of this network are then
trained to match the time derivatives _p_ ˙ and _q_ ˙ observed along the trajectories in state space.

The ordinary differential equations (ODEs) that express Hamiltonian dynamics are famous
for both their mathematical elegance and their challenges to numerical integration techniques. Except maybe for the simplest Hamiltonian systems, discretization errors and measurement noise lead to quickly diverging trajectories. In other words, Hamiltonian systems
can often be _stiff_, a concept that usually refers to differential equations where we have to
take very small time-steps of integration so that the numerical solution remain stable (Lambert, 1991). A plethora of numerical integration methods, _symplectic_ _integrators_, have been
developed to respect the conserved quantities in Hamiltonian systems, thereby usually being
more stable and structure-preserving than non-symplectic ones (Hairer et al., 2002). For
example, the simplest symplectic integrator is the well-known leapfrog method, also known
as the Stömer-Verlet integrator (Leimkuhler and Reich, 2005). However, even the best integrators remain severely challenged by phenomena as intuitive as a mechanical rebound or a
slingshot effect, which are more severe forms of stiffness. Such numerical issues are almost
doomed to conflict with the inherently approximate nature of a learning algorithm.

In the first part of this paper, we propose _Symplectic_ _Recurrent_ _Neural_ _Networks_ (SRNNs),
where ( _i_ ) the partial derivatives of the neural-network-parametrized Hamiltonian are integrated with the leapfrog integrator and where ( _ii_ ) the loss is back-propagated through
the ODE integration over multiple time steps. We find that in the presence of observation noise, SRNN are far more usable than HNNs. Further improvements are achieved by
simultaneously optimizing the initial state and the Hamiltionian network, presenting an interesting contrast to previous literature on the hardness of general initial state optimization
(Peifer and Timmer, 2007). The optimization can be motivated from a maximum likelihood estimation perspective, and we provide heuristic arguments for why the initial state
optimization is likely convex given the symplecticness of the system. Furthermore, experiments in the three-body problem show that the SRNN-trained Hamiltonian compensates for
discretization errors and can even outperform numerically solving the ODE using the true
Hamiltonian and the same time-step size. This could be of particular interest to researchers
who study the application of machine learning to numerically solving differential equations.

The second part of this paper focuses on _perfect_ _rebound_ as an example of the more severe
form of stiffness. When a point mass rebounds without loss of energy on a perfectly rigid


1


Published as a conference paper at ICLR 2020


obstacle, the motion of the point mass is changed in ways that can be interpreted as an
infinite force applied during an infinitesimal time. The precise timing of this event affects the
trajectory of the point mass in ways that essentially make it impossible to merely simulate
the Hamiltonian system on a predefined grid of time points. In order to address such
events in learning, we augment the leapfrog integrator used in our SRNN with an additional
trainable operator that models the rebound events and relates their occurrence to visual
hints. Training such an augmented SRNN on observed trajectories not only learns the point
mass dynamics but also learns the visual appearance of the obstacles.


2 Related work


**Learning** **physics** **with** **neural** **networks** A popular category of methods attempts to
replicate the intuitive ways in which humans perceive simple physical interactions, identifying objects and learning how they relate to each other (Battaglia et al., 2016; Chang et al.,
2016). Since such methods cannot be used for more general physical systems, another category of methods seeks to learn which differential equations govern the evolution of a physical
system on the basis of observed trajectories. Brunton et al. (2016) assemble a small number
of predefined primitives in order to find an algebraically simple solution. Lutter et al. (2019)
use a neural network to model the Lagrangian function of a robotic system. Most closely
related to ours, Greydanus et al. (2019) use a neural network to learn the Hamiltonian of
the dynamical system in such a way that its partial derivatives match the time derivatives
of the position and momentum variables, which are both assumed to be observed. Although
the authors show success on a simple pendulum system, this approach does not perform
well on a more complex system such as a three-body problem. Concurrently to our work,
Sanchez-Gonzalez et al. (2019) proposes to learn Hamiltonian dynamics by combining graph
networks with ODE integrators.

**ODE-based** **learning** **and** **recurrent** **neural** **networks** **(RNNs)** To learn an ODE
that underlies some observed time series data, Chen et al. (2018a) proposes to solve a neuralnetwork-parameterized ODE numerically and minimize the distance between the generated
time series with the observed data. To save memory, they propose to use the adjoint
ODE instead of back-propagating through the ODE solver. Using stability analysis of
ODEs, Chang et al. (2019) propose the AntisymmetricRNN with better trainability. Niu
et al. (2019) establish a correspondence between RNNs and ODEs, and propose an RNN
architecture inspired by a universal quantum computation scheme.

_Summary_ _of_ _our_ _main_ _contributions:_ In this paper, we propose SRNN, which


_•_ learns Hamiltonian dynamics directly from position and momentum time series

_•_ performs well on noisy and complex systems such as a spring-chain system and a
three-body system, and is compatible with initial state optimzation

_•_ is augmented to handle perfect rebound, an example of very stiff Hamiltonian dynamics


3 Framework


3.1 Hamiltonian systems


A Hamiltonian system of dimension _d_ is described by two vectors _p, q_ _∈_ R _[d]_ . Typically,
they correspond to the momentum and position variables, respectively. The evolution of
the system is determined by the Hamiltonian function _H_ : ( _p, q, t_ ) _∈_ R [2] _[d]_ [+1] _�→_ _H_ ( _p, q, t_ ) _∈_ R
through a system of ordinary differential equations called _Hamilton’s_ _equations_,



_p_ ˙ = _−_ _[∂H]_



(1)
_∂p_ _[,]_




_[∂H]_ _q_ ˙ = + _[∂H]_

_∂q_ _[,]_ _∂p_



where we use the dot notation to compactly represent derivatives with respect to the time
variable _t_ . We are focusing in this work on Hamiltonians that are _conservative_, [1] that is,


1In our opinion, extending this work to non-conservative systems should not be done by adding a
time dependency in the Hamiltonian, but by adding additional dissipation or intervention operators
in the numerical integration schema, as illustrated in section 6.


2


Published as a conference paper at ICLR 2020


they do not depend on the time variable _t_, and _separable_, [2] that is, they can be written as a
sum _H_ ( _p, q_ ) = _K_ ( _p_ ) + _V_ ( _q_ ). In this case, (1) becomes

_p_ ˙ = _−V_ _[′]_ ( _q_ ) _,_ _q_ ˙ = _K_ _[′]_ ( _p_ ) (2)


With a proper choice of the _p_ and _q_ variables, the evolution of essentially all physical systems
can be described with the Hamiltonian framework. In other words, Hamilton’s equations
restrict the vast space of dynamical systems to the considerably smaller space of dynamical
systems that are physically plausible.

Therefore, instead of modeling the dynamics of a physical system with a neural network
_fθ_ ( _p, q_ ) whose outputs are interpreted as estimates of the time derivatives _p_ ˙ and _q_ ˙, we
can also use a neural network _Hθ_ ( _p, q_ ) = _Kθ_ 1( _p_ ) + _Vθ_ 2( _q_ ) with _θ_ = [ _θ_ 1 _, θ_ 2], whose partial
derivatives _−Vθ_ _[′]_ 2 [(] _[q]_ [)] [and] _[K]_ _θ_ _[′]_ 1 [(] _[p]_ [)] [are] [interpreted] [as] [the] [time] [derivatives] _[p]_ [˙] [and] _[q]_ [˙][.] [We] [refer]
to the former as ODE neural networks (O-NET) and the latter approach as Hamiltonian
neural networks (H-NET). In order to define a complete learning system, we need to explain
how to determine the parameter _θ_ of the neural networks on the basis of observed discrete
trajectories. For instance, Greydanus et al. (2019) trains H-NET in a fully supervised
manner using the observed tuples ( _p, q,_ _p,_ ˙ _q_ ˙).


3.2 From ODEs to discrete trajectories


A numerical integrator (or ODE solver) approximates the true solution of an ODE of the
form _z_ ˙ = _f_ ( _z, t_ ) at discrete time steps _t_ 0 _, t_ 1 _. . .tT_ . For instance, the simplest integrator,
Euler’s integrator, starts from the initial state _z_ 0 at time _t_ 0 and estimates the function _z_ ( _t_ )
at uniformly spaced time points _tn_ = _t_ 0 + _n_ ∆ _t_ with the recursive expression


_zn_ +1 = _zn_ + ∆ _t f_ ( _zn, tn_ ) (3)


In stiff ODE systems, however, using Euler’s method could easily lead to unstable solutions
unless the time-step is chosen to be very small (Lambert, 1991). The development of efficient
and accurate numerical integrators is the object of considerable research (Hairer et al.,
2008; Hairer and Wanner, 2013). Symplectic integrators [3] are particularly attractive for
the integration of Hamilton’s equations (Leimkuhler and Reich, 2005). They are able to
preserve quadratic invariants, and therefore usually have desired stability properties as well
as being structure-preserving (McLachlan et al., 2004), even for certain non-Hamiltonian
systems (Chen et al., 2018b). A simple and widely-used symplectic integrator is the leapfrog
integrator. When the Hamiltonian is conservative and separable (2), it computes successive
estimates ( _pn, qn_ ) with



_pn_ +1 _/_ 2 = _pn −_ [1] 2 [∆] _[t V]_ _[′]_ [(] _[q][n]_ [)]

_qn_ +1 = _qn_ + ∆ _t K_ _[′]_ ( _pn_ +1 _/_ 2)



(4)



_pn_ +1 = _pn_ +1 _/_ 2 _−_ [1] 2 [∆] _[tV]_ _[′]_ [(] _[q][n]_ [+1][)]


Repeatedly executing update equations (4) is called the _leapfrog_ _algorithm_, which is as
computationally efficient as Euler’s method yet considerably more accurate when the ODE
belongs to a Hamiltonian system (Leimkuhler and Reich, 2005).


3.3 Learning ODEs from discrete trajectories


Following Chen et al. (2018a), let the right hand side of the ODE be a parametric function
_fθ_ ( _z, t_ ) and let _z_ 0 _. . . zT_ be an observed trajectory measured at uniformly spaced time points
_t_ 0 _. . . tT_ . We can estimate the parameter _θ_ that best represents the dynamics of the observed
trajectory by minimizing the mean squared error [�] _i_ _[T]_ =1 _[∥][z][i][ −]_ _[z]_ [ˆ] _[i]_ [(] _[θ]_ [)] _[∥]_ [2] [between] [the] [observed]
trajectory _{zi}_ _[T]_ _i_ =0 [and] [the] [trajectory] _[{][z]_ [ˆ] _[i]_ [(] _[θ]_ [)] _[}][T]_ _i_ =0 [generated] [with] [our] [integrator] [of] [choice,]

_{z_ ˆ _i_ ( _θ_ ) _}_ _[T]_ _i_ =0 [=] _[ Integrator]_ [(] _[z]_ [0] _[, f][θ][,][ {][t][i][}]_ _i_ _[T]_ =0 [)] _[.]_


For instance, this minimization can be achieved using stochastic gradient descent after
back-propagating through the steps of our numerical integration algorithm of choice and


2Extending this work to non-separable Hamiltonians can be achieved by rewriting the numerical
integration schema using an extended phase space (Tao, 2016).
3An integrator is symplectic if applied to Hamiltonian systems, its flow maps are symplectic for
short enough time-steps. For details, see Hairer et al. (2002) and Leimkuhler and Reich (2005).


3


Published as a conference paper at ICLR 2020


then through each call to the functions _fθ_ . This can be done when _fθ_ ( _z_ ) is a neural network
(O-NET), or is the concatenation [ _−Vθ_ _[′]_ 2 [(] _[q]_ [)] _[, K]_ _θ_ _[′]_ 1 [(] _[p]_ [)]] [of] [the] [partial] [derivatives] [of] [an] [H-NET]
_Hθ_ ( _p, q_ ) = _Kθ_ 1( _p_ ) + _Vθ_ 2( _q_ ), where the partial derivatives can be expressed using the same
parameters _θ_ as the Hamiltonian _Hθ_ ( _p, q_ ), for instance using automatic differentiation. We
can then predict trajectories at testing time using the trained _fθ∗_ and initial state _z_ 0 [test],

_{z_ ˆ _i_ [test] _}_ _[T]_ _i_ =0 [ test] [=] _[ Integrator]_ [(] _[z]_ 0 [test] _, fθ∗_ _, {ti}_ _[T]_ _i_ =0 [ test][)] _[.]_


Note that neither the integrator, nor the number of steps, nor the step size, need to be the
same at training and testing.


3.4 Symplectic Recurrent Neural Network


This framework provides a number of nearly orthogonal design options for the construction
of algorithms that model dynamical systems using trajectories:


_•_ The time derivative model could be an O-NET or H-NET.

_•_ The training integrator can be any explicit integrators. In our experiments, we only
focus on Euler’s integrator and the leapfrog integrator.

_•_ The training trajectories can consist of a single step, _T_ =1, or multiple steps, _T>_ 1.
We refer to the first case as _single-step_ and the second case as _multi-step_ or _recurrent_
training, because back-propagating through multiple steps of the training integrator
is comparable to back-propagating through time in recurrent networks.

_•_ The testing integrator can also be chosen freely and can use a different time-step
size as it does not involve back-propagation.

In order to save space while describing the possibly different integrators used for training
and testing, we use the labels “E-E”, “E-L”, and “L-L”, where the first letter tells which
integrator was used for training —“E” for Euler and “L” for leapfrog— and the second letter
indicates which integrator was used as testing time. For instance, with our terminology, the
HNN model of Greydanus et al. (2019) is a “single-step E-E H-NET” with the additional
subtlety that they supervise the training with actual derivatives instead of relying on finite
differences between successive steps of the observed trajectories.

A _Symplectic_ _Recurrent_ _Neural_ _Network_ (SRNN) is a recurrent H-NET that relies on a
symplectic integrator for both training and testing, such as, for instance, a "recurrent L-L
H-NET". As shown in the rest of this paper, SRNNs are far more usable and robust than
the alternatives, especially when the Hamiltonian gets complex and the data gets noisy.
We believe that SRNNs may also have other potential benefits: because leapfrog preserves
volumes in the state space (Hairer et al., 2002), we conjecture that vanishing and exploding
gradients’ issues in backpropagating through entire state sequences are ameliorated (Arjovsky et al., 2015). Finally, because the leapfrog integrator is reversible in time, there is
no need to store states during the forward pass as they can be recomputed exactly during
the backward pass. We leave studying these other computational and optimization benefits
as a topic of future work.


4 SRNN can learn complex and noisy Hamiltonian dynamics


As an example of a complex Hamiltonian system, we first present experiments performed on
the spring-chain system: a chain of 20 masses with neighbors connected via springs. Each of
the two masses on the ends are connected to fixed ground via another spring. The chain can
be assumed to lay horizontally and the masses move vertically but no gravity is assumed.
The 20 masses and the 21 spring constants are chosen randomly and independently. The
training data consist of 1000 trajectories of the same chain, each of which starts from
a random initial state of positions and momenta of the masses and is 10-time-step long
(including the initial state). We thus take _T_ = 9 when performing recurrent training. When
performing single-step training, each training trajectory of length 10 is instead considered
as 9 consecutive trajectories of length 2. In this way, 1000 sample trajectories of length
10 ( _T_ =9) are turned into 9000 sample trajectories of length 2 ( _T_ =1), allowing for a fair
comparison between single-step training and recurrent training. During testing, the trained
model is given 32 random initial states in order to predict 32 trajectories of length 100.
Detailed experiment setups and model architectures are provided in Appendix A.1, and a
PyTorch implementation can be found at `[https://github.com/zhengdao-chen/SRNN.git](https://github.com/zhengdao-chen/SRNN.git)` .


4


Published as a conference paper at ICLR 2020



10


8


6


4


2


0



|Col1|Single-step E-E O-NET<br>(HNN) Single-step E-E H-NET<br>Single-step E-L O-NET<br>Single-step E-L H-NET<br>Single-step L-L O-NET<br>Single-step L-L H-NET|
|---|---|
|||


time step





4


2


0


2


4





|Col1|(HNN) Single-step E-E H-NET<br>Single-step E-L H-NET<br>Single-step L-L H-NET<br>Observations|
|---|---|
|||


time step



Figure 1: Testing results in the noiseless case by single-step methods. Left: Prediction error
of each method over time, measured by the L2 distance between the true and predicted
positions of the 20 masses. Right: Each curve represents the position of one of the masses
(number 5) as a function of time predicted by the three single-step-trained H-NET models.
Plots of the other masses’ positions are provided in Appendix D.1.


4.1 Going symplectic - rescuing HNN with the leapfrog integrator


First, we consider the noiseless case, where the training data consist of exact values of the
positions ( _q_ ) and momenta ( _p_ ) of the masses on the chain at each discrete time point. As
shown in figure 1, the prediction of a single-step E-E H-NET deviates from the ground truth
quickly and is unable to capture the periodic motion. By comparison, a single-step E-E ONET yields predictions that is qualitatively reasonable. This shows that using Hamiltonian
models without paying attention to the integration scheme may not be a good idea.

We then replace Euler’s integrator used during testing by a leapfrog integrator, yielding a
Single-step E-L H-NET. Figure 1 shows that this helps the H-NET produce predictions that
remain stable and periodic over a longer period of time. Since the training process remains
the same, this implies that part of the instability and degeneration of H-NET’s predictions
comes from the nature of Euler’s integrator rather than the lack of proper training.

In contrast, using a leapfrog integrator for both training and testing substantially improve
the performance, as also shown again in figure 1. This improvement shows the importance
of consistency between the integrators used in training and predicting modes. This can be
understood with the concept of _modified_ _equations_ (Hairer, 1994): when we use a numerical
integrator to solve an ODE, the numerical solution usually does not strictly follow the
original equation due to discretization, but can be regarded as a solution to a modified
version of the original equation that depends on the integrator and the time-step size.
Therefore, training and testing with the same numerical integrator and time-step size could
allow the system to learn a modified Hamiltonian that corrects some of the errors caused
by the discretization scheme.


4.2 Going recurrent - using multi-step training when noise is present


Since noise is prevalent in real-world observations, we also test our models on noisy trajectories. Independent and identically distributed Gaussian noise is added to both the position
and the momentum variables at each time step. Applying the single-step methods described
above yield considerably worse predictions, as shown in Figure 2 (left).

This phenomenon can be controlled by training on multiple steps, effectively arriving at a
type of recurrent neural network: if noise is added independently at each time-step, then
having data from multiple consecutive time steps may allow us to discern the actual noiseless
trajectory, analogous to performing linear regression on multiple (more than 2) noisy data
points. As we see in Figure 2 (left), recurrent training consistently improves the predictions
except for E-E H-NET. The best performing model is the SRNN (recurrent L-L H-Net)
which improves substantially over the single-step L-L H-NET. Interestingly, the recurrent
E-E H-NET does not improve over the single-step E-E H-NET, which means that recurrent
training does not help if one uses a naïve integrator.


4.3 Initial state optimization (ISO)


However, one issue remains to be addressed: in the framework that we have adopted so far,
the initial states _p_ 0 and _q_ 0 are treated as the actual initial states from which the system


5


Published as a conference paper at ICLR 2020



10


8


6


4


2


0



time step



time step





10


8


6


4


2


0





Figure 2: Prediction error of all methods in the noisy case measured by L2 distance, presented in two plots due to the large number of methods. Included in the left plot are the
single-step-trained methods, recurrently trained methods, vanilla RNN and LSTM. Included
in the right plot are the (same) recurrently trained methods, the recurrently trained methods
with initial state optimization (ISO), as well as vanilla RNN and LSTM with ISO.



4


2


0


2


4



4


2


0


2


4



4


2


0


2


4



HNN
SRNN
SRNN-ISO
Noiseless dynamic
Noisy observations



time step



time step



time step



(a) HNN (b) SRNN (c) SRNN-ISO


Figure 3: Predictions made by three methods in the noisy case. The Y-axis corresponds to
the position of one of the masses (number 5) on the chain.


begins to evolve despite the added noise in observation. With noise added to the observation
of _p_ 0 and _q_ 0, our dynamical models will start from these noisy states and remain biased as
we advance in time in both the training and the testing mode.

To mitigate this issue, we propose to introduce two new parameter vectors for each sample,
_p_ ˆ0 and _q_ ˆ0, interpreted as our estimate of the actual initial states, and we let our dynamical
models evolve starting from them instead of the observed _p_ 0 and _q_ 0. Treating _p_ ˆ0 and _q_ ˆ0
as parameters, we can optimize them based on the loss function while fixing the model’s
parameters, a process that we call _initial_ _state_ _optimization_ (ISO). When the model is good
enough, we hope that this will guide us towards the true initial states without observation noise. In Appendix B, we motivate the use of ISO from the perspective of maximum
likelihood inference. In actual training, we first train the neural network parameters for
100 epochs as usual, and starting from the 101st, after every epoch we perform ISO with
the L-BFGS-B algorithm (Zhu et al., 1997) on the _p_ ˆ0 and _q_ ˆ0 parameters for every training
trajectory. At testing time, the model is given the noisy values of _p_ and _q_ for the first 10
time steps and must complete the trajectory for the next 200 steps. These 10 initial time
steps allow us to perform the same L-BFGS-B optimization to determine the initial state
_p_ ˆ0 before advancing in time to predict the entire trajectory.

As seen in Figure 2 (right), SRNN-ISO (i.e. SRNN equipped with ISO) clearly yields the
best prediction among all the methods. Figure 3 shows the predictions of HNN, SRNN and
SRNN-ISO on one test sample, and we clearly see the qualitative improvements thanks to
recurrent training and ISO. O-NET also benefits from ISO while vanilla RNN and LSTM
do not seem to, likely because the initial state optimization only works when we already
have a reasonable model of the system. In Appendix C, we give a heuristic argument for
the convexity of ISO, which helps to explain the success of using L-BFGS-B for ISO.

In summary, we have proposed three extensions to learning complex and noisy dynamics
with H-NET and demonstrated the improvements they lead to: a) using the leapfrog integrator instead of Euler’s integrator; b) using recurrent instead of single-step training; and c)


6


Published as a conference paper at ICLR 2020


Table 1: Testing results of predicting the dynamics of the spring-chain system by methods
based on fixed _p_ 0, _q_ 0 (i.e., not optimizing _p_ 0, _q_ 0 as parameters). The error is defined as the
discrepancy between the (noisy) ground truth and the predictions at each time step averaged
over the first 200 time steps, where the discrepancy is measured by the L2 distance between
the true and predicted positions of the 20 masses in the chain, both of which considered
as 20-dimensional vectors. The mean and standard deviation are computed based on 32
testing samples, each starting from a random configuration of the chain.

|Col1|Model|Integrator (tr)|Integrator (te)|Error mean|Error std|
|---|---|---|---|---|---|
|single-step|O-NET|Euler|Euler|6.93|1.22|
|single-step|O-NET|Euler|Leapfrog|5.87|1.04|
|single-step|O-NET|Leapfrog|Leapfrog|7.28|1.48|
|single-step|H-NET|Euler|Euler|7.24|0.64|
|single-step|H-NET|Euler|Leapfrog|3.32|0.89|
|single-step|H-NET|Leapfrog|Leapfrog|3.36|0.67|
|recurrent|O-NET|Euler|Euler|2.88|0.45|
|recurrent|O-NET|Euler|Leapfrog|4.12|0.41|
|recurrent|O-NET|Leapfrog|Leapfrog|3.34|0.86|
|recurrent|H-NET|Euler|Euler|7.58|0.63|
|recurrent|H-NET|Euler|Leapfrog|5.26|0.63|
|recurrent|H-NET|Leapfrog|Leapfrog|**2.37**|**0.87**|
|recurrent|Vanilla RNN|N/A|N/A|4.80|0.82|
|recurrent|LSTM|N/A|N/A|5.95|1.05|



Table 2: Testing results of predicting the dynamics of the spring-chain system by methods
that optimize on _p_ 0 and _q_ 0 starting from their observed (noisy) values using L-BFGS-B, as
explained in the text. The definition of the errors is the same as in the above table.

|Model|Integrator (tr)|Integrator (te)|Error mean|Error std|
|---|---|---|---|---|
|O-NET|Euler|Euler|2.13|0.37|
|O-NET|Euler|Leapfrog|3.59|0.50|
|O-NET|Leapfrog|Leapfrog|2.27|0.60|
|H-NET|Euler|Euler|6.26|0.60|
|H-NET|Euler|Leapfrog|3.00|0.63|
|H-NET|Leapfrog|Leapfrog|**1.45**|**0.32**|
|Vanilla RNN|N/A|N/A|4.72|0.94|
|LSTM|N/A|N/A|5.81|0.98|



optimizing the initial states of each trajectory as parameters when data are noisy. Thorough
comparisons of test errors are given in Tables 1 and 2, where we highlight that the SRNN
(recurrent L-L H-NET) models achieve the lowest errors.


5 SRNN can learn the dynamics of a three-body system


Next, we test SRNN with the three-body system, which is a well-known example of a chaotic
system, meaning that a small difference in the initial condition could lead to drastically different evolution trajectories, even without noise added. As a result, even when the exact
equations are known, simulating it with different time-step sizes could also lead to qualitatively different solutions. Moreover, Greydanus et al. (2019) mentions that HNN does not
outperform a baseline method using O-NET in learning the three-body system’s evolution.
Here, we test our SRNN together with other baselines on the noiseless three-body system
with the same configurations as Greydanus et al. (2019). The detailed experimental setup
and model architectures are provided in Appendix A.2.

As we see in Table 3, the best-performing model is SRNN and the second-best is the singlestep L-L H-NET. Interestingly, and perhaps counter-intuitively, they even outperform the
baseline method of simulating the correct equation with the same time-step size. How is this
possible? In short, our explanation is that the error introduced by numerical discretization
could be learned and therefore compensated for by the models we train. More concretely,
once again using the concept of modified equations mentioned in Section 4.1, we argue that
the ODE-based learning models, including both H-NET and O-NET models, could learn _not_


7


Published as a conference paper at ICLR 2020


Table 3: Prediction error results for the three-body system with time-step ∆ _t_ = 1. The last
row corresponds to numerically solving the correct underlying equations using the leapfrog
integrator with time-step ∆ _t_ = 1. The other rows correspond to the different learning-based
methods, same as in the spring-chain experiments.

|Col1|Model|Integrator (tr)|Integrator (te)|Error mean|Error std|
|---|---|---|---|---|---|
|single-step|O-NET|Euler|Euler|0.65|0.16|
|single-step|O-NET|Euler|Leapfrog|1.36|0.18|
|single-step|O-NET|Leapfrog|Leapfrog|1.33|0.20|
|single-step|H-NET|Euler|Euler|1.64|0.25|
|single-step|H-NET|Euler|Leapfrog|0.88|0.33|
|single-step|H-NET|Leapfrog|Leapfrog|0.35|0.09|
|recurrent|O-NET|Euler|Euler|0.51|0.11|
|recurrent|O-NET|Euler|Leapfrog|1.27|0.18|
|recurrent|O-NET|Leapfrog|Leapfrog|0.49|0.10|
|recurrent|H-NET|Euler|Euler|0.79|0.17|
|recurrent|H-NET|Euler|Leapfrog|1.76|0.62|
|recurrent|H-NET|Leapfrog|Leapfrog|**0.26**|**0.07**|
|simulation|true eqns.|(no training)|Leapfrog|0.47|0.18|



the correct underlying equation, but rather the equation whose modified equation associated
with our choice of numerical integrator and time-step size is the original equation. Hence,
when the time-step size is large and the error of numerical discretization is not negligible,
it is possible that the learned equation could yield better predictions than the correct one.

In addition, we also see that the recurrently trained models outperform the corresponding
single-step-trained models. Plots of the predicted trajectories are provided in Appendix E.


6 Learning perfect rebound with an augmented SRNN


We focus in this section on the _perfect_ _rebound_ problem as a prototypical example of stiff
ODE in a physical system. We consider a heavy billiard, subject to gravitational forces
pointing downwards, and bouncing around a two-dimensional square domain delimited by
impenetrable walls. Whenever it hits a wall, the billiard rebounds without loss of energy, by
reversing the component of its momentum orthogonal to the wall surface. Microscopically,
when the billiard hits the wall, the atomic structure deformation produces strong electromagnetic forces that reverse the momentum during a very brief timescale. Simulating this
microscopic phenomenon with a Hamiltonian ODE would not only be computationally expensive, but also require a detailed knowledge of the atomic structures of the billiard and
the walls. The perfect rebound is a macroscopic approximation that treats the billiard as a
point mass and the rebound as an event with zero duration infinite forces. Although this
approximation is convenient for high-school level derivations, the singularity makes it hard
to simulate using Hamiltonian dynamics.

We propose to approach this problem by augmenting each time step of a leapfrog-based
SRNN with an additional operation that models a possible rebound event,

_p_ _[post]_ _t_ _←_ _p_ _[pre]_ _t_ _−_ 2( _p_ _[pre]_ _t_ _· n_ ) _n_ _,_ (5)

where _p_ _[pre]_ _t_ is the pre-rebound momentum vector and _p_ _[post]_ _t_ is the post-rebound momentum
vector. When the vector _n_ is zero, this operation does not change the momentum in any
way. When _n_ is a unit vector orthogonal to a wall, this operation computes the momentum
reversal that is characteristic of a perfect rebound. Vectors _n_ of smaller length could also be
used to model energy dissipation in manner that is reminiscent of the famous LSTM forget
gate (Hochreiter and Schmidhuber, 1997).

Because the billiard trajectory depends on the exact timing of the rebound event, we also
need a scalar _α ∈_ [0 _,_ 1] that precisely places the rebound event at time _t_ + _α_ ∆ _t_ between the
successive time steps _t_ and _t_ + ∆ _t_ . The augmented leapfrog schema then becomes


_leapfrog_

[ _p_ _[pre]_ _t_ + _α_ ∆ _t_ _[, q]_ _t_ _[pre]_ + _α_ ∆ _t_ []] _←−−−−−−_ [ _pt, qt_ ] (6)
_α_ ∆ _t_

_p_ _[post]_ _t_ + _α_ ∆ _t_ [=] _[ p]_ _t_ _[pre]_ + _α_ ∆ _t_ _[−]_ [2(] _[p]_ _t_ _[pre]_ + _α_ ∆ _t_ _[·][ n]_ [)] _[n]_ (7)


8


Published as a conference paper at ICLR 2020


Figure 4: Actual versus predicted trajectories of the heavy billiard with perfect rebound.
The predictions are obtained by an SRNN plus the rebound module described in section 6.


_leapfrog_

[ _pt_ +∆ _t, qt_ +∆ _t_ ] _←−−−−−−_ _t_ + _α_ ∆ _t_ _[, q]_ _t_ _[post]_ + _α_ ∆ _t_ []] (8)
(1 _−α_ )∆ _t_ [[] _[p][post]_


where equations (6) and (8) represent ordinary leapfrog updates (4) for time steps of respective durations _α_ ∆ _t_ and (1 _−_ _α_ )∆ _t_ . More precisely, we first compute a tentative position
_q_ ˜ _t_ +∆ _t_ and momentum _p_ ˜ _t_ +∆ _t_ assuming no rebound,


_leapfrog_

[˜ _pt_ +∆ _t,_ ˜ _qt_ + _α_ ∆ _t_ ] _←−−−−−−_ [ _pt, qt_ ] _,_ (9)
∆ _t_


then compute both _n_ and _α_ as parametric functions of the tentative position _q_ ˜ _t_ +∆ _t_ as well
as the current position _qt_, and finally apply the forward model (6–8). Note that the final
state is equal to the tentative state when no rebound occurs, that is, when _n_ = 0.

Directly modeling _n_ and _α_ with a neural network taking _q_ ˜ _t_ +∆ _t_ as the input would be very
inefficient because we would need to train with a lot of rebound events to precisely reveal
the location of the walls. We chose instead to use _visual_ _cues_ in the form of a background
image representing the walls. We model _n_ as the product of a direction vector _n_ ¯ and a
magnitude _γ_ _∈_ [0 _,_ 1], and we want the latter to take value close to 1 when perfect rebound
actually occurs between _t_ and _t_ + ∆ _t_ and close to 0 otherwise. Both _n_ ¯ and _α_ are modeled
as MLPs that take as input two 10x10 neighborhoods of the background image, centered
at positions _qt_ and _q_ ˜ _t_ +∆ _t_, respectively. In contrast, _γ_ is modeled as an MLP that takes
as input a smaller 2x2 neighborhood centered at _q_ ˜ _t_ +∆ _t_ and is trained with an additional
regularization term _∥γ∥_ 1 in order to switch the rebound module off when it is not needed.

Training is achieved by back-propagating through the successive copies of the augmented
leapfrog scheme, through the models of _n_ ¯, _α_, and _γ_, and also through the computation of
the tentative _p_ ˜ _t_ +∆ _t_ and _q_ ˜ _t_ +∆ _t_ . We use 5000 training trajectories of length 10 starting from
a randomly-sampled initial positions and velocities. Similarly, we use 32 testing trajectories
of length 60. Detailed exprimental setup is included in Appendix A.3. Figure 5 plots
some predicted and actual testing trajectories. Appendix F compares these results with
the inferior results obtained with several baseline methods, including SRNN without the
rebound module, and SRNN with a rebound module that does not learn _α_ . One limitation
of our method, however, results from the assumption that there is at most one rebound
event per time step. Although this assumption fails when the billiard rebounds twice near
a corner, as shown in the bottom right plot in Figure 5, our method still outperforms the
baseline methods even in this case.


9


Published as a conference paper at ICLR 2020


7 Conclusion


We propose the Symplectic Recurrent Neural Network, which learns the dynamics of Hamiltonian systems from data. Thanks to symplectic integration, multi-step training and initial
state optimization, it outperforms previous methods in predicting the evolution of complex
and noisy Hamiltonian systems, such as the spring-chain and the three-body systems. It can
even outperform simulating with the exact equations, likely by learning to compensate for
numerical discretization error. We further augment it to learn perfect rebound from data,
opening up the possibility to handle stiff systems using ODE-based learning algorithms.


Acknowledgments


The authors acknowledge stimulating discussions with Dan Roberts, Marylou Gabrié, Anna
Klimovskaia, Yann Ollivier and Joan Bruna.


References

Arjovsky, M., Shah, A., and Bengio, Y. (2015). Unitary evolution recurrent neural networks.
_CoRR_, abs/1511.06464.


Battaglia, P. W., Pascanu, R., Lai, M., Rezende, D. J., and Kavukcuoglu, K. (2016). Interaction networks for learning about objects, relations and physics. _CoRR_, abs/1612.00222.


Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). Discovering governing equations from
data by sparse identification of nonlinear dynamical systems. _Proceedings_ _of_ _the_ _National_
_Academy_ _of_ _Sciences_, 113(15):3932–3937.


Chang, B., Chen, M., Haber, E., and Chi, E. H. (2019). AntisymmetricRNN: A dynamical system view on recurrent neural networks. In _International_ _Conference_ _on_ _Learning_
_Representations_ .


Chang, M. B., Ullman, T., Torralba, A., and Tenenbaum, J. B. (2016). A compositional
object-based approach to learning physical dynamics. _CoRR_, abs/1612.00341.


Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. (2018a). Neural ordinary
differential equations. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., CesaBianchi, N., and Garnett, R., editors, _Advances in Neural Information Processing Systems_
_31_, pages 6571–6583. Curran Associates, Inc.


Chen, Z., Raman, B., and Stern, A. (2018b). Structure-preserving numerical integrators for
hodgkin-huxley-type systems.


Greydanus, S., Dzamba, M., and Yosinski, J. (2019). Hamiltonian neural networks. _arXiv_
_preprint_ _arXiv:1906.01563_ .


Hairer, E. (1994). Backward analysis of numerical integrators and symplectic methods.
_Annals_ _of_ _Numerical_ _Mathematics_, 1:107–132. ID: unige:12640.


Hairer, E., Lubich, C., and Wanner, G. (2002). _Geometric Numerical Integration:_ _Structure-_
_Preserving_ _Algorithms_ _for_ _Ordinary_ _Differential_ _Equations_ . Springer series in computational mathematics. Springer.


Hairer, E., Nørsett, S. P., and Wanner, G. (2008). _Solving_ _Ordinary_ _Differential_ _Equations_
_I:_ _Nonstiff_ _Problems_ . Springer Series in Computational Mathematics. Springer Berlin
Heidelberg.


Hairer, E. and Wanner, G. (2013). _Solving_ _Ordinary_ _Differential_ _Equations_ _II:_ _Stiff_ _and_ _Dif-_
_ferential_ _-_ _Algebraic_ _Problems_ . Springer Series in Computational Mathematics. Springer
Berlin Heidelberg.


Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. _Neural_ _Computation_,
9(8):1735–1780.


Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. _arXiv_
_preprint_ _arXiv:1412.6980_ .


10


Published as a conference paper at ICLR 2020


Lambert, J. D. (1991). _Numerical_ _Methods_ _for_ _Ordinary_ _Differential_ _Systems:_ _The_ _Initial_
_Value_ _Problem_ . John Wiley & Sons, Inc., New York, NY, USA.


Leimkuhler, B. and Reich, S. (2005). _Simulating_ _Hamiltonian_ _Dynamics_ . Cambridge Monographs on Applied and Computational Mathematics. Cambridge University Press.


Lutter, M., Ritter, C., and Peters, J. (2019). Deep lagrangian networks: Using physics as
model prior for deep learning. In _International_ _Conference_ _on_ _Learning_ _Representations_ .


McLachlan, R. I., Perlmutter, M., and Quispel, G. R. W. (2004). On the nonlinear stability
of symplectic integrators. _BIT_ _Numerical_ _Mathematics_, 44(1):99–117.


Niu, M. Y., Horesh, L., and Chuang, I. (2019). Recurrent neural networks in the eye of
differential equations. _arXiv_ _preprint_ _arXiv:1904.12933_ .


Peifer, M. and Timmer, J. (2007). Parameter estimation in ordinary differential equations
for biochemical processes using the method of multiple shooting. _The_ _Institution_ _of_ _En-_
_gineering_ _and_ _Technology,_ _Systems_ _Biology_ .


Sanchez-Gonzalez, A., Bapst, V., Cranmer, K., and Battaglia, P. (2019). Hamiltonian graph
networks with ode integrators. _arXiv_ _preprint_ _arXiv:1909.12790_ .


Stapor, P., Fröhlich, F., and Hasenauer, J. (2018). Optimization and profile calculation of
ODE models using second order adjoint sensitivity analysis. _Bioinformatics_, 34(13):i151–
i159.


Tao, M. (2016). Explicit symplectic approximation of nonseparable hamiltonians: Algorithm
and long time performance. _Phys._ _Rev._ _E_, 94:043303.


Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). Algorithm 778: L-bfgs-b: Fortran
subroutines for large-scale bound-constrained optimization. _ACM_ _Trans._ _Math._ _Softw._,
23(4):550–560.


11


Published as a conference paper at ICLR 2020


A Experiment setup


A.1 The spring-chain experiment


We set ∆ _t_ = 0 _._ 1. The ground truth trajectories in both training and testing are simulated
by the leapfrog integrator using ∆ _t_ _[′]_ = 0 _._ 001 and coarsened into time-grids of 0 _._ 1 with a
factor of 100, since simulating with a much smaller time-step leads to much more accurate
solution, which we will treat as the ground truth solution.

The O-NET that represents _fθ_ ( _p, q_ ) is a one-hidden-layer MLP with 40 input units, 2048
hidden units and 40 output units. The H-NET that represents _Hθ_ ( _p, q_ ) = _Kθ_ 1( _p_ ) + _Vθ_ 2( _q_ )
consists of two one-hidden-layer MLPs, one for _Kθ_ 1 and the other for _Vθ_ 2. Each of the MLPs
have 20 input units, 2048 hidden units and 1 output unit. The vanilla RNN and LSTM
models also have hidden states of size 2048. Implemented in PyTorch, the models are trained
over 1000 epochs with the Adam optimizer (Kingma and Ba, 2014) with initial learning rate
0.001 and using the _ReduceLROnPlateau_ scheduler [4] with patience 15 and factor 0.7.


A.2 The three-body experiment


The ground truth trajectories are simulated by SciPy’s `solve_ivp` adaptive solver [5] with
method RK45. We coarse-grain the simulated ground truth trajectories into time-steps of
∆ _t_ = 1, so that the models developed in section 4 are numerically integrated with time-step
∆ _t_ = 1 in both training and testing. We intentionally set the time-step to be relatively
large, so that it becomes interesting to compare these models with a baseline method of
simulating the true equations with time-step ∆ _t_ = 1. In addition, the training data consist
of 100 sample trajectories of length 10 _·_ ∆ _t_ = 10, which are then turned into 900 trajectories of
length 2 and 600 trajectories of length 5, respectively for single-step and recurrent training,
in the same way as for the spring-chain experiments above.

The O-NET that represents _fθ_ ( _p, q_ ) is a three-hidden-layer MLP with 12 input units,
512 hidden units in each hidden layer and 12 output units. The H-NET that represents
_Hθ_ ( _p, q_ ) = _Kθ_ 1( _p_ ) + _Vθ_ 2( _q_ ) consists of two three-hidden-layer MLPs, one for _Kθ_ 1 and the
other for _Vθ_ 2. Each of the MLPs have 6 input units, 512 hidden units in each hidden layer
and 1 output unit. The vanilla RNN and LSTM models also have hidden states of size 512.
Implemented in PyTorch, the models are trained over 1000 epochs with the Adam optimizer
with initial learning rate 0.0003 and using the _ReduceLROnPlateau_ scheduler with patience
15 and factor 0.7.


A.3 The heavy billiard experiment


The full image has size 128x128 pixels. The thickness of the wall is 12 pixels on each of the
four sides, which leaves the free space of size 104x104 pixels in the middle for the billiard to
move within. The billiard has size 3x3 pixels.

The O-NET that represents _fθ_ ( _p, q_ ) is a one-hidden-layer MLP with 4 input units, 32 hidden
units and 4 output units. The H-NET that represents _Hθ_ ( _p, q_ ) = _Kθ_ 1( _p_ ) + _Vθ_ 2( _q_ ) consists
of two one-hidden-layer MLPs, one for _Kθ_ 1 and the other for _Vθ_ 2. Each of the MLPs have
2 input units, 32 hidden units and 1 output unit. The vanilla RNN model also has hidden
states of size 32. For the rebound module, _n_ ¯ is computed as the normalized output of a
two-hidden-layer MLP, with 200 input units, 128 units in the first hidden layer, 32 units
in the second hidden layer and 2 output units. _α_ is also computed using a two-hiddenlayer MLP, sharing the first hidden layer units with the MLP for _n_ ¯, and having 32 units
in the second hidden layer and 1 output unit. _γ_ is computed by passing through sigmoid
the output of a two-hidden-layer MLP, with 4 input units, 16 units in each hidden layer
and 1 output unit. All of the activation functions are tanh except for the hidden-to-output
activation in the MLP for _α_, where ReLU is used. Implemented in PyTorch, the models
are trained over 1500 epochs with the Adam optimizer with initial learning rate 0.005 and
using the _ExponentialLR_ scheduler [6] with decay factor 0.99 until the learning rate reaches
0.0001. We set ∆ _t_ = 0 _._ 1, and use 5000 trajectories of length 10 _·_ ∆ _t_ = 1 as training data,
and 32 trajectories of length 60 _·_ ∆ _t_ = 6 as testing data.


4
```
  https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
```

5
```
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
```

6
```
  https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ExponentialLR

```

12


Published as a conference paper at ICLR 2020


B The Maximum Likelihood Estimation perspective


In the presence of noise, we can interpret the learning problem described in section 3.3 above
from the perspective of maximum likelihood inference, which also provides justification for
treating the initial states as trainable parameters. We define models as follow:

_z_ ˆ _i_ ( _θ_ ) = _Integrator_ (ˆ _z_ 0 = _z_ 0 _, fθ, {ti}_ _[T]_ _i_ =0 [)]



1
_q_ ( _zi_ ; _θ_ ) = �(2 _π_ ) _[d]_ _σ_ [2] _[d]_ _[e][−∥][z][i][−][z]_ [ˆ] _[i]_ [(] _[θ]_ [)] _[∥]_ 2 [2] _[/]_ [(2] _[σ]_ [2][)]



(10)



~~_√_~~
(



_P_ ( _{zi}_ _[n]_ _i_ =1 _[|][θ]_ [) =]



_n_





- _q_ ( _zi_ ; _θ_ ) = ~~_√_~~ 1

_i_ =1 ( 2



2 _πσ_ [2] ) _[nd]_



_n_

- _e_ _[−∥][z][i][−][z]_ [ˆ] _[i]_ [(] _[θ]_ [)] _[∥]_ 2 [2] _[/]_ [(2] _[σ]_ [2][)] _,_


_i_ =1



and _L_ ( _θ|{zi}_ _[n]_ _i_ =1 [) =] _[ P]_ [(] _[{][z][i][}][n]_ _i_ =1 _[|][θ]_ [)][ is the likelihood function given the time-series data] _[ {][z][i][}][n]_ _i_ =1 [.]
Note that this model assumes independence between _zi_ and _zj_ for _i ̸_ = _j_ once _θ_ is fixed.

If we are to perform maximum likelihood inference, we arrive at the following:



max : log _L_ ( _θ|{zi}_ _[n]_ _i_ =1 [) =] _[ −]_ _[nd]_ �2 _πσ_ [2][�] _−_ 1
_θ_ 2 [log] 2 _σ_ [2]



_n_

- _∥zi −_ _z_ ˆ _i_ ( _θ_ ) _∥_ [2] 2 _[,]_ (11)


_i_ =1



which is equivalent to



min
_θ_



_n_

- _∥zi −_ _z_ ˆ _i_ ( _θ_ ) _∥_ [2] 2 (12)


_i_ =1



This provides a motivation for using the _L_ [2] loss, as we did in the experiments.

So far, we consider _θ_ as the only parameter of the model defined by equations 10, and
therefore the only argument of the likelihood function, while _z_ ˆ0 is fixed to be the observed
initial state _z_ 0. As a generalization, we can consider a strictly larger family of models by
allowing _z_ 0 to vary as well. In this way, we treat both _θ_ and _z_ 0 as the parameters in the
model and therefore arguments of the likelihood function that we optimize on. In other
words, the model becomes

_z_ ˆ _i_ ( _θ,_ ˆ _z_ 0) = _Integrator_ (ˆ _z_ 0 _, fθ, {ti}_ _[T]_ _i_ =0 [)]



1
_q_ ( _zi_ ; _θ,_ ˆ _z_ 0) = �(2 _π_ ) _[d]_ _σ_ [2] _[d]_ _[e][−∥][z][i][−][z]_ [ˆ] _[i]_ [(] _[θ,][z]_ [ˆ][0][)] _[∥]_ 2 [2] _[/]_ [(2] _[σ]_ [2][)]



(13)



~~_√_~~
(



_P_ ( _{zi}_ _[n]_ _i_ =1 _[|][θ,]_ [ ˆ] _[z]_ [0][) =]



_n_





- _q_ ( _zi_ ; _θ_ ) = ~~_√_~~ 1

_i_ =1 ( 2



2 _πσ_ [2] ) _[nd]_



_n_

- _e_ _[−∥][z][i][−][z]_ [ˆ] _[i]_ [(] _[θ]_ [)] _[∥]_ 2 [2] _[/]_ [(2] _[σ]_ [2][)] _,_


_i_ =1



and the optimization problem becomes



min
_θ,z_ ˆ0



_n_

- _∥zi −_ _z_ ˆ _i_ ( _θ,_ ˆ _z_ 0) _∥_ [2] 2 _[,]_ (14)


_i_ =1



which justifies optimizing over the initial states _p_ 0, _q_ 0 in addition to the neural network
parameters _θ_ as described in the previous section.

Such an interpretation is similar to approaches for parameter estimation in the literature
of inverse problems and systems biology, though in those cases the parameters of interest
appear directly in ODEs instead of via neural networks (Peifer and Timmer, 2007; Stapor
et al., 2018). In particular, jointly optimizing the parameters in the model as well as the
initial value is called the initial value approach. However, despite the success we demonstrate
in section 4.2, two difficulties of this approach have been pointed out: 1) The optimization
could converge to local minima; 2) The numerical solution of the ODE can be unstable
(Peifer and Timmer, 2007). As explained in section 4.1, using HNN together with the
leapfrog integrator mitigates the second issue. But what about the first issue? In particular,
even if we assume that the optimization of the neural network can work “magically” well
and do not suffer from bag local minima, what about optimizing the initial value _z_ ˆ0?


C Symplecticness and initial-state-optimization convexity


The success of optimizing on the initial state of the system in addition to the recurrent
H-NET and O-NET models as described in section 4.2 raises the following question: If we


13


Published as a conference paper at ICLR 2020


already have a relatively well-trained H-NET or O-NET, is the optimization on the initial
values convex? We formalize the question below and provide a heuristic answer.

For simplicity, we restrict our attention to autonomous ODEs, which means that the function
_f_ in _[dz]_ _dt_ [=] _[ f]_ [(] _[z]_ [)][ does not depend on] _[ t]_ [.] [Assuming existence and uniqueness of solutions, there]

exists a function _φt_ that maps each initial state _z_ ˆ0 to the state of the system after evolving
from _z_ ˆ0 for time _t_, _φt_ (ˆ _z_ 0). This function is usually called the flow map. Flow maps have also
been defined for numerical solutions of ODEs, by letting _φt_ (ˆ _z_ 0) = _Integrator_ ( _z_ 0 _, f, {ti}_ _[T]_ _i_ =0 [)]
with _t_ 0 = 0. We can extend this definition to all the trainable models we have considered,
including the models based on O-NET and H-NET by defining _φt_ (ˆ _z_ 0) to be the state of the
system after letting the system evolve from initial state _z_ ˆ0 for time _t_, for suitable choices of
_t_ . For example, for O-NET, we have _φt_ (ˆ _z_ 0) = _Integrator_ ( _z_ 0 _, fθ, {ti}_ _[T]_ _i_ =0 [)][.]
Suppose we impose an L2 loss on _φt_ (ˆ _z_ 0), _et_ (ˆ _z_ 0) = _∥φt_ (ˆ _z_ 0) _−_ _zt∥_ [2] 2 [,] [where] _[z][t]_ [corresponds] [to]
the observed data at time _t_ . The question is, is _et_ ( _z_ ) a (perhaps locally) convex function of
_z_, for what functions and numerical integrators? To understand convexity, we compute the
gradient and the Hessian as follow.


_∂_
(15)
_∂z_ _[e][t]_ [(] _[z]_ [) = 2(] _[φ][t]_ [(] _[z]_ [)] _[ −]_ _[z][t]_ [)][⊺] _[·][ F][t]_ [(] _[z]_ [)]

_∂_ [2]

_∂z_ [2] _[e][t]_ [(] _[z]_ [) = 2(] _[φ][t]_ [(] _[z]_ [)] _[ −]_ _[z][t]_ [)] _[ ·]_ [(3)] _[ G][t]_ [(] _[z]_ [) +] _[ F][t]_ [(] _[z]_ [)][⊺] _[·][ F][t]_ [(] _[z]_ [)] _[,]_ (16)

where _Ft_ ( _z_ ) is the Jacobian matrix of the flow map, defined as _Ft_ ( _z_ ) _ij_ = _∂z∂j_ [(] _[φ][t]_ [(] _[z]_ [)] _[i]_ [)][,] [and]
_Gt_ ( _z_ ) is a third-order tensor contains the second order derivatives of the flow map, defined
_∂_ [2]
as _Gt_ ( _z_ ) _ijk_ = _∂zizj_ [(] _[φ][t]_ [(] _[z]_ [)] _[k]_ [)][.] [We] [use] _[·]_ [(3)] [to] [denote] [the] [dot] [product] [in] [the] [third] [dimension.]

_Ft_ ( _z_ ) [⊺] _· Ft_ ( _z_ ) is symmetric positive semidefinite for any matrix _Ft_ ( _z_ ). If _φt_ corresponds to
either the exact flow map of a Hamiltonian system or the flow of a symplectic integrator,
such as the leapfrog integrator, applied to a Hamiltonian system, then _Ft_ ( _z_ ) is a symplectic
matrix, implying that det( _Ft_ ( _z_ )) = 1. Hence, det( _Ft_ ( _z_ ) [⊺] _· Ft_ ( _z_ )) = 1, which further implies
that _Ft_ ( _z_ ) [⊺] _·Ft_ ( _z_ ) is a positive definite matrix. Therefore, non-rigorously, when _∥φt_ ( _z_ ) _−zt∥_ 2
is small and so the first term on the right hand side of equation 16 is negligible compared
_∂_ [2]
to the least eigenvalue of _Ft_ ( _z_ ) [⊺] _· Ft_ ( _z_ ), the entire Hessian matrix _∂z_ [2] _[e][t]_ [(] _[z]_ [)] [is] [also] [positive]
definite, implying strong convexity of the optimization problem.
If _φt_ is the exact flow map, then _∥φt_ ( _z_ ) _−_ _zt∥_ being small means that noise in the data
is small. If _φt_ is the flow map of a learned model, then it means that we have a model
close to the true underlying system in addition to not having too much noise in the data.
Translating back to the learning problem, we see that, heuristically, when the model we use
is close to symplectic, which is likely if the underlying system is a Hamiltonian system, and
trained to be close enough to the true underlying system, and the noise in the data is small
enough, then the optimization problem on the initial state is strongly convex.


14


Published as a conference paper at ICLR 2020


D Additional plots of the spring-chain experiments


D.1 Noiseless data (section 4.1)



4


2


0


2


4


4


2


0


2


4


4


2


0


2


4


4


2


0


2


4


4


2


0


2


4



time step



4


2


0


2


4


4


2


0


2


4


4


2


0


2


4


4


2


0


2


4


4


2


0


2


4



time step





Figure 5: Extension of Figure 1 to 10 masses on the chain (1st being the closest to one end,
and 10th being in the center).


15


Published as a conference paper at ICLR 2020


D.2 Noisy data (section 4.2)


4


2



Single-step E-E O-NET
Noiseless dynamic
Observations



0


2


4


4


2


0


2


4


4


2


0


2


4



time step


Figure 6: Single-step E-E O-NET


(HNN) Single-step E-E H-NET
Noiseless dynamic
Observations


time step


Figure 7: Single-step E-E H-NET


Single-step L-L O-NET
Noiseless dynamic
Observations


time step


Figure 8: Single-step L-L O-NET


16


Published as a conference paper at ICLR 2020


4


2



Single-step L-L H-NET
Noiseless dynamic
Observations



0


2


4


4


2


0


2


4


4


2


0


2


4



time step


Figure 9: Single-step L-L H-NET


Recurrent E-E O-NET
Noiseless dynamic
Observations


time step


Figure 10: Recurrent E-E O-NET


Recurrent E-E H-NET
Noiseless dynamic
Observations


time step


Figure 11: Recurrent E-E H-NET


17


Published as a conference paper at ICLR 2020


4


2



Recurrent L-L O-NET
Noiseless dynamic
Observations



0


2


4


4


2


0


2


4


4


2


0


2


4



time step


Figure 12: Recurrent L-L O-NET


(SRNN) Recurrent L-L H-NET
Noiseless dynamic
Observations


time step


Figure 13: (SRNN) Recurrent L-L H-NET


Recurrent E-E O-NET w/ ISO
Noiseless dynamic
Observations


time step


Figure 14: Recurrent E-E O-NET w/ ISO


18


Published as a conference paper at ICLR 2020


4


2



Recurrent E-E O-NET w/ ISO
Noiseless dynamic
Observations



0


2


4


4


2


0


2


4



time step


Figure 15: Recurrent E-E H-NET w/ ISO


Recurrent L-L O-NET w/ ISO
Noiseless dynamic
Observations


time step


Figure 16: Recurrent L-L O-NET w/ ISO


(SRNN-ISO) Recurrent L-L H-NET w/ ISO
Noiseless dynamic
Observations



4


2


0


2


4



time step


Figure 17: (SRNN-ISO) Recurrent L-L H-NET w/ ISO


19


Published as a conference paper at ICLR 2020


4


2



0


2


4


4


2


0


2


4



time step


Figure 18: Vanilla RNN


time step


Figure 19: LSTM


20



Vanilla RNN
Noiseless dynamic
Observations


LSTM
Noiseless dynamic
Observations


