# Revisiting Overestimation Bias Problem of Q-learning: Settling Large Discrete Action Space via Action Intersection

Pu Li<sup>a</sup>, Tao Tan<sup>b</sup>, Hong Xie<sup>b</sup>, Xiaoyu Shi<sup>a</sup> and Mingsheng Shang<sup>a</sup>

<sup>a</sup>Chongqing Institute ofGreen and Intelligent Technology, Chongqing, China

<sup>b</sup>University of Science and Technology of China, Hefei, Anhui, China

A R T I C L E I N F O

Keywords: Coupling Paradigm Decoupling Paradigm Semi-decoupling Paradigm Action Intersection

## A BS T R AC T

This paper considers the overestimation bias problem of Q-learning in the setting of a large action space, for the purpose of relieving the bottleneck of existing methods. We find that the large action space increases the randomness in Q-value estimation. The randomness makes two paradigms that drive the major literature on the overestimation problem have their own bottlenecks: the coupling paradigm, i.e., the optimal action and its Q-value are estimated with the same Q-function, always has a positive bias. This is because randomness leads to some actions having abnormally high estimated values than their true values, and the coupling methods prefer these actions. The decoupling paradigm, i.e., the optimal action and its Q-value are estimated with two independent Q-functions, always has a negative bias. This is because randomness increases the estimation gap between the two independent Q-tables for the same action. This paper shows that action intersection can be a simple yet powerful strategy to relieve these bottlenecks. The action intersection strategy enables semi-decoupling via two designs: (1) it allows two Q-functions to share a certain fraction of trajectory data; (2) if a data sample is shared, each Q-function is updated using the coupling paradigm; otherwise, using the decoupling paradigm. Two properties make the action intersection strategy powerful: (1) attaining a large bias range, i.e., varying the data sharing fraction, the estimation bias varies from underestimating to overestimating; (2) fine granularity: the action intersection size can be made arbitrarily finer to enable finer control. We consider two experiment settings, i.e., tabular and deep RL, deep RL experiments show that our method outperforms several SOTA baselines drastically; tabular experiments reveal why our method can achieve superior performance. We believe that our work opens a new door for mitigating the overestimation bias of Q-learning.

## 1. Introduction<sub>[</sub>

Q-learning Watkins and Dayan (1992) is one of the most widely adopted and studied Reinforcement Learning (RL) algorithms Chu, Li, Liu, Hu, Li and Wang (2022)Zhang, Zou, Lai and Xu (2023)Zhao, Wang and Wang (2023). However, Q-learning sufers from the overestimation biasThrun and Schwartz (2014)Hasselt (2010). Previous work shows that in Q-learning, function approximators induce noise in the output predictions, and such noise can lead to a systematic overestimation of Q-values. Note that the noisy estimates of Q-values are caused by stochastic and unknown rewards and state transitions. Double Q-learning is a classical algorithm to remove the overestimation bias of Q-learning, but it may lead to the underestimation bias. Due to the simplicity and convergence guarantees under some common assumptions Melo (2001), these value-estimation algorithms, such as Q-learning and Double Q-learning, have become a practical tool, for instance, as a value estimator in RL fine-tuning Large Language ModelsZhai, Bai, Lin, Pan, Tong, Zhou, Suhr, Xie, LeCun, Ma et al. (2024)Rafailov, Hejna, Park and Finn (2024), where the action space is large. Considering that other potential application scenarios also feature large action spaces (e.g., recommender systems), we ask the following question:Will large action spaces become a bottleneckfor bias control?

To answer this question, we conduct an experiment to investigate how the estimation bias changes as the number of arms increases. The result shows that indeed large action space amplifies the estimation bias. Example 1 illustrates our empirical study.

Example 1. Consider a multi-armed bandit setting, which includes one state � with � actions $\{ a _ { 1 } , . . . , a _ { N } \}$ . Following previous work Zhang, Pan and Kochenderfer (2017), each action $a _ { i }$ returns a reward obeys $\mathcal { N } ( 0 , 1 0 ^ { 2 } )$ . We set the discountfactor � = 0.95, and the maximum expected Q-value is 0. Our experiments investigate how the estimation error of these algorithms varies as the action space size increases. Figure 1(A) shows the maximum Q-value of Q-learning algorithm after a 10,000-step update with diferent numbers ofarms. We can observe that when the number ofarms is 80, the curve ofQ-learning is at the top, and when the number ofarms is 20, the curve is at the bottom. And all curves ofQ-learning are above the curve ofunbiased curve. This implies that Q-learning has a positive estimation bias, and the estimation bias increases as the size ofthe action space increases. Figure 1(B) shows the maximum Q-value ofthe Double Q-learning algorithm after a 10,000-step update with diferent numbers of arms. We can observe that when the number of arms is 80, the curve of Double Q-learning is at the bottom, and when the number of arms is 20, the curve is at the top. And all curves of Double Q-learning are under the curve of the unbiased curve. This implies that Double Q-learning has a negative estimation bias, and the estimation bias increases as the size of the action space increases. Figure 1(C) shows the relationship between the size ofthe action space and the maximum Q-values. One can observe that as the action space increases, the length of all bars rises, i.e., the maximum Q-value estimated by each method progressively deviatesfrom the optimal Q-value of0. This implies that current methods suferfrom estimation bias in environments with large action spaces, and the estimation bias and the size ofthe action space have a positive correlation. More experiment details refer to Section 7.1.

![](images/c88a13cf6ccc517ab1422e8d775d664dd0397d6f731e1e5e60d86fe64e072595.jpg)  
(A) Q learning

![](images/aa47d4ac6dbd637c6630746eacfa502bcd4385e54d311fcf9c85b0e30e50e678.jpg)  
(B) Double Q-learning

![](images/4f63f54dc459c4e1b5440fdd8d90dfb10ac4e45b60a7a7460de1b61538d824d5.jpg)  
(C) Comparison  
Figure 1: The maximum Q-value of Q-learning and Double Q-learning in a large action space.

Existing estimation bias control methods also face the challenge of the large action space. To analyze the reason, we categorize existing methods into two classes, coupling methods and decoupling methods, for separate discussion. Coupling methods evaluate the maximum expected Q-value using a single Q-function, i.e., both the optimal action and its Q-value are obtained from the same Q-function, and always have a positive bias. Q-learning, Maxmin Qlearning, Averaged Q-learning, and Order Q-learning all belong to coupling methods. More specifically, in the setting of a large action space, the randomness in Q-values increases with the number of actions, leading to some actions having abnormally higher estimated values than their true values. Coupling-based methods consistently select these overestimated actions to update the estimator, thereby introducing positive bias. Decoupling methods evaluate the maximum expected Q-value using two independent Q-functions, i.e., the optimal action and its Q-value are estimated by two independent Q-functions, and always have a negative bias. Double Q-learning and EBQL belong to decoupling methods. In this setting of a large action space, the estimation gap between the two independent Q-tables for the same action widens as the number of actions increases. The estimated value given by Q-table 2 for the optimal action selected by Q-table 1 may be abnormally low relative to the true value, thus resulting in negative bias.

According to the above analysis, the core challenge in the setting of a large action space is the high randomness in selecting the optimal action and estimating its value. To alleviate this problem, we propose an action intersection strategy based on Double Q-learning, and obtain Action Intersection Double Q-learning (AIDQ). The core idea is maintaining two Q-tables (e.g., $Q ^ { 1 }$ and $Q ^ { 2 } )$ , and using $Q ^ { 1 }$ to select a set of optimal actions(e.g., ���� actions), using $Q ^ { 2 }$ to identify the optimal action in the set by estimating their Q-values. Note that selecting ���� actions is a simple yet efective strategy in the large action space setting, because it decreases the randomness in selecting the optimal action and estimating its value. When � = 1, our method turns to Double Q-learning, which has a negative bias. When $K = | { \mathcal { A } } |$ , where || is the size of the action space, our method turns to two Q-tables alternately performing

Q-learning updates, which has a positive bias. This feature helps to enable fine-grained control over a large bias range,   
from negative bias to positive bias in the large action space setting. In summary, this paper makes the following contributions:

• To control the overestimation bias in a large action space setting, i.e., the estimation bias is always positive or negative, this paper proposed an action intersection strategy, i.e., control the estimation bias to vary continuously from negative bias to positive bias.

• Based on the Double Q-learning, we introduce an action intersection Double Q-learning, which is a simple yet powerful method that enables fine-grained control over a large bias range in the setting of a large action space.

• We evaluate our method in both tabular and deep RL settings. Extensive experiment results show that our method outperforms SOTA baselines drastically in diferent settings. More specifically, our method improves the average return per episode over SOTA baselines by at least 53.21% in the Asterix environment.

## 2. Related Work

The overestimation problem of Q learning was first identified and researched by Sebastian Thrun et al. in Thrun and Schwartz (2014). Specifically, this work shows that function approximators induce noise in the output predictions, and such noise can lead to a systematic overestimation of Q-values. As a result of this overestimation bias, the agent could fail to learn an optimal policy in certain situations. To address the overestimation problem, many methods were proposed. We divide works aiming to address overestimation bias in Q learning into two categories: coupling method and decoupling method.

## 2.1. Coupling Method

They evaluate the maximum expected Q-value using a single Q-function, i.e., both the optimal action and its Q-value are obtained from the same Q-function, always has a positive bias. Double Q-learningHasselt (2010) points out that the max operator leads to the significant issue of overestimation. Motivated by this perspective, Softmax Q-learning Song, Parr and Carin (2019), Soft Mellowmax Q-learning Song et al. (2019) and Mellowmax Q-learning Asadi and Littman (2017) use the softmax operator to replace the max operation in the generalized value iteration framework. And they control the estimation bias via the softmax temperature hyperparameter. Their theoretical results quantify how the softmax Bellman operator reduces the overestimation bias. Also motivated by the perspective that the max operator in Q-learning can lead to overestimation, Averaged Q-learning Anschel, Baram and Shimkin (2017) suggests a diferent solution to the overestimation phenomenon. Averaged Q-learning points out that the magnitude of the overestimation bias is controlled by the target approximation error variance. Thus, Averaged Q-learning builds an averaged Q-estimator over multiple Q-functions to reduce the target approximation error variance and uses this averaged Q-estimator to estimate the maximum expected Q-value. The estimation bias of Averaged Q-learning is inversely proportional to the number of Q-functions. However, with a finite number of Q-functions, it always has a positive bias. Maxmin Q-learning Lan, Pan, Fyshe and White (2020) builds a minimum Q-estimator over multiple Q-functions, and uses this minimum Q-estimator to estimate the maximum expected Q-value. The estimation bias of Maxmin Q-learning is inversely proportional to the number of Q-functions. However, the bias control is coarse-grained and discrete, i.e., it depends on the number of Q-functions. AdaEQ Wang, Lin and Zhang (2021) is a natural extension of Maxmin Q-learning to adaptively select the number of Q-functions and aims to remove the estimation bias. The theoretical finding shows that approximation error characterization can serve as the feedback for flexibly controlling the ensemble size. Doubly Bounded Q-Learning Ren, Zhu, Hu, Han, Chen and Zhang (2021) leverages an abstracted dynamic programming as a second value estimator to reduce the underestimation bias and to eliminate non-optimal fixed points of Double Q-learning. Order Q-learning Tan, Xie and Lian (2024a) builds an order Q-estimator over multiple Q-functions, and uses this order Q-estimator to estimate the maximum expected Q-value. However, the bias control is discrete, i.e., it depends on the number of Q-functions and the index of the order statistic function.

## 2.2. Decoupling Methods

They evaluate the maximum expected Q-value using two independent Q-functions, i.e., the optimal action and its Q-value are estimated by two independent Q-functions, always has a negative bias. Double Q-learning Hasselt (2010) uses two independent Q-functions to evaluate the maximum expected Q-value, where one is used to select the optimal action and the other to estimate its Q-value. It removes the overestimation bias of Q-learning, but leads to the underestimation bias. Note that this underestimation bias can cause a slower learning speed and a larger performance penalty than the overestimation bias Abliz and Ying (2022). EBQL Peer, Tessler, Merlis and Meir (2021) is a natural extension of Double Q-learning to ensembles; this works aim to address the underestimation bias of Double Q-learning. It maintains � Q-functions. For each update step, it randomly chooses a Q function to select the optimal action and uses all the other functions to estimate the Q-value of the optimal action. Theoretical analysis proves that EBQL-like updates yield lower MSE when estimating the maximal mean of a set of independent random variables. However, it always has a negative bias. AC-CDQ Jiang, Xie and Yang (2021) attempts to balance the overestimation bias and the underestimation bias via a clipping operation. However, the bias control is coarse-grained and discrete, i.e., it depends on the number of action candidates.

There are also some other works on the setting of a large action space. Dulac-Arnold, Evans, van Hasselt, Sunehag, Lillicrap, Hunt, Mann, Weber, Degris and Coppin (2015) proposed approach leverages prior information about the actions to embed them in a continuous space upon which it can generalize. And then uses an approximate nearest neighbor search to find the set of closest discrete actions in logarithmic time. Weisz, Budzianowski, Su and Gašić (2018) examines the application of reinforcement learning to dialogue policy optimisation in a spoken dialogue system, which has a relatively large action set. The key algorithm is ACERWang, Bapst, Heess, Mnih, Munos, Kavukcuoglu and De Freitas (2016)Munos, Stepleton, Harutyunyan and Bellemare (2016), which combines a lot of technologies such as actor-critic methods, of-policy reinforcement learning with experience replay, and various methods aimed at reducing the bias and variance of estimators. Although this algorithm has achieved relatively good performance, it is overly complex and lacks theoretical analysis. Majeed and Hutter (2021) addresses the large action-space problem by sequentializing actions, which can reduce the action space size significantly, even down to two actions. However, it leads to an increased planning horizon, thereby increasing costs. Ming, Gao, Liu and Zhao (2023) proposes to decompose the complex decision task in a large action space into multiple decision sub-tasks in small action subsets, using a rule-based action division method. However, these methods are not specifically designed to address the overestimation problem in environments with large action spaces. Additionally, these methods primarily focus on the deep learning domain, often involving complex algorithms and models, and ofer limited theoretical analysis of estimation errors.

This paper proposes a new semi-decoupling method, which evaluates the maximum expected Q-value using two dependent Q-functions, i.e., the optimal action and its Q-value are estimated by two dependent Q-functions. We build the dependence between two Q-functions via an action intersection strategy, and enable fine-grained and continuous control over a large bias range, i.e., varying the data sharing fraction smoothly shifts the bias from negative to positive.

## 3. Preliminary

We consider a Markov Decision Process (MDP Chen, Wang, Zhou and Ross (2021)) defined by the tuple $( S , \mathcal { A } , P , R , \gamma )$ , where  is the state space,  is the action space, $P ( s ^ { \prime } \mid s , a )$ is the state transition probability, $R ( s , a )$ is the reward function, and $\gamma \in ( 0 , 1 )$ is the discount factor. At each time step $t \in \mathbb { N } .$ , the agent observes the current state $s _ { t } \in S _ { : }$ , selects an action $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ according to a policy �, transitions to the next state $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } )$ , and receives a reward $r _ { t } = R ( s _ { t } , a _ { t } )$ . The objective of the agent is to learn an optimal policy that maximizes the expected discounted return.

## 3.1. Q-learning

Q-learning Watkins and Dayan (1992) maintains a single Q-table denoted by $Q _ { t } ( s , a )$ . It first initializes the $\mathrm { Q } \mathrm { - }$ table $Q _ { 0 } ( s , a )$ , and then collects the interaction sample $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } )$ by �-greedy policy Ghasemipour, Schuurmans and Gu (2021), where $\epsilon ~ \in ~ [ 0 , 1 ]$ . Note that under the �-greedy policy, the agent selects a greedy action from arg $\operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { t } ( s _ { t } , a )$ with probability $( 1 - \epsilon )$ , and selects a random action from  with probability �. In each time step �, the updating rule of Q-learning is:

$$
\left\{ \begin{array} { l l } { a _ { t + 1 } ^ { * } = \arg \operatorname* { m a x } _ { a _ { t + 1 } } Q _ { t } ( s _ { t + 1 } , a _ { t + 1 } ) , } \\ { Y _ { t } = r _ { t } + \gamma Q _ { t } ( s _ { t + 1 } , a _ { t + 1 } ^ { * } ) , } \\ { Q _ { t + 1 } ( s _ { t } , a _ { t } ) = ( 1 - \alpha ) Q _ { t } ( s _ { t } , a _ { t } ) + \alpha Y _ { t } , } \end{array} \right.\tag{1}
$$

where � is the learning rate, $Y _ { t }$ is the TD target Q-value for $Q _ { t } ( s _ { t } , a _ { t } )$

Equation (1) illustrates that, during the Q-learning update, both the optimal action selection and its value evaluation rely on the same Q-table. This coupling mechanism is the root cause of the overestimation bias in Q-learning (i.e., $\begin{array} { r } { { \mathbb { E } } [ \operatorname* { m a x } _ { a } Q ( s , a ) ] \ \geq \ \operatorname* { m a x } _ { a } { \mathbb { E } } [ Q ( s , a ) ] ) } \end{array}$ ), which can be theoretically explained in the Lemma of Thrun and Schwartz (2014).

## 3.2. Double Q-Learning

Double Q-learning Hasselt (2010) maintains two independent Q-tables denoted by $Q _ { t } ^ { A } ( s , a ) , Q _ { t } ^ { B } ( s , a )$ . Diferent from Q-learning, Double Q-learning uses the �-greedy policy with arg ma $\mathtt { { x } } _ { a \in \mathcal { A } } \frac { Q _ { t } ^ { A } ( s _ { t } , a ) + Q _ { t } ^ { B } ( s _ { t } , a ) } { 2 }$ to sampling. In each time step �, Double Q-learning random selects a Q-table, such as $Q _ { t } ^ { A }$ , to update as:

$$
\left\{ \begin{array} { l l } { a _ { t + 1 } ^ { * } = \arg \operatorname* { m a x } _ { a _ { t + 1 } } Q _ { t } ^ { A } ( s _ { t + 1 } , a _ { t + 1 } ) , } \\ { Y _ { t } ^ { A } = r _ { t } + \gamma Q _ { t } ^ { B } ( s _ { t + 1 } , a _ { t + 1 } ^ { * } ) , } \\ { Q _ { t + 1 } ^ { A } ( s _ { t } , a _ { t } ) = ( 1 - \alpha ) Q _ { t } ^ { A } ( s _ { t } , a _ { t } ) + \alpha Y _ { t } ^ { A } , } \end{array} \right.\tag{2}
$$

where $Y _ { t } ^ { A }$ is the TD target Q-value for $Q _ { t } ^ { A } ( s _ { t } , a _ { t } )$

Equation (2) illustrates that, during the Double Q-learning update, the optimal action selection and its value evaluation rely on two independent Q-tables. This decoupling mechanism is the root cause of the underestimation bias in Double Q-learning, which can be theoretically explained in Lemma 1 of Hasselt (2010).

## 4. Method

Example 1 shows that existing methods fail in settings with large action spaces. To overcome this problem, this paper proposed a new method called Action Intersection Double Q-learning (AIDQ). The core idea is maintaining two Q-tables $( \mathrm { e . g . , } Q ^ { 1 }$ and $Q ^ { 2 } )$ , and using $Q ^ { 1 }$ to select a set of optimal actions (e.g., ���� actions), using $Q ^ { 1 }$ to identify the optimal action in the set by estimating their Q-values.

Figure 2(b) shows our action intersection Double Q-learning. We denote the two Q-functions at time step � as $Q _ { t } ^ { 1 }$ and $Q _ { t } ^ { 2 }$ . Firstly, randomly initialize two Q-functions as $Q _ { 0 } ^ { 1 }$ and $Q _ { 0 } ^ { 2 }$ at time step $t = 0$ . Secondly, at any time step �, we generate a policy $\pi _ { t } : S  { \mathcal { A } }$ by

$$
a = \pi _ { t } ( s ) \stackrel { \mathrm { d e f } } { = } \arg \operatorname* { m a x } _ { a ^ { \prime } } \left[ Q _ { t } ^ { 1 } ( s , a ^ { \prime } ) + Q _ { t } ^ { 2 } ( s , a ^ { \prime } ) \right] .\tag{3}
$$

Secondly, at any time step �, the agent is at state $s _ { t }$ and uses �-greedy policy with $\pi _ { t }$ to decide the action $a _ { t }$ to take where $\epsilon \in [ 0 , 1 ]$

$$
a _ { t } = \left\{ \begin{array} { l l } { S a m p l e ( \mathcal { A } ) } & { P r o b . = \epsilon , } \\ { \pi _ { t } ( s _ { t } ) } & { P r o b . = 1 - \epsilon , } \end{array} \right.\tag{4}
$$

where ������() is to choose a random action from  with equal probability. The environment receives the action $a _ { t }$ and then gives the reward $r _ { t }$ and the next state $s _ { t + 1 }$ . This process generates a sampling data $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } )$ . The data will be used to update the Q-functions.

Thirdly, randomly select either $Q _ { t } ^ { 1 }$ or $Q _ { t } ^ { 2 }$ with equal probability to update (e.g., select $Q _ { t } ^ { 1 }$ to update). Then we select the ���� actions with the highest Q-values for state $s _ { t + 1 }$ according to $Q _ { t } ^ { 1 }$ as the action intersection set $\boldsymbol { \mathcal { A } } _ { t + 1 } ^ { K * }$

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } _ { t + 1 } ^ { K * } = \arg \operatorname { t o p } \mathbf { K } _ { a ^ { \prime } \in \mathcal { A } } \boldsymbol { Q } _ { t } ^ { 1 } ( \boldsymbol { s } _ { t + 1 } , a ^ { \prime } ) , } \end{array}\tag{5}
$$

where we extend the max operation to a topK operation. This set is called the action intersection set because it is not only selected by $Q _ { t } ^ { 1 }$ , but also used by $Q _ { t } ^ { 2 }$ . Note that $| \mathcal { A } _ { t + 1 } ^ { K * } | = t o p K$ . The other Q-function $Q _ { t } ^ { 2 }$ selects the optimal action for state $s _ { t + 1 }$ from the action intersection set $\boldsymbol { \mathcal { A } } _ { t + 1 } ^ { K * }$ and gives its Q-value as:

$$
\operatorname* { m a x } _ { a _ { t + 1 } \in \mathcal { A } _ { t + 1 } ^ { K * } } Q _ { t } ^ { 2 } ( s _ { t + 1 } , a _ { t + 1 } ) .\tag{6}
$$

![](images/71cf3dd7e6639f88658e7daeaa738f17fc1271fc574b08b8cb1d3252924256f9.jpg)  
Figure 2: Double Q learning with action intersection

The target value $Y _ { t } ( s _ { t } , a _ { t } )$ for $Q _ { t } ^ { 1 }$ is calculated as:

$$
Y _ { t } ^ { 1 } ( s _ { t } , a _ { t } ) = r _ { t } + \gamma \operatorname* { m a x } _ { a _ { t + 1 } \in \mathcal { A } _ { t + 1 } ^ { K * } } Q _ { t } ^ { 2 } ( s _ { t + 1 } , a _ { t + 1 } ) .\tag{7}
$$

Finally, update $Q _ { t } ^ { 1 }$ as:

$$
Q _ { t + 1 } ^ { 1 } ( s _ { t } , a _ { t } ) = ( 1 - \alpha ) Q _ { t } ^ { 1 } ( s _ { t } , a _ { t } ) + \alpha Y _ { t } ^ { 1 } ( s _ { t } , a _ { t } ) .\tag{8}
$$

If $Q _ { t } ^ { 2 }$ is chosen to update, the update rule for $Q _ { t } ^ { 2 }$ is defined by swapping the roles of the two Q-functions in the update rule of $Q _ { t } ^ { 1 }$ , which is given by:

$$
\left\{ \begin{array} { l l } { \mathcal { A } _ { t + 1 } ^ { K * } = \underset { a ^ { \prime } \in A } { \arg \mathrm { t o p K } } Q _ { t } ^ { 2 } ( s _ { t + 1 } , a ^ { \prime } ) , } \\ { Y _ { t } ^ { 2 } ( s _ { t } , a _ { t } ) = r _ { t } + \gamma \operatorname* { m a x } _ { a _ { t + 1 } \in A _ { t + 1 } ^ { K } \ast } Q _ { t } ^ { 1 } ( s _ { t + 1 } , a _ { t + 1 } ) , } \\ { Q _ { t + 1 } ^ { 2 } ( s _ { t } , a _ { t } ) = ( 1 - \alpha ) Q _ { t } ^ { 2 } ( s _ { t } , a _ { t } ) + \alpha Y _ { t } ^ { 2 } ( s _ { t } , a _ { t } ) . } \end{array} \right.\tag{9}
$$

Algorithm 1 outlines our Action Intersection Double Q-learning (AIDQ). When ���� = 1, our method turns to Double Q-learning, which has a negative bias. When ��� $\mathbf { \nabla } \cdot K \mathbf { \overline { { = } } } \left| \mathcal { A } \right|$ , where || is the size of the action space, our method turns to two Q-tables alternately performing Q-learning updates, which has a positive bias. This feature helps to enable fine-grained control over a large bias range, from negative bias to positive bias in the large action space setting. Thus, we intuitively show that the estimation bias of our method can be adjusted in a fine-grained and continuous manner from negative bias to positive bias by increasing ����, and there exists a suitable � to remove the estimation bias.

Algorithm 1 Action Intersection Double Q-learning   
Init: Q-tables $Q _ { 0 } ^ { 1 } , Q _ { 0 } ^ { 2 } ;$ start state $s _ { 0 }$   
Parameter: Action Intersection set size ����   
1: for � in $\{ 0 , 1 , 2 , \ldots \}$ do   
2: Choose action $a _ { t }$ at state $s _ { 0 }$ by �-greedy policy   
3: $r _ { t } , s _ { t + 1 } \gets E n v ( s _ { t } , a _ { t } )$   
4: Sample a random probability $u \sim$ Uniform(0,1)   
5: if $u < 0 . 5$ then   
6: $\begin{array} { r } { \mathcal { A } _ { t + 1 } ^ { K * } \gets \arg \operatorname { t o p K } _ { a ^ { \prime } \in \mathcal { A } } Q _ { t } ^ { 1 } ( s _ { t + 1 } , a ^ { \prime } ) } \end{array}$   
7: $Y _ { t } = r _ { t } + \gamma$ max $Q _ { t } ^ { 2 } ( s _ { t + 1 } , a ^ { \prime } )$   
$a ^ { \prime } \in \mathcal { A } _ { t + 1 } ^ { K * }$   
8: $Q _ { t + 1 } ^ { 1 }  ( 1 - \alpha ) Q _ { t } ^ { 1 } + \alpha Y _ { t }$   
9: else if $u > = 0 . 5$ then   
10: $\begin{array} { r } { \mathcal { A } _ { t + 1 } ^ { K * } \gets \arg \operatorname { t o p K } _ { a ^ { \prime } \in \mathcal { A } } Q _ { t } ^ { 2 } ( s _ { t + 1 } , a ^ { \prime } ) } \end{array}$   
11: $Y _ { t } = r _ { t } + \gamma \operatorname* { m a x } _ { \substack { \scriptscriptstyle \int \_ { t } = \scriptscriptstyle \int \_ { t + 1 } , a ^ { \prime } ) } } Q _ { t } ^ { 1 } ( s _ { t + 1 } , a ^ { \prime } )$   
$a ^ { \prime } \in \mathcal { A } _ { t + 1 } ^ { K * }$   
12: $Q _ { t \pm 1 } ^ { 2 }  ( 1 - \alpha ) \dot { Q } _ { t } ^ { 2 } + \alpha Y _ { t }$   
13: end if   
14: end for

## 5. Theoretical Analysis

Let $W = \{ W _ { 1 } , . . , W _ { M } \}$ be a � dimension vector of independent random variables. We wish to estimate

$$
\operatorname* { m a x } _ { i } \mathbb { E } [ W _ { i } ] .\tag{10}
$$

We have a sample set � which is randomly sampled from $W$ . Given split parameter $\rho \in [ 0 , 1 ]$ , we can randomly split � into two disjoint subsets $S ^ { A }$ and $S ^ { B }$ , and $S ^ { C }$ with equal proportion. e.g.

$$
\left\{ \begin{array} { l } { { S = S ^ { A } \cup S ^ { B } , } } \\ { { S ^ { A } \cap S ^ { B } = \phi , } } \\ { { | S ^ { A } | : | S ^ { B } | = 1 : 1 . } } \end{array} \right.\tag{11}
$$

Furthermore,We can interpret $S ^ { A }$ and $S ^ { B }$ as collections of random variables:

$$
\left\{ { S } ^ { A } = \{ X _ { i , j } | i \in \{ 1 , . . . , M \} , j \in \{ 1 , . . . , N _ { A } \} \} , \right.\tag{12}
$$

which satisfies:

$$
N _ { A } : N _ { B } = 1 : 1 .\tag{13}
$$

For example, $X _ { i , j }$ means the $j ^ { t h }$ sample of $W _ { i }$

We define the following empirical mean estimator based on $S ^ { A }$ and $S ^ { B }$ , respectively:

$$
{ \bar { X } _ { i } } \ { \stackrel { \mathrm { d e f } } { = } } \ { \frac { \sum _ { j = 1 } ^ { N _ { A } } X _ { i , j } } { | N _ { A } | } } ,\tag{14}
$$

$$
\bar { Y } _ { i }  { \stackrel { \mathrm { d e f } } { = } } \frac { \sum _ { j = 1 } ^ { N _ { B } } Y _ { i , j } } { | N _ { B } | } ,\tag{15}
$$

Thus, the optimal action set $A ^ { * }$ with size ���� is

$$
\mathcal { A } ^ { * } = \arg \mathrm { t o p K } _ { i \in \{ 0 , 1 , \ldots , M - 1 \} } \bar { X } _ { i }\tag{16}
$$

Our estimator is

$$
{ \bar { Y } } _ { a ^ { * } } ,\tag{17}
$$

where

$$
a ^ { * } = \arg \operatorname* { m a x } _ { i \in \mathcal { A } ^ { * } } \bar { Y } _ { i } .\tag{18}
$$

Theorem 1. Let $S ^ { A }$ and $S ^ { B }$ be the independent subsets ofobservations samplingfrom the random variable $W$ , with an equal proportion. Let ${ \bar { X } } _ { i }$ and ${ \bar { X } } _ { i }$ be the mean estimator $o f \mathbb { E } [ W _ { i } ]$ based on $S ^ { A }$ and $S ^ { B }$ , respectively. Let $\bar { Y } _ { a * }$ be the estimator ofmax $\mathbb { E } [ W _ { i } ]$ , where $a ^ { * } = \arg \operatorname* { m a x } _ { i \in \mathcal { A } ^ { * } } \bar { X } _ { i }$ , where $\mathcal { A } ^ { * } = \arg \mathrm { t o p K } _ { i \in \{ 0 , 1 , \ldots , M - 1 \} } \bar { X } _ { i }$ Then, there exists a ���� such that $\mathbb { E } [ \bar { Y } _ { a * } ] = \operatorname* { m a x } _ { i } \mathbb { E } [ W _ { i } ]$

Proof. The proof is provided in Section 8.

## 6. Action Intersection Double DQN

We denote the two deep Q-networks as $Q ( s , a ; \theta _ { t } ^ { 1 } )$ and $Q ( s , a ; \theta _ { t } ^ { 2 } )$ that are parameterized by $\theta _ { t } ^ { 1 }$ and $\theta _ { t } ^ { 2 }$ , respectively. The input of the deep neural network is state � and action �, the output is the estimated Q-value of state-action pair $( s , a )$ , i.e.,

$$
\left\{ \begin{array} { l l } { \hat { Q } _ { s , a } ^ { 1 } = Q ( s , a ; \theta _ { t } ^ { 1 } ) , } \\ { \hat { Q } _ { s , a } ^ { 2 } = Q ( s , a ; \theta _ { t } ^ { 2 } ) . } \end{array} \right.\tag{19}
$$

Then we denote two target networks as $Q ( s , a ; \theta _ { t } ^ { 1 , t a r g e t } )$ and $Q ( s , a ; \theta _ { t } ^ { 2 , t a r g e t } )$ that are parameterized by $\theta _ { t } ^ { 1 , t a r g e t }$ and $\theta _ { t } ^ { 2 , t a r g e t }$ , respectively. They are delayed copies of the online networks.

At the first step $t \ = \ 0 ,$ , we randomly initialize the two deep neural networks $Q ( s , a ; \theta _ { t } ^ { 1 } )$ and $Q ( s , a ; \theta _ { t } ^ { 2 } )$ with parameters $\theta _ { 0 } ^ { 1 }$ and $\theta _ { 0 } ^ { 2 }$ , respectively. Then we copy $\theta _ { 0 } ^ { 1 }$ and $\theta _ { 0 } ^ { 2 }$ to $\theta _ { 0 } ^ { 1 , t a r g e t }$ and $\theta _ { 0 } ^ { 2 , t a r g e t }$ , respectively (i.e., $\theta _ { 0 } ^ { 1 , t a r g e t } = \theta _ { 0 } ^ { 1 }$ and $\theta _ { 0 } ^ { 2 , t a r g e t } = \theta _ { 0 } ^ { 2 } )$

At any time step �, we generate a policy $\pi _ { t } : S  { \mathcal { A } }$ by

$$
a = \pi _ { t } ( s ) \overset { \mathrm { d e f } } { = } \arg \operatorname* { m a x } _ { a ^ { \prime } } \left[ Q ( s , a ^ { \prime } ; \theta _ { t } ^ { 1 } ) + Q ( s , a ^ { \prime } ; \theta _ { t } ^ { 2 } ) \right] .\tag{20}
$$

Then, at any time step �, the agent is at state $s _ { t }$ and uses �-greedy policy with $\pi _ { t }$ to decide the action $a _ { t }$ to take where $\epsilon \in [ 0 , 1 ]$

$$
a _ { t } = \left\{ \begin{array} { l l } { S a m p l e ( \mathcal { A } ) } & { P r o b . = \epsilon } \\ { \pi _ { t } ( s _ { t } ) } & { P r o b . = 1 - \epsilon } \end{array} \right.\tag{21}
$$

where ������() is to choose a random action from  with equal probability. The environment receives the action $a _ { t }$ and then gives the reward $r _ { t }$ and the next state $s _ { t + 1 }$ . This process generates a sampling data $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } )$ . The data will be added to the replay bufer :

$$
\boldsymbol { B } = B \cup \{ ( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } ) \}\tag{22}
$$

The agent will repeat the above exploration process 1000 times. It makes sure that there are enough samples in the replay bufer to update networks in the following steps.

At time step $t > 1 0 0 0$ , the agent first interacts with the environment to sample trajectory data and update bufer  using the above exploration process. Secondly, sample a mini-batch $B _ { m i n i }$ uniformly at random without replacement from the bufer . Then update the Q-networks with the following rules

Firstly, select either $Q ( \cdot ; \theta _ { t } ^ { 1 } )$ or $Q ( \cdot ; \theta _ { t } ^ { 2 } )$ with equal probability to update $( { \bf e . g . } , \ Q ( \cdot ; \theta _ { t } ^ { 1 } ) )$ . For any sample $( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 }$ in $B _ { m i n i }$ , we select the optimal action intersection set $\boldsymbol { \mathcal { A } } _ { i + 1 } ^ { K * }$ for state $s _ { i + 1 }$

$$
\begin{array} { r } { \mathcal { A } _ { i + 1 } ^ { K * } = \arg \mathrm { t o p K } _ { a ^ { \prime } \in \mathcal { A } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 1 } ) . } \end{array}\tag{23}
$$

The target value for $Q ( s _ { i } , a _ { i } ; \theta _ { t } ^ { 1 } )$ is

$$
Y _ { i } ( s _ { i } , a _ { i } ) = r _ { i } + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } _ { i + 1 } ^ { K * } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 2 , t a r g e t } ) .\tag{24}
$$

The loss function is

$$
J ( \theta _ { t } ^ { 1 } ) = \frac { 1 } { | \mathcal { B } _ { m i n i } | } \sum _ { ( s _ { i } , a _ { r } , r _ { i } , s _ { i + 1 } ) \in \mathcal { B } _ { m i n i } } \left[ Y _ { i } ( s _ { i } , a _ { i } ) - Q ( s _ { i } , a _ { i } ; \theta _ { t } ^ { 1 } ) \right] ^ { 2 } .\tag{25}
$$

update parameters $\theta _ { t } ^ { 1 }$ as:

$$
\theta _ { t + 1 } ^ { 1 } = \theta _ { t } ^ { 1 } + \alpha \frac { \partial J ( t h e t a _ { t } ^ { 1 } ) } { \partial \theta _ { t } ^ { 1 } } ,\tag{26}
$$

where � is the learning rate.

If $Q ( \cdot ; \theta _ { t } ^ { 2 } )$ is chosen to be updated, then exchange the role of the two Q-networks in the update rule of $Q ( \cdot ; \theta _ { t } ^ { 1 } )$

$$
\begin{array} { r l } & { \left\{ \begin{array} { l l } { \mathcal { A } _ { i + 1 } ^ { K * } = \arg \mathrm { t o p K } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 2 } ) , } \\ { \mathcal { I } _ { i } ( s _ { i } , a _ { i } ) = r _ { i } + \gamma \displaystyle \operatorname* { m a x } _ { a ^ { \prime } \in A _ { i + 1 } ^ { K * } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 1 , t a r g e t } ) , } \\ { J ( \theta _ { t } ^ { 2 } ) = \frac { 1 } { | B _ { m i n i } | } \sum _ { ( s _ { i } , a _ { r } , r _ { i } , s _ { i + 1 } ) \in B _ { m i n i } } \left[ Y _ { i } ( s _ { i } , a _ { i } ) - Q ( s _ { i } , a _ { i } ; \theta _ { t } ^ { 2 } ) \right] ^ { 2 } , } \\ { \theta _ { t + 1 } ^ { 2 } = \theta _ { t } ^ { 2 } + \alpha \frac { \partial J ( \theta _ { t } ^ { 2 } ) } { \partial \theta _ { t } ^ { 2 } } . } \end{array} \right. } \end{array}\tag{27}
$$

Periodically, the parameters of the online network are copied to the target network, e.g., if � mod $a = = 0$ , then

$$
\left\{ \begin{array} { l l } { \theta _ { t + 1 } ^ { 1 , t a r g e t } = \theta _ { t + 1 } ^ { 1 } , } \\ { \theta _ { t + 1 } ^ { 2 , t a r g e t } = \theta _ { t + 1 } ^ { 2 } , } \end{array} \right.\tag{28}
$$

where a is the interval of copy.

Algorithm 2 extends our AIDQ to complex and high-dimensional domains, leads to Action Intersection Double DQN (AIDDQN). We first parameterize the two Q-functions with deep neural networks $Q ( s , a ; \theta _ { t } ^ { 1 } )$ and $Q ( s , a ; \theta _ { t } ^ { 2 } )$ . And then to ensure stable learning in deep reinforcement learning setting Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra and Riedmiller (2013); Mnih, Kavukcuoglu, Silver, Rusu, Veness, Bellemare, Graves, Riedmiller, Fidjeland, Ostrovski et al. (2015), we use two target networks denoted as $Q ( s , a ; \theta _ { t } ^ { 1 , t a r g e t } )$ and $Q ( s , a ; \theta _ { t } ^ { 2 , t a r g e t } )$ , which are delayed copies of the online networks. These target networks are used to compute the temporal-diference targets, thereby preventing the feedback loops that can lead to divergence.

## 7. Experiment

In this section, we evaluate the proposed method on both tabular settings and deep network settings. In the tabular setting, we evaluate and compare our method with 8 baselines in a Multi-armed bandit. In the deep network setting, we evaluate and compare our method with 8 baselines on 6 simulated environments. The code is at anonymous https://anonymous.4open.science/r/AIDQ-D826/Readme.txt.

Algorithm 2 Action Intersection DDQN   
Init: Four Q-networks with random parameters $\theta _ { 0 } ^ { 1 } , \theta _ { 0 } ^ { 2 } , \theta _ { 0 } ^ { 1 , \mathrm { t a r g e t } } , \theta _ { 0 } ^ { 2 , \mathrm { t a r g e t } }$ ; replay bufer $B = \{ \}$ ; start state $s _ { 0 }$   
Parameter: Action Intersection set size $t o p K ;$ target Q-networks update step �   
1: for � in $\{ 0 , 1 , 2 , \ldots \}$ do   
2: if � mod $V = = 0$ then   
3: $\theta _ { t } ^ { 1 , \mathrm { t a r g e t } }  \theta _ { t } ^ { 1 }$   
4: $\boldsymbol { \theta } _ { t } ^ { \mathrm { 2 , t a r g e t } }  \boldsymbol { \theta } _ { t } ^ { \mathrm { 2 } }$   
5: end if   
6: Choose action $a _ { t }$ at state $s _ { t }$ by �-greedy policy   
7: $r _ { t } , s _ { t + 1 } \gets E n v ( s _ { t } , a _ { t } )$   
8: $\boldsymbol { B } \gets \boldsymbol { B } \cup \{ ( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } ) \}$ {Update bufer}   
9: Sample minibatch $B _ { m i n i }$ of transitions from    
10: Sample a random probability $u \sim \mathrm { U n i f o r m } ( 0 , 1 )$   
11: if $u < 0 . 5$ then   
12: for $( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 } ) \in B _ { m i n i }$ do   
13: $\begin{array} { r } { \mathcal { A } _ { i } ^ { K * } \gets \arg \operatorname { t o p K } _ { a ^ { \prime } \in \mathcal { A } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 1 , t a r g e t } ) } \end{array}$   
14: $Y _ { i } = r _ { i } + \gamma \operatorname* { m a x } _ { \substack { \scriptscriptstyle \mathrm {  \beta ^ { \prime } } \scriptscriptstyle \mathrm {  \beta } } } Q _ { t } ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 2 , t a r g e t } )$   
�<sup>′</sup>∈   
15: end for   
16: minimize $\mathbb { E } _ { ( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 } ) \in { \mathcal { B } _ { m i n i } } } \left[ Y _ { i } - Q ( s _ { i } , a _ { i } ; \theta _ { t } ^ { 1 } ) \right] ^ { 2 }$   
17: else   
18: for $( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 } ) \in B _ { m i n i }$ do   
19: $\begin{array} { r } { \mathcal { A } _ { i } ^ { K * } \gets \arg \operatorname { t o p K } _ { a ^ { \prime } \in \mathcal { A } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 2 , t a r g e t } ) } \end{array}$   
20: $Y _ { i } = r _ { i } + \gamma \operatorname* { m a x } _ { \substack { \scriptscriptstyle \mathrm { ~ , ~ } \ldots \ldots } } Q ( s _ { i + 1 } , a ^ { \prime } ; \theta _ { t } ^ { 1 , t a r g e t } )$   
$a ^ { \prime } \in \mathcal { A } _ { i } ^ { K * }$   
21: end for   
22: minimize $\mathbb { E } _ { ( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 } ) \in { \cal B } _ { m i n i } } \left[ Y _ { i } - Q ( s _ { i } , a _ { i } ; \theta _ { t } ^ { 2 } ) \right] ^ { 2 }$   
23: end if   
24: end for

## 7.1. Multi-Armed Bandit

Environment: Following previous work Zhang et al. (2017), we consider a simple multi-armed bandit setting. It includes one unterminated state � with � arms, i.e., $\mathcal { A } = \{ a _ { i } | i = 1 , 2 , . . , N \}$ . For any arm $a _ { i } .$ , it yields a reward drawn from a normal distribution $\mathcal { N } ( \mu _ { i } , \sigma _ { R } ^ { 2 } )$ , where $\mu _ { i } = 0$ Parameters: Following previous work Zhang et al. (2017), we set the discount factor $\gamma = 0 . 9 5$ , the learning rate $\begin{array} { r } { \alpha _ { t } ( S , a ) = \frac { 1 } { n _ { t } ( S , a ) ^ { 0 . 8 } } } \end{array}$ , the exploration parameter $\begin{array} { r } { \epsilon _ { t } ( S ) = \frac { 1 } { n _ { t } ( S ) ^ { 0 . 5 } } } \end{array}$ , where $n _ { t } ( S )$ and $n _ { t } ( S , a )$ are the visited number of state � and state-action pair (�, �), respectively. Note that under the above setting, the optimal policy always chooses the action $a _ { 1 }$ at state �, and the maximum expected Q-value is as $\begin{array} { r } { \operatorname* { l i m } _ { T \to + \infty } \sum _ { t = 0 } ^ { T } \gamma ^ { t } = 0 } \end{array}$ . The initial Q-values obey $\mathcal { N } ( 0 , \sigma _ { O } ^ { 2 } )$ , and all results are averaged over 1000 runs.

Baseline: We consider eight SOTA baselines as follows:

• Q-learning Watkins and Dayan (1992) is one of the most classical algorithms of reinforcement learning. However, it faces the challenge of the overestimation bias.

• Double Q-learning Hasselt (2010) was proposed to tackle the overestimation issue in standard Q-learning. It decouples the action selection from the value evaluation by maintaining two separate Q-value estimators. While efectively reducing overestimation, this decoupling can introduce an underestimation bias, potentially slowing down convergence.

• Weighted Double Q-learning Zhang et al. (2017) introduces a weighted averaging scheme to balance between the two Q-estimators in Double Q-learning. By dynamically adjusting the weights based on the uncertainty of each estimator, it aims to strike a better bias-variance trade-of, mitigating both overestimation and severe underestimation.

• Averaged Q-learning Anschel et al. (2017) proposes a simple yet efective approach that averages multiple Qvalue estimates during the update. This averaging operation naturally reduces the variance of the value estimates, leading to more stable learning, though it may sometimes oversmooth the value function.

• Maxmin Q-learning Lan et al. (2020) adopts a conservative strategy by using the minimum value among an ensemble of Q-functions for the target update. This design robustly controls overestimation and has shown strong performance in environments with high stochasticity, but may become overly pessimistic in deterministic settings.

• Ensemble Bootstrapped Q-learning (EBQL) Peer et al. (2021) leverages an ensemble of Q-estimators, each trained on a diferent bootstrapped sample of the experience replay bufer. The diversity within the ensemble helps in better exploration and more robust value estimation, though at the cost of increased computational complexity.

• Order Q-learning Tan et al. (2024a) is a recent approach that employs order statistics to combine multiple Q-value predictions. By selecting a specific percentile (e.g., median or a higher-order statistic) as the target value, it attempts to interpolate between overestimation and underestimation, ofering a flexible framework for bias control.

• AC-C Double Q-learning (AC-CDQ) Jiang et al. (2021) proposes a clipped double estimator for Double Qlearning, in order to reduce the underestimation bias. It maintains two estimators. When calculating the temporal diference target, the corresponding action value of the selected action in the first set of estimators is clipped by the maximum value in the second set of estimators.

To ensure fairness, all algorithms maintain two Q-tables, except for Q-learning. For the hyperparameters of algorithms, Weighted Double Q-learning Zhang et al. (2017) sets the adaptive adjustment factor as 10; Order Q-learning Tan, Xie, Xia, Shi and Shang (2024b) sets the index of the order function as 2. AC-CDQ sets the candidate size as 2. Note that these hyperparameters of comparison baselines are chosen from the recommended hyperparameters, which are fine-tuned.

We consider three diferent settings to evaluate the efect of our data multiplexing strategy as follows:

• Case 1: diferent arms with $N \in \{ 2 0 , 4 0 , 6 0 , 8 0 \}$ ;

• Case 2: diferent reward variance with $\sigma _ { R } \in \{ 5 , 1 0 , 1 5 , 2 0 \}$

• Case 3: diferent initial Q-values variance with $\sigma _ { Q } \in \{ 1 , 2 , 4 , 8 \}$

We set the arms with $N = 4 0$ , the reward variance with $\sigma _ { R } = 1 0$ , and the initial Q-values variance with $\sigma _ { Q } = 1$ for default.

Baseline Comparison Figure 3 shows all experiment result of all baselines under 3 cases. For any subplot in Figure 3, the x-axis represents the update steps, and the y-axis represents the maximum Q-value estimated by the algorithm for state �. The closer an algorithm’s curve is to the dashed unbiased estimation curve, the more accurate its Q-value estimation for state �.

Figure 3A(1)-A(4) shows the experiment result under case 1, where the number of arms is varied. There are three observations. (1) under diferent cases, the red curve of our method is in the middle and most closely aligns with the dashed unbiased estimation line. This implies that our method is able to achieve unbiased estimation with a large action space. Specifically, after 10,000 training steps, the maximum Q-value estimated by our algorithm aligns almost perfectly with the true Q-value. (2) Under diferent cases, the curve of the baseline method deviates from the dashed unbiased estimation line. This implies that baselines sufer from estimation bias under the large space action setting. More specifically, Double Q and EBQL have a negative bias; other methods have a positive bias. (3) As the number of arms increases, the deviation between the curve of the baseline algorithm and the unbiased estimation curve widens; the number of training steps required for our algorithm’s curve to begin converging toward the unbiased estimation curve increases.

![](images/8ac41357b7ea931413d22ac33c59f5ac1564b132146ba3d00a1fab8aee13d5cd.jpg)  
Figure 3: The average return per episode of eight diferent algorithms in comparison.

Figure 3B(1)-B(4) shows the experiment result under case 2, where the standard deviation of reward are varied. There are two observations: (1) under diferent cases, the red curve of our method is in the middle and most closely aligns with the dashed unbiased estimation line. This implies that the action intersection strategy is able to achieve unbiased estimation. (2) As the standard deviation of reward increases, The distance between the curve of the baseline algorithm and the unbiased estimation curve increases significantly(e.g., when ���� = 5, the maximum Q value of Order Q can be up to 40; when ���� = 20, the maximum Q value of Order Q can be up to 150); Our algorithm converges to the unbiased estimator in a similar number of steps (25,000) under all four conditions. This implies that our action intersection strategy is less afected by reward variance.

Figure 3C(1)-C(4) shows the experiment result under case 3, where the standard deviation of the initialized Q-value is varied. There are two observations: (1) under diferent cases, the red curve of our method is in the middle and most closely aligns with the dashed unbiased estimation line. (2) As the standard deviation of initialized Q-values increases, all curves of our method and baselines maintain the same relative positions. This implies that the variance in the initialized Q-values has little impact on the algorithm’s performance.

Parameter Sensitivity Figure 4 shows the impact of the hyperparameter ���� on our algorithm under 3 cases. For any subplot in Figure 4, the x-axis represents the update steps, and the y-axis represents the maximum Q-value estimated by the algorithm for state �. The closer an algorithm’s curve is to the dashed unbiased estimation curve, the more accurate its Q-value estimation for state �.

![](images/20b1e77bae70ffd916d29f418c5c4a9c5613f7fe6af2043e8ba64aeb02a23b38.jpg)  
Figure 4: The maximum Q-value of our AIDQ with diferent ����.

Figure 4A(1)-A(4) shows the experimental results on the sensitivity of parameter ���� in our algorithm under diferent action space sizes. There are three observations: (1) Under diferent settings for the number of actions, as the value of ���� increases, the estimation bias can vary from negative bias to positive bias. This implies that it is possible to find a suitable value for ���� such that the estimated maximum Q-value approximates the unbiased Q-value. (2) Under diferent settings for the number of actions, our algorithm can always find a suitable ���� such that the estimated maximum Q-value approximates the unbiased Q-value. This implies that it is feasible for our algorithm to find a suitable parameter ��� − � during deployment. (3) As the number of arms increases, the suitable value of ���� increases. When the number of arms is 20, the suitable ���� is 3; when the number of arms is 80, the suitable value of ���� is 6. The optimal value of parameter ���� increases approximately linearly with the number of actions. This implies that it aligns well with empirical intuition regarding our method’s hyperparameter tuning.

Figure 4B(1)-B(4) shows the experimental results on the sensitivity of parameter ���� in our algorithm under diferent standard deviations of reward. One can observe that (1) Under diferent settings for the standard deviation, as the value of ���� increases, the estimation bias can vary from negative bias to positive bias. This implies that it is possible to find a suitable value for ���� such that the estimated maximum Q-value approximates the unbiased Q-value. (2) Under diferent settings for the standard deviation, our algorithm can always find a suitable ����, which is 5, such that the estimated maximum Q-value approximates the unbiased Q-value. This implies that the parameter ���� is insensitive to reward variance.

Figure 4C(1)-BC(4) shows the experimental results on the sensitivity of parameter ���� in our algorithm under diferent standard deviations of initialized Q-values. One can observe that (1) Under diferent settings for the standard deviation, as the value of ���� increases, the estimation bias can vary from negative bias to positive bias. This implies that it is possible to find a suitable value for ���� such that the estimated maximum Q-value approximates the unbiased Q-value. (2) As the standard deviation of initialized Q-values increases, the positions of the curves for diferent values of parameter K show little variation. Our algorithm can always find a suitable ����, which is 4, such that the estimated maximum Q-value approximates the unbiased Q-value. This implies that the parameter ���� is insensitive to the variance of initialized Q-values.

Table1:ThemaximumQ-valueofdiferentalgorithmsunderthreediferentsettings.Theoptimalestimationvaluesineachcolumnarehighlightedinbold.The secondbestresultineachcolumnareunderlined.
<table><tr><td rowspan="2">Algorithm</td><td colspan="4">N arm</td><td colspan="4">σR</td><td colspan="4">σQ</td></tr><tr><td>20</td><td>40</td><td>60</td><td>80</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>2</td><td>4</td><td>8</td></tr><tr><td>Q</td><td>24.64</td><td>32.84</td><td>36.72</td><td>39.60</td><td>16.30</td><td>32.84</td><td>48.81</td><td>65.30</td><td>32.84</td><td>32.89</td><td>33.80</td><td>34.96</td></tr><tr><td>Double Q</td><td>-27.37</td><td>-40.23</td><td>-48.87</td><td>-53.48</td><td>-20.44</td><td>-40.23</td><td>-59.96</td><td>-80.29</td><td>-40.23</td><td>-40.33</td><td>-40.40</td><td>-44.75</td></tr><tr><td>Weighted Q</td><td>28.86</td><td>42.46</td><td>48.92</td><td>51.60</td><td>15.65</td><td>42.46</td><td>72.69</td><td>101.75</td><td>42.46</td><td>46.33</td><td>50.89</td><td>58.87</td></tr><tr><td>Averaged Q</td><td>17.37</td><td>21.46</td><td>23.58</td><td>24.64</td><td>10.98</td><td>21.46</td><td>32.49</td><td>42.81</td><td>21.46</td><td>21.89</td><td>22.15</td><td>22.83</td></tr><tr><td>Maxmin Q</td><td>4.47</td><td>8.51</td><td>10.80</td><td>12.19</td><td>4.48</td><td>8.51</td><td>12.80</td><td>17.09</td><td>8.51</td><td>8.78</td><td>9.46</td><td>10.64</td></tr><tr><td>EBQL</td><td>-27.61</td><td>-39.95</td><td>-48.73</td><td>-53.52</td><td>-20.51</td><td>-39.95</td><td>-60.75</td><td>-80.52</td><td>-39.95</td><td>-40.79</td><td>-41.08</td><td>-43.14</td></tr><tr><td>AC-CDQ</td><td>29.11</td><td>49.98</td><td>67.01</td><td>78.79</td><td>30.08</td><td>49.98</td><td>70.34</td><td>91.65</td><td>49.98</td><td>51.14</td><td>54.15</td><td>59.05</td></tr><tr><td>Order Q</td><td>50.22</td><td>62.83</td><td>69.31</td><td>73.29</td><td>31.50</td><td>62.83</td><td>94.33</td><td>125.05</td><td>62.83</td><td>62.84</td><td>63.84</td><td>64.93</td></tr><tr><td>AIDQ(topK=2)</td><td>-8.79</td><td>-20.14</td><td>-28.87</td><td>-34.90</td><td>-10.14</td><td>-20.14</td><td>-29.22</td><td>-40.92</td><td>-20.14</td><td>-20.48</td><td>-20.72</td><td>-20.85</td></tr><tr><td>AIDQ(topK=3)</td><td>-0.06</td><td>-10.20</td><td>-18.76</td><td>-23.61</td><td>-5.10</td><td>-10.20</td><td>-15.67</td><td>-19.55</td><td>-10.20</td><td>-9.91</td><td>-10.78</td><td>-11.45</td></tr><tr><td>AIDQ(topK=4)</td><td>6.56</td><td>-2.35</td><td>-9.86</td><td>-14.89</td><td>-1.38</td><td>-2.35</td><td>-3.17</td><td>-4.38</td><td>-2.35</td><td>-2.75</td><td>-3.53</td><td>-3.72</td></tr><tr><td>AIDQ(topK=5)</td><td>11.75</td><td>3.93</td><td>-2.45</td><td>-7.96</td><td>1.87</td><td>3.93</td><td>5.50</td><td>6.98</td><td>3.93</td><td>3.74</td><td>3.15</td><td>2.32</td></tr><tr><td>AIDQ(topK=6)</td><td>15.31</td><td>8.12</td><td>3.46</td><td>-0.62</td><td>3.79</td><td>8.12</td><td>11.98</td><td>16.46</td><td>8.12</td><td>8.03</td><td>7.89</td><td>7.61</td></tr><tr><td>AIDQ(topK=7)</td><td>18.84</td><td>11.52</td><td>6.99</td><td>4.09</td><td>5.83</td><td>11.52</td><td>18.05</td><td>23.57</td><td>11.52</td><td>11.80</td><td>12.05</td><td>11.89</td></tr><tr><td>AIDQ(topK=8)</td><td>22.03</td><td>15.27</td><td>10.37</td><td>7.88</td><td>7.92</td><td>15.27</td><td>23.19</td><td>30.85</td><td>15.27</td><td>15.02</td><td>15.45</td><td>15.60</td></tr></table>

<table><tr><td rowspan="2">DQN - Maxmin DQN</td><td rowspan="2">DDQN EBDQN</td><td rowspan="2">Averaged DQN</td><td>Weighted DDQN</td><td rowspan="2">AIDDQN(Our Best)</td></tr><tr><td>ACC DDQN Order DQN</td></tr></table>

![](images/e5039e8aed45568fc9a6e992c1cefb4cc773b5b793584aa3d38e437e7c52ef4d.jpg)  
(a) Pixelcopter

![](images/a3f4b2a84273542ea67ccbdac794d215498ecc206021923a36008a78c81fd28f.jpg)  
(b) Asterix

![](images/eb305b844161a4585922462f49235956d77a58b715231aae0d8d6dfcecd330ee.jpg)  
(c) Breakout

![](images/6ae24895a30f4ecf55ff76c077f570f6ed30535c0b8f23fdd8fcedda8345c6a0.jpg)  
(d) Seaquest

![](images/4692f2a7e8209d39b679e0d8ec9cb73dc6929ab80d189ddf84b1b7fcd3697679.jpg)  
(e) Spacelnvaders

![](images/ed4bb0972d5cb5ec4678d602c1b68282eed02cb29676e6a9bdea3f6d19eb6479.jpg)  
(f) Pong  
Figure 5: The average return per episode of eight diferent algorithms in comparison.

Table 1 comprehensively shows the maximum Q-value of diferent algorithms under three diferent settings, where the values are evaluated in the final learning steps. One can observe that across all experimental configurations (each column), our algorithm yields estimates that are closest to the ground truth value of 20.

## 7.2. Deep Reinforcement Learning

Environment: Following previous work Lan et al. (2020), this paper selects Pixelcopter, Asterix, Breakout, Seaquest, SpaceInvaders, and Pong from Gymnasium Towers, Kwiatkowski, Terry, Balis, De Cola, Deleu, Goulão, Kallinteris, Krimmel, KG et al. (2024), PyGame Learning Environment (PLE) Tasfi (2016), and MinAtarLan (2019) as our deep reinforcement learning environments. For example, Seaquest environment is one of Atari 2600 games, but a simplified one to make experimentation more accessible and eficient. The default observation is a visual input of the game. Thus, the state space is complex, and it is hard for the agent to capture the relationship among states, actions, and rewards.

![](images/eacb9f782fe7a2a28bfce1443c503bfd6a8894cd22ad9b719fb492072acd11f8.jpg)  
(d) Seaquest  
(e) Spacelnvaders  
(f) Pong  
Figure 6: The average return per episode of our AIDQ with diferent ����.

Parameter: Following previous work Lan et al. (2020), we set the discount factor $\gamma = 0 . 9 9$ , the Adam optimizer Zhang (2018) with the learning rate $\alpha ~ = ~ 0 . 0 0 0 5 , 0 . 0 0 1 , 0 . 0 0 1 , 0 . 0 0 1 , 0 . 0 0 1 , 0 . 0 0 0 9 , 0 . 0 0 0 1$ for environments Pixelcopter, Asterix, Breakout, Seaquest, SpaceInvaders, and Pong, respectively the mini-batch size $| B _ { m i n i } | = 3 2$ , the replay bufer size $| B | = 1 0 0 , 0 0 0$ , the update steps of the target Q-network 200. Note that �-greedy is applied as the exploration strategy with decreasing linearly from 1.0 to 0.01 in 1, 000 steps, and after 1, 000 steps, � is fixed to 0.01. All results reported are the average and standard deviation over 20 independent runs with diferent random seeds. Note that the original action spaces of these environments were not large. To simulate a large action-space setting, we multiplied their action spaces by a factor. The scaling factors for environments Pixelcopter, Asterix, Breakout, Seaquest, SpaceInvaders, and Pong are 40, 20, 20, 20, 20, and 20, respectively.

Baseline: To ensure consistency, the configurations of baseline algorithms are the same with the previous experiments in Section 7.1.

Baseline Comparison Figure 5 shows the average return per episode of nine diferent algorithms in comparison across step �, where our AIDDQN varies the $t o p K = 2 , 3 , 4 , 5 , 6 , 7 , 8$ and chooses the best result. One can observe that: in all environments, the red reward curve of our method is above all the baseline curves. This implies that in the setting of a large action space, our method can achieve a better performance than other baselines. This is beneficial from the action intersection strategy, which is aiming at a large action space setting. More specifically, in (a) Pixecopter, the red curve of our method is at the top; the curve of Weighted DQN is at the bottom; the curves of other baselines are similar. This implies that in a large action space, all baselines have their bottleneck. In (b) Asterix, the curve of Weighted DQN is at the bottom; the curves of Maxmin DQN, ACC DDQN, and Order DQN are similar; the curves of other baselines are similar, which are at the middle. This implies that in this environment, In (c) Breakout, all curves are well-separated from each other. In (d) Seaquest, all baseline curves show low performance levels. Several algorithms exhibit a performance trend that declines after peaks. Note that, in the parameter sensitivity experiment, when ���� takes certain values, the corresponding curves of our algorithm also exhibit a similar pattern. In (e) SpaceInvaders, the curves of Maxmin DQN, ACC DDQN, and Order DQN are similar, which are in the middle; the curves of other baselines are similar, which are at the bottom. In (d) Pong, all curves exhibit considerable fluctuations and high variance. Nevertheless, the curve of our method is above all the baseline curves for most of the time.

Parameter Sensitivity Figure 6 shows the average return per episode of our AIDDQN with diferent ���� across step �. In (a) Pixelcopter, as the value of $t o p K$ increases, the corresponding curve gradually rises in position. This implies that the performance of our algorithm exhibits a clear positive correlation with the value of parameter ���� in this environment. In (b) Asterix and (d) Seaquest, as the value of ���� increases, the position of the corresponding curve first increases and then decreases, with the optimal value achieved at $t o p K = 3$ . In (c) Breakout, as the value of ���� increases, a noticeable shift in the corresponding curve occurs when ���� changes from 2 to 3, while subsequent changes in ���� result in minimal variation in the curve’s position. In (d) SpaceInvaders, as the value of ���� increases, the position of the corresponding curve first decreases and then increases, with the optimal value achieved at $t o p K = 2$ In (e) Pong, all curves exhibit considerable fluctuations and high variance. Nevertheless, the optimal parameter is identified as $t o p K = 8$

Table 2 shows the average return per episode of diferent SOTA algorithms and our AIDDQN with diferent ����, where the values represent the mean of the reward data from the final 10% of the training process. One can observe that our AIDDQN improves the average return per episode over SOTA baselines by at leas $( 1 0 . 4 0 - 1 . 8 5 ) / 1 . 8 5 =$ 463.45%, $( 8 . 6 0 \ - \ 5 . 6 1 ) / 5 . 6 1 \ = \ 5 3 . 2 1 \% , \ ( 1 3 . 9 0 \ - \ 1 2 . 2 3 ) / 1 2 . 2 3 \ = \ 1 3 . 6 6 \% , \ ( 1 . 0 3 \ - \ 0 . 2 2 ) / 0 . 2 2 \ = \ 3 6 6 . 9 5 \%$ $( 4 0 . 2 6 - 3 6 . 3 8 ) / 3 6 . 3 8 = 1 0 . 6 5 \% , ( ( - 1 3 . 0 6 ) - ( - 1 3 . 5 4 ) ) / | - 1 3 . 5 4 | = 3 . 5 5 \%$ , in Pixelcopter, Asterix, Breakout, Seaquest, SpaceInvaders, and Pong, respectively.

## 8. Proof of Theorem 1

Proof. We define the optimal action selected from the action space  by ${ \bar { Y } } _ { i }$ as:

$$
a _ { Y } ^ { * } = \arg \operatorname* { m a x } _ { i \in \mathcal { A } } \bar { Y } _ { i } .\tag{29}
$$

The probability that $a _ { _ Y } ^ { \ast }$ equals $a ^ { * }$ is a function of ����:

$$
P r ( a _ { Y } ^ { * } = a ^ { * } ) = f ( t o p K )\tag{30}
$$

When $a _ { \scriptscriptstyle Y } ^ { * } = a ^ { * }$ , our estimator is Q Estimator:

$$
\bar { Y } _ { a * } = \bar { Y } _ { a _ { Y } ^ { * } } = \operatorname* { m a x } _ { i } \bar { Y } _ { i } .\tag{31}
$$

When $a _ { Y } ^ { * } = a ^ { * }$ , our estimator is Double Q Estimator:

$$
\bar { Y } _ { a * } = \bar { Y } _ { a _ { Y ^ { - } } ^ { * } } ,\tag{32}
$$

where $a _ { Y } ^ { * } .$ means an action which is independent of ${ \bar { Y } } _ { i }$

The expectation of our estimator is

$$
\mathbb { E } [ \bar { Y } _ { a ^ { * } } ] = P r ( a _ { Y } ^ { * } = a ^ { * } ) \underbrace { \mathbb { E } [ \bar { Y } _ { a _ { Y } ^ { * } } ] } _ { t e r m 1 } + [ ( 1 - P r ( a _ { Y } ^ { * } = a ^ { * } ) ] \underbrace { \mathbb { E } [ \bar { Y } _ { a _ { Y } ^ { * } - } ] } _ { t e r m 2 } .\tag{33}
$$

For term 1, it is a Single estimator, thus we have

$$
\mathbb { E } [ \bar { Y } _ { a _ { Y } ^ { * } } ] = \mathbb { E } [ \operatorname* { m a x } _ { i } \bar { Y } _ { i } ] \approx \operatorname* { m a x } _ { i } \mathbb { E } [ W _ { i } ] + n ,\tag{34}
$$

Table2:Theaveragereturnperepisodeofdiferentalgorithmsunderdiferentenvironments.Theoptimalaveragedreturnineachcolumnarehighlightedinbold. Thesecondbestresultineachcolumnareunderlined
<table><tr><td rowspan="2">Algorithm</td><td colspan="2">Pixelcopter</td><td colspan="2">Asterix</td><td colspan="2">Breakout</td><td colspan="2">Seaquest</td><td colspan="2">SpaceInvaders</td><td colspan="2">Pong</td></tr><tr><td>Reward</td><td>Improv</td><td>Reward</td><td>Improv</td><td>Reward</td><td>Improv</td><td>Reward</td><td>Improv</td><td>Reward</td><td>Improv</td><td>Reward</td><td>Improv</td></tr><tr><td>DQN</td><td>-0.31</td><td>-117.00%</td><td>5.61</td><td>0.00%</td><td>11.24</td><td>-8.08%</td><td>0.07</td><td>-68.16%</td><td>28.36</td><td>-22.06%</td><td>-14.51</td><td>-7.22%</td></tr><tr><td>DDQN</td><td>-1.28</td><td>-169.33%</td><td>3.47</td><td>-38.20%</td><td>10.95</td><td>-10.43%</td><td>0.15</td><td>-30.99%</td><td>28.02</td><td>-22.98%</td><td>-13.54</td><td>0.00%</td></tr><tr><td>Averaged DQN</td><td>-0.91</td><td>-149.33%</td><td>5.25</td><td>-6.47%</td><td>10.80</td><td>-11.67%</td><td>0.07</td><td>-66.52%</td><td>28.94</td><td>-20.45%</td><td>-14.71</td><td>-8.67%</td></tr><tr><td>Maxmin DQN</td><td>1.73</td><td>-6.20%</td><td>3.16</td><td>-43.66%</td><td>12.23</td><td>0.00%</td><td>0.22</td><td>0.00%</td><td>35.89</td><td>-1.37%</td><td>-13.60</td><td>-0.45%</td></tr><tr><td>Weighted DQN</td><td>-3.34</td><td>-280.85%</td><td>0.92</td><td>-83.60%</td><td>4.82</td><td>-60.56%</td><td>0.19</td><td>-13.14%</td><td>6.87</td><td>-81.13%</td><td>-14.40</td><td>-6.34%</td></tr><tr><td>EBDQN</td><td>-0.59</td><td>-131.86%</td><td>3.42</td><td>-39.02%</td><td>10.33</td><td>-15.51%</td><td>0.15</td><td>-32.93%</td><td>28.65</td><td>-21.26%</td><td>-14.10</td><td>-4.13%</td></tr><tr><td>ACC DDQN</td><td>-1.34</td><td>-172.65%</td><td>2.98</td><td>-47.00%</td><td>8.52</td><td>-30.36%</td><td>0.16</td><td>-28.66%</td><td>36.38</td><td>0.00%</td><td>-14.85</td><td>-9.71%</td></tr><tr><td>Order DQN</td><td>1.85</td><td>0.00%</td><td>3.17</td><td>-43.49%</td><td>11.06</td><td>-9.53%</td><td>0.17</td><td>-24.79%</td><td>35.88</td><td>-1.39%</td><td>-14.28</td><td>-5.49%</td></tr><tr><td>topK=2</td><td>-1.72</td><td>-193.42%</td><td>0.98</td><td>-82.56%</td><td>11.41</td><td>-6.68%</td><td>0.11</td><td>-50.24%</td><td>40.26</td><td>10.65%</td><td>-14.85</td><td>-9.67%</td></tr><tr><td>topK=3</td><td>-2.09</td><td>-213.25%</td><td>8.60</td><td>53.21%</td><td>13.30</td><td>8.79%</td><td>0.48</td><td>119.58%</td><td>38.34</td><td>5.37%</td><td>-14.23</td><td>-5.11%</td></tr><tr><td>topK=4</td><td>0.93</td><td>-49.47%</td><td>8.11</td><td>44.47%</td><td>13.12</td><td>7.26%</td><td>1.03</td><td>366.95%</td><td>34.72</td><td>-4.56%</td><td>-14.49</td><td>-7.01%</td></tr><tr><td>topK=5</td><td>1.11</td><td>-40.04%</td><td>6.32</td><td>12.52%</td><td>12.79</td><td>4.60%</td><td>0.83</td><td>273.93%</td><td>36.65</td><td>0.72%</td><td>-13.83</td><td>-2.16%</td></tr><tr><td>topK=6</td><td>4.25</td><td>129.91%</td><td>6.70</td><td>19.37%</td><td>13.26</td><td>8.46%</td><td>0.68</td><td>206.04%</td><td>36.73</td><td>0.96%</td><td>-14.30</td><td>-5.66%</td></tr><tr><td>topK=7</td><td>8.05</td><td>335.75%</td><td>6.71</td><td>19.56%</td><td>13.90</td><td>13.66%</td><td>0.68</td><td>206.46%</td><td>37.48</td><td>3.02%</td><td>-13.76</td><td>-1.62%</td></tr><tr><td>topK=8</td><td>10.40</td><td>463.45%</td><td>5.99</td><td>6.68%</td><td>13.72</td><td>12.17%</td><td>0.67</td><td>203.93%</td><td>37.05</td><td>1.83%</td><td>-13.06</td><td>3.55%</td></tr></table>

where $n > 0 .$

For term 2, it is a Double estimator, thus we have

$$
\mathbb { E } [ \bar { Y } _ { a _ { Y ^ { - } } ^ { * } } ] \approx \operatorname* { m a x } _ { i } \mathbb { E } [ W _ { i } ] - m ,\tag{35}
$$

where $m > 0$

Therefor,we can estimate the expectation of our estimator by

$$
\begin{array} { r l } & { \mathbb { E } [ \bar { Y } _ { a ^ { * } } ] \approx P r ( a _ { Y } ^ { * } = a ^ { * } ) ( \underset { i } { \mathrm { m a x } } \mathbb { E } [ W _ { i } ] - m ) + [ 1 - P r ( a _ { Y } ^ { * } = a ^ { * } ) ] ( \underset { i } { \mathrm { m a x } } \mathbb { E } [ W _ { i } ] + n ) } \\ & { \qquad = \underset { i } { \mathrm { m a x } } \mathbb { E } [ W _ { i } ] + [ 1 - P r ( a _ { Y } ^ { * } = a ^ { * } ) ] n - P r ( a _ { Y } ^ { * } = a ^ { * } ) m . } \end{array}\tag{36}
$$

Let

$$
[ 1 - P r ( a _ { Y } ^ { * } = a ^ { * } ) ] n - P r ( a _ { Y } ^ { * } = a ^ { * } ) m = 0 .\tag{37}
$$

We get

$$
\begin{array} { r } { P r ( a _ { Y } ^ { * } = a ^ { * } ) = \cfrac { n } { m + n } . } \\ { f ( t o p K ) = \cfrac { n } { m + n } } \end{array}\tag{38}
$$

We can thus conclude that there exists a ���� such that $\mathbb { E } [ \bar { Y } _ { a ^ { * } } ] = \operatorname* { m a x } _ { i } \mathbb { E } [ W _ { i } ]$

## 9. Conclusion

This paper identifies and addresses the challenge of bias amplification in Q-learning within large action spaces. We propose Action Intersection Double Q-learning (AIDQ), a novel method that mitigates the estimation bias by enabling fine-grained, continuous bias control through a simple top-K action intersection mechanism. Our method ofers control over the bias range from negative to positive. Extensive experiments in both tabular and deep RL settings demonstrate that AIDQ consistently outperforms state-of-the-art baselines, particularly in environments with large action spaces. This work opens a new direction for robust bias mitigation in value-based reinforcement learning. Future work focuses on the adaptive parameter ���� tuning.

## Acknowledgments

This work is supported in part by the National Natural Science Foundation of China (Grant No. 62372427, 62476261), in part by the Strategic Priority Research Program of Chinese Academy of Sciences (Grant No. XDB1590300), in part by the Science and Technology Innovation Key R&D Program of Chongqing (CSTB2023TIAD-STX0031, CSTB2025TIAD-STX0023), and in part by the Natural Science Foundation of Chongqing (CSTB2025NSCQ-LZX0061).

## References

Abliz, P., Ying, S., 2022. Underestimation estimators to q-learning. Information Sciences 607, 173–185.

Anschel, O., Baram, N., Shimkin, N., 2017. Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning, in: International conference on machine learning, PMLR. pp. 176–185.

Asadi, K., Littman, M.L., 2017. An alternative softmax operator for reinforcement learning, in: International Conference on Machine Learning, PMLR. pp. 243–252.

Chen, X., Wang, C., Zhou, Z., Ross, K., 2021. Randomized ensembled double q-learning: Learning fast without a model. arXiv preprint arXiv:2101.05982 .

Chu, C., Li, Y., Liu, J., Hu, S., Li, X., Wang, Z., 2022. A formal model for multiagent q-learning dynamics on regular graphs., in: IJCAI, pp. 194–200.

Dulac-Arnold, G., Evans, R., van Hasselt, H., Sunehag, P., Lillicrap, T., Hunt, J., Mann, T., Weber, T., Degris, T., Coppin, B., 2015. Deep reinforcement learning in large discrete action spaces. arXiv preprint arXiv:1512.07679 .

Ghasemipour, S.K.S., Schuurmans, D., Gu, S.S., 2021. Emaq: Expected-max q-learning operator for simple yet efective ofline and online rl, in: International Conference on Machine Learning, PMLR. pp. 3682–3691.

Hasselt, H., 2010. Double q-learning. Advances in neural information processing systems 23.

Jiang, H., Xie, J., Yang, J., 2021. Action candidate based clipped double q-learning for discrete and continuous action tasks, in: Proceedings of the AAAI conference on artificial intelligence, pp. 7979–7986.

Lan, Q., 2019. Gym compatible games for reinforcement learning. https://github.com/qlan3/gym-games.

Lan, Q., Pan, Y., Fyshe, A., White, M., 2020. Maxmin q-learning: Controlling the estimation bias of q-learning. arXiv preprint arXiv:2002.06487 .

Majeed, S.J., Hutter, M., 2021. Exact reduction of huge action spaces in general reinforcement learning, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 8874–8883.

Melo, F.S., 2001. Convergence of q-learning: A simple proof. Institute Of Systems and Robotics, Tech. Rep , 1–4.

Ming, F., Gao, F., Liu, K., Zhao, C., 2023. Cooperative modular reinforcement learning for large discrete action space problem. Neural Networks 161, 281–296.

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 .

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., et al., 2015. Human-level control through deep reinforcement learning. nature 518, 529–533.

Munos, R., Stepleton, T., Harutyunyan, A., Bellemare, M., 2016. Safe and eficient of-policy reinforcement learning. Advances in neural information processing systems 29.

Peer, O., Tessler, C., Merlis, N., Meir, R., 2021. Ensemble bootstrapping for q-learning, in: International conference on machine learning, PMLR. pp. 8454–8463.

Rafailov, R., Hejna, J., Park, R., Finn, C., 2024. From � to �<sup>∗</sup>: Your language model is secretly a q-function. arXiv preprint arXiv:2404.12358 .

Ren, Z., Zhu, G., Hu, H., Han, B., Chen, J., Zhang, C., 2021. On the estimation bias in double q-learning. Advances in Neural Information Processing Systems 34, 10246–10259.

Song, Z., Parr, R., Carin, L., 2019. Revisiting the softmax bellman operator: New benefits and new perspective, in: International conference on machine learning, PMLR. pp. 5916–5925.

Tan, T., Xie, H., Lian, D., 2024a. Adaptive order q-learning, in: Proceedings of the Thirty-Third International Joint Conference on Artificia Intelligence, pp. 4946–4954.

Tan, T., Xie, H., Xia, Y., Shi, X., Shang, M., 2024b. Adaptive moving average q-learning. Knowledge and Information Systems 66, 7389–7417.

Tasfi, N., 2016. Pygame learning environment. https://github.com/ntasfi/PyGame-Learning-Environment.

Thrun, S., Schwartz, A., 2014. Issues in using function approximation for reinforcement learning, in: Proceedings of the 1993 connectionist models summer school, Psychology Press. pp. 255–263.

Towers, M., Kwiatkowski, A., Terry, J., Balis, J.U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., et al., 2024. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032 .

Wang, H., Lin, S., Zhang, J., 2021. Adaptive ensemble q-learning: Minimizing estimation bias via error feedback. Advances in neural information processing systems 34, 24778–24790.

Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R., Kavukcuoglu, K., De Freitas, N., 2016. Sample eficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224 .

Watkins, C.J., Dayan, P., 1992. Q-learning. Machine learning 8, 279–292.

Weisz, G., Budzianowski, P., Su, P.H., Gašić, M., 2018. Sample eficient deep reinforcement learning for dialogue systems with large action spaces. IEEE/ACM Transactions on Audio, Speech, and Language Processing 26, 2083–2097.

Zhai, S., Bai, H., Lin, Z., Pan, J., Tong, P., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., et al., 2024. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems 37, 110935–110971.

Zhang, Z., 2018. Improved adam optimizer for deep neural networks, in: 2018 IEEE/ACM 26th international symposium on quality of service (IWQoS), Ieee. pp. 1–2.

Zhang, Z., Pan, Z., Kochenderfer, M.J., 2017. Weighted double q-learning., in: IJCAI, pp. 3455–3461.

Zhang, Z., Zou, Y., Lai, J., Xu, Q., 2023. M2dqn: A robust method for accelerating deep q-learning network, in: Proceedings of the 2023 15th International Conference on Machine Learning and Computing, pp. 116–120.

Zhao, F., Wang, Q., Wang, L., 2023. An inverse reinforcement learning framework with the q-learning mechanism for the metaheuristic algorithm. Knowledge-Based Systems 265, 110368.