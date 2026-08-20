# Quantum Tensor Network Learning with DMRG

Gustav J L J¨ager<sup>1,2</sup> Martin B Plenio<sup>2</sup> Hans-Martin Rieser<sup>1</sup>

1- Deutsches Zentrum f¨ur Luft- und Raumfahrt e.V. – Institut f¨ur KI-Sicherheit Wilhelm-Runge-Straße 10, 89081 Ulm, Germany

2- Universit¨at Ulm – Institut f¨ur Theoretische Physik and IQST, Albert-Einstein-Allee 11, 89081 Ulm, Germany

Abstract. Tensor Networks are a relatively new machine learning approach. The architectures proposed initially are inspired by approaches from quantum many-body physics simulations. One common layout is the matrix product state (MPS) also known as a tensor train optimized with gradient descent techniques. We introduce a global normalization condition, so that the MPS represents a quantum state. We investigate two optimization methods that find the locally optimal tensors and compare them regarding their efectiveness. One is based on gradient descent and the other on an adaptation of DMRG.

## 1 Introduction

Tensor networks (TN) originate from the field of quantum many-body physics [1, 2]. They are used as a prominent method to simulate quantum systems [3], and approximate their properties such as ground states [4]. Their main advantage lies in their ability to approximate large quantum states with limited storage. Quantum-inspired tensor networks have found an application in classical machine learning models, see [5].

The application of tensor networks to quantum machine learning connects both fields [6]. To raise the whole potential of TN approaches developed in quantum physics it is necessary to transfer the local optimization techniques, namely Density Matrix Renormalization Group (DMRG), to the machine learning context. In this work we explore how this DMRG approach with the underlying Lanczos method to quantum machine learning. One challenge that arises, is that quantum states and quantum channels have normalization conditions due to quantum probability conservation [7].

The basic tensor network ansatz we investigate is inspired by [5]. However, we introduce a normalization condition on the tensor network, such that contracting it yields a normalized vector that may be mapped to a quantum state. We then present how the gradient descent algorithm can be modified to take into account this condition and how a modified DMRG algorithm can be applied for optimization as well. The latter is based on a post processing step taking into account the loss function of the machine learning problem. Finally, we compare how the introduction of the normalization condition impacts the accuracy and the loss, when classifying the MNIST dataset.

## 2 Tensor Network based Machine Learning

Our tensor network approach is based on an MPS similar to the method outlined in [5]. Additionally, we introduce a normalization condition, such that contracting the tensor network yields a normalized vector, see Fig. 1. Inference of this setup can be run directly on a quantum computer: Either checking each label $| l _ { i } \rangle$ separately or maximally entangling system B with a readout site. The generating MPS can be applied as a quantum circuit following a translation method, e.g. the one given by [8].

![](images/94324915ded45166467f68d96fb38db887ec1e502d85c4123009033e4b45bad1.jpg)  
Fig. 1: Architecture of the underlying tensor network. The trainable part of the setup is the MPS, here in blue. The encoded input data of datapoint i of the dataset is represented by the orange tensors. The yellow tensor represents the single site which serves as the output of the label.

To maintain a normalized MPS, we use site canonization while sweeping, which is the process of optimizing the sites iteratively until the MPS converges [3]. This approach is found both in gradient based learning [5] and in DMRG [4]. To deal with complex quantum states, we modify the mean square error loss function from [5] to be

$$
L = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left( \left. f _ { i } \right| _ { B } - \left. l _ { i } \right| _ { B } \right) \left( \left| f _ { i } \right. _ { B } - \left| l _ { i } \right. _ { B } \right) ,\tag{1}
$$

where

$$
| f _ { i } \rangle _ { B } = \left( \langle d _ { i } | _ { D } \otimes \mathbb { 1 } _ { B } \right) | \Psi \rangle _ { D B } .\tag{2}
$$

Here, $| l _ { i } \rangle _ { B } ,$ the $\mathrm { M P S } \ | \Psi \rangle _ { D B }$ , andthe data representation $\left| d _ { i } \right. _ { D }$ are normalized, but the ML output $| f _ { i } \rangle _ { B }$ is not. Given Eq. (2), we obtain the loss

$$
L = \frac { 1 } { 2 } + \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \left. \Psi \right| \left( \left| d _ { i } \right. \left. d _ { i } \right| _ { D } \otimes \mathbb { 1 } _ { B } \right. \left| \Psi \right. - 2 \mathrm { R e } \left[ \left. \Psi \right| _ { D B } \left( \left| d _ { i } \right. _ { D } \otimes \left| l _ { i } \right. _ { B } \right) \right] \right] .
$$

As abbrevations, we define the Hermitian A and the vector b

$$
A = \sum _ { i = 1 } ^ { N } | d _ { i } \rangle \langle d _ { i } | _ { D } \otimes \mathbb { 1 } _ { B } \quad { \mathrm { a n d } } \quad b = \sum _ { i = 1 } ^ { N } | d _ { i } \rangle _ { D } \otimes | l _ { i } \rangle _ { B } .
$$

For local optimization, one needs locally efective variants $A _ { s }$ and $b _ { s }$ on the site s. They are obtained by removing the tensor of the respective site in the MPS and contracting it with A or b, [5, 4]. Both A and $A _ { s }$ are positive semidefinite. $\langle k | b \rangle = 0 { \mathrm { ~ i f ~ } } | k \rangle$ is in the kernel of $A ,$ , otherwise a negative loss could be found, which contradicts Eq. (1). Thus, from now on the kernel of A is removed from $A ,$ when calculating $A ^ { - 1 } \left| b \right.$ . We use the same initialization procedure as [5] for the normalized MPS to avoid vanishing gradients. Furthermore, a lookup table of major building blocks of $A _ { s }$ and $b _ { s }$ is maintained while sweeping. This ensures that the sweep only depends linearly on the number of sites, but at the cost of storage space.

## 3 Gradient descent with Normalization

The local optimization problem we tackle is given by

$$
| \Psi _ { s } \rangle = \operatorname * { a r g m i n } _ { | \Psi \rangle \forall \langle \Psi | \Psi \rangle = 1 } \langle \Psi | A _ { s } | \Psi \rangle - 2 \mathrm { R e } ( \langle \Psi | b _ { s } \rangle ) .\tag{3}
$$

I $\therefore | \Psi _ { s } \rangle$ is not normalized, the optimum is $A ^ { - 1 } \left| b \right.$ , see [9, 10]. Normalizing this optimal solution directly does not yield the optimal solution for the problem with the normalization constraint. This can be seen from the following example:

$$
A _ { s } = { \binom { 2 } { 1 } } \ \mathrm { ~ l a n d ~ } | b _ { s } \rangle = { \binom { 1 } { 3 } } \mathrm { , ~ w i t h ~ } | \Psi _ { s } \rangle = { \binom { 0 } { 1 } } \mathrm { ~ b u t ~ } A _ { s } ^ { - 1 } | b _ { s } \rangle \approx { \binom { 0 . 1 4 } { 0 . 7 1 } }
$$

A simple approach to solve the constraint problem is to use the gradient $| g \rangle =$ $A _ { s } \mathinner { | { \Psi } \rangle } - \mathinner { | { b _ { s } } \rangle }$ to give a small update with step size α on |Ψ⟩ and then normalize afterwards. This is known as projected gradient descent [11]. Thus, we obtain

$$
| \Psi ^ { \prime } \rangle = \frac { | n \rangle } { \sqrt { \langle n | n \rangle } } = \frac { | \Psi \rangle - \alpha | g \rangle } { \sqrt { ( \langle \Psi | - \alpha \langle g | ) ( | \Psi \rangle - \alpha | g \rangle ) } } .
$$

Thus, |Ψ⟩ is updated along the negative gradient on the surface of the unit sphere of $| \Psi \rangle _ { \mathrm { { s } } }$ Hilbert space. One simple improvement of the approach is optimizing the step size α. Setting the derivative of the loss w.r.t. α equal to zero yields the condition

$$
0 = \operatorname { R e } ( \langle g ^ { \prime } | n \rangle ) \operatorname { R e } ( \langle n | g \rangle ) - \operatorname { R e } ( \langle g ^ { \prime } | g \rangle ) \left. n | n \right.\tag{4}
$$

with the gradient $| g ^ { \prime } \rangle$ of $\vert \Psi ^ { \prime } \rangle$ obtained by the next iterative step. We disregard the solution for $\langle n | n \rangle \to \infty$ as this is associated with an infinite step size α. Calculating $| g ^ { \prime } \rangle$ requires one additional matrix-vector multiplication $A _ { s } \mathinner { | { g } \rangle }$ that can be reused to calculate $| \Psi ^ { \prime } \rangle$ after the optimal α is found.

We obtain the derivative of Eq. (4) by using the product rule and the derivatives:

$$
{ \frac { d } { d \alpha } } \left| g \right. = 0 , { \frac { d } { d \alpha } } \left| n \right. = - \left| g \right. , { \mathrm { a n d ~ } } { \frac { d } { d \alpha } } \left| g ^ { \prime } \right. = A _ { s } \left( { \frac { | n \rangle \operatorname { R e } ( \left. n | g \right. ) } { \langle n | n \rangle ^ { \frac { 3 } { 2 } } } } - { \frac { | g \rangle } { \sqrt { \langle n | n \rangle } } } \right) .
$$

We then solve for α with Newton iterations. A local maximum α $\neq \pm \infty$ may exist. To avoid it, $\alpha = 0$ is chosen as the starting point of the Newton iteration.

## 4 A Modified DMRG Algorithm

Optimization techniques based on Krylov subspaces have shown to be far more efective than simple gradient descent [10]. Exemplary methods are conjugate gradient descent algorithm, and the Lanczos algorithm used in DMRG. However, DMRG with the underlying Lanczos algorithm does not involve the second term in Eq. (3), which appears in the machine learning problem. Thus, to modify DMRG to fit this problem, we define a post-processing step after the Lanczos algorithm. First, we compress $A _ { s }$ into $A _ { s } ^ { \prime } = V T V ^ { \dagger }$ with the unitary V and tridiagonal and real $T$ obtained from the Lanczos algorithm. With the fast diagonalization of T [10], we obtain the eigendecomposition of $\begin{array} { r } { A _ { s } ^ { \prime } = \sum _ { i = 1 } ^ { m } e _ { i } \left| \lambda _ { i } \right. \left. \lambda _ { i } \right| } \end{array}$ The starting vector of the Lanczos iteration is chosen to be $\left| b _ { s } \right.$ so that $\left| b _ { s } \right.$ is fully represented in the basis of the Lanczos vectors and thus the eigenbasis of $A _ { s } ^ { \prime } . { } ^ { 1 }$ After compressing, our local efective loss function results in the gradient

$$
x \left. \Psi _ { s } \right. \overset { ! } { = } \left. \mathrm { g } \right. : = A _ { s } ^ { \prime } \left. \Psi _ { s } \right. - \left. b _ { s } \right. .\tag{5}
$$

For the optimal normalized $\left| \Psi _ { s } \right.$ the gradient is not 0 but equal to $x | \Psi _ { s } \rangle$ , with a real scaling factor $x = \pm { \sqrt { \langle \mathrm { g } | \mathrm { g } \rangle } }$ . This ensures the constraint of the local optimum to the unit sphere on which $\left| \Psi _ { s } \right.$ lives, i.e., the gradient is perpendicular to the unit sphere. Eq. (5) can also be derived by Lagrangian optimization with the normalization constraint on $\left| \Psi _ { s } \right.$ . Assuming $\left( A _ { s } ^ { \prime } - x \mathbb { 1 } \right)$ is invertible, $\left| \Psi _ { s } \right.$ is given by

$$
\left| { \Psi _ { s } } \right. = \left( { A _ { s } ^ { \prime } - x \mathbb { 1 } } \right) ^ { - 1 } \left| { b _ { s } } \right. .
$$

Given that $\left| b _ { s } \right.$ is completely decomposable into the $| \lambda _ { i } \rangle$ , we obtain

$$
\mathinner { | { \Psi _ { s } } \rangle } = ( \sum _ { i = 1 } ^ { N } \frac { 1 } { e _ { i } - x } \mathinner { | { \lambda _ { i } } \rangle } \mathinner { \langle { \lambda _ { i } } | } ) \mathinner { | { b _ { s } } \rangle } .
$$

We define $b _ { s i } = \langle b _ { s } | \lambda _ { i } \rangle \langle \lambda _ { i } | b _ { s } \rangle$ , the overlap of the vector $\left| b _ { s } \right.$ and the respective eigenvector. With the normalization condition $\langle \Psi _ { s } | \Psi _ { s } \rangle = 1$ we obtain

$$
1 = \sum _ { i = 1 } ^ { m } { \frac { b _ { s i } } { ( e _ { i } - x ) ^ { 2 } } } .\tag{6}
$$

We recall that $A _ { s }$ and subsequently $A _ { s } ^ { \prime }$ are positive semidefinite. Thus, their smallest eigenvalue is non-negative $0 \leq e _ { \mathrm { m i n } }$ . Eq. (6) suggests multiple solutions $x _ { s } ,$ yet there is only one $x _ { s } < e _ { \mathrm { m i n } }$ because the right-hand side of Eq. (6) is strictly monotone for $x < e _ { \mathrm { m i n } }$ , i.e. $x _ { 1 } < x _ { 2 } \Leftrightarrow \mathrm { R H S } ( x _ { 1 } ) < \mathrm { R H S } ( x _ { 2 } )$

Using the compressed $A ^ { \prime }$ in the loss function Eq. (3), we obtain the loss

$$
L ( x ) = \sum _ { i = 1 } ^ { m } { \frac { e _ { i } b _ { s i } } { ( e _ { i } - x ) ^ { 2 } } } - 2 { \frac { b _ { s i } } { ( e _ { i } - x ) } }
$$

with $0 < e _ { i } , b _ { s i }$ . For $L ( x )$ we find $L ( - | x | ) \leq L ( | x | )$ . Thus, the global minimum can only be obtained by selecting $x _ { s } < 0$ . As established above, there can be only one solution $x _ { s } < 0$ , which thus is the global minimum. In the gradient descent algorithm, we expect the gradient to converge on the same $\left| \Psi _ { s } \right.$ . The solution given by $x _ { s } ~ < ~ 0$ is obtained because the resulting gradient update is positive and thus amplifies the direction $\left| \Psi _ { s } \right.$ over other directions. Thus, the gradient descent algorithm converges on the global minimum.

Without an analytical solution to Eq. (6), we find this $x _ { s } ~ < ~ 0$ by using a Newton iteration. Due to the asymptotical nature of Eq. (6), it only converges if the iteration is initialized with $x _ { s } < x _ { \mathrm { i n i t } } < 0$ . We find $x _ { \mathrm { i n i t } }$ by checking whether the right-hand side of Eq. (6) is larger than 1. As a side note, during the Lanczos iteration, to avoid numerical instability, we substract $a _ { i } | v _ { k } \rangle \langle v _ { k } |$ and $\eta _ { i } ( | v _ { k - 1 } \rangle \langle v _ { k } | + | v _ { k } \rangle \langle v _ { k - 1 } | )$ from A, with entries of the tridiagonal matrix $a _ { i }$ and $\eta _ { i }$ , and the Lanczos vectors |v<sub>k</sub>⟩.

## 5 Experimental validation

![](images/0c876dfe45513262124f09fb46721046f5b7da9170b67739d8c3bafda6ccf7a6.jpg)  
Fig. 2: The loss is displayed over the iterative steps that are given by three sweeps. The modified DMRG method, and the gradient descent with normalization are compared with conjugate gradient descent.

To validate our approach and to compare to standard methods, we apply the standard conjugate gradient descent and our two outlined algorithms to the same subset of 5000 MNIST images that are rescaled to 7x7. 4000 images are used in training and 1000 for testing because the matrix A scales with the squar of the number of images. The three approaches were tested for three full two-site sweeps with a maximum bond dimension of 20. Thus, we have $( 4 9 \times 2 - 1 ) \times 3 = 2 9 1$ total local iteration steps, where each iterative step solves our local optimization problem. Fig. 2 displays the loss over the number of local iteration steps during the sweep. The results for loss and accuracy are given in Tab. 1. We observe that introducing the normalization condition significantly impacts both loss and accuracy. However, the norm of the MPS obtained with conjugate gradient descent is $3 . 9 \cdot 1 0 ^ { 6 }$ . Normalizing this MPS, gives only vanishing overlaps in the loss function Eq. (1), which results in a trivial loss of ≈ 0.5.

<table><tr><td>method</td><td>training loss</td><td>test loss</td><td>training acc.</td><td>testing acc.</td></tr><tr><td>initialization</td><td>0.43646</td><td>0.43984</td><td>67.725%</td><td>63.300%</td></tr><tr><td>conjugate grad. desc.</td><td>0.06487</td><td>0.08680</td><td>96.950%</td><td>94.700%</td></tr><tr><td>norm. grad. desc.</td><td>0.35820</td><td>0.36256</td><td>75.750%</td><td>73.000%</td></tr><tr><td>modified Lanczos</td><td>0.35820</td><td>0.36257</td><td>75.775%</td><td>73.100%</td></tr></table>

Table 1: Overview of accuracy and loss of the diferent methods.

## 6 Conclusion and Outlook

In this work, we take the step to adapt tensor network methods to approach quantum machine learning more eficiently. Implementing the normalization condition is necessary to make the algorithm fit for deployment on quantum computers. However, the results of the validation show that the performance needs to be increased to be competitive with other contemporary algorithms. Clearly, further improvements are needed to achieve parity with classical tensor network methods. To this end, more complex normalization conditions should be studied, like those in quantum channels and improvements in the evaluation of the methods need to be developed. Overall, major research still needs to be done to bring machine learning eficiently to quantum computers.

## References

[1] R. Or´us. A practical introduction to tensor networks: Matrix product states and projected entangled pair states. Annals of Physics, 349:117–158, 2013.

[2] J. C. Bridgeman and C. T. Chubb. Hand-waving and interpretive dance: an introductory course on tensor networks. J. Phys. A: Math. Theor., 50:223001, 2017.

[3] S.-J. Ran, E. Tirrito, C. Peng, L. Tagliacozzo X. Chen, G. Su, and M. Lewenstein. Tensor Network Contractions. Springer Cham, 2020.

[4] U. Schollw¨ock. The density-matrix renormalization group. Rev. Mod. Phys., 77:259, 2005.

[5] M. E. Stoudenmire and D. J. Schwab. Supervised learning with tensor networks. NeurIPS, 29:4799, 2016.

[6] H.-M. Rieser, F. K¨oster, and A. P. Raulf. Tensor networks for quantum machine learning. Proc. R. Soc. A., 479:20230218, 2023.

[7] M. M. Wilde. Quantum Information Theory. Cambridge University Press, 2 edition, 2017.

[8] S.-J. Ran. Encoding of matrix product states into quantum circuits of one- and two-qubit gates. Phys. Rev. A, 101:032310, 2020.

[9] J. Shewchuk. An introduction to the conjugate gradient method without the agonizing pain, 1994.

[10] G´erard Meurant. The Lanczos and Conjugate Gradient Algorithms. Society for Industrial and Applied Mathematics, 2006.

[11] S. Bubeck. Convex optimization: Algorithms and complexity. Foundations and Trends<sup>®</sup> in Machine Learning, 8(3-4):231–357, 2015.

[12] P-G. Martinsson and J. A. Tropp. Randomized numerical linear algebra: Foundations and algorithms. Acta Numerica, 29:403–572, 2020.