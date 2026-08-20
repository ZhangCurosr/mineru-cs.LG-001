# Enhancing EBSD throughput of battery electrode materials using super-resolution generative adversarial networks

John Mangum<sup>1</sup>, Andrew Glaws<sup>1</sup>, Francois Usseglio-Viretta<sup>1</sup>, Steven Spurgeon<sup>1</sup>, Donal P. Finegan<sup>1</sup>

<sup>1</sup>National Laboratory of the Rockies (NLR), 15013 Denver W Parkway, Golden, CO 80401, USA

## Abstract

Quantitative microstructural characterization of Li-ion battery electrode materials using electron backscatter difraction (EBSD) has been proven as a critical method for optimizing cell performance. However, the inherently slow nature of EBSD can hinder the throughput of analyses needed for statistical representation of a material microstructure being developed. This work demonstrates a machine learning super-resolution framework using a generative adversarial network (SRGAN) to significantly increase EBSD throughput. The SRGAN model was trained on EBSD data of $\mathsf { L i N i _ { x } M n _ { y } C o _ { z } O _ { 2 } }$ (NMC) cathode particles to computationally enhance low-resolution datasets and its performance is compared against classical interpolation methods across various upscaling factors (2× to 12×).

Both qualitative image metrics and quantitative microstructural analysis verified that the SRGAN systematically outperformed classical methods, particularly in preserving small grains and maintaining realistic grain boundaries. We demonstrate that a 5× upscaling factor, corresponding to a 25× speed-up in acquisition time or a 25× larger field of view, is practical while maintaining acceptable accuracy in key metrics like grain size and shape. For instance, at 5× upscaling, relative errors were +5.7%, +8.2%, and -14.6% on grain area-equivalent diameter, grain maximum sphereinscribed diameter, and grain boundary length, respectively. The SRGAN methodology developed in this work significantly enhances the eficiency of EBSD acquisition for more statistically robust microstructural dataset, enabling EBSD as a high-throughput characterization tool for materials research and industrial process development.

## Introduction

Lithium (Li) -ion cells are pivotal for energy-storage systems spanning portable electronics and gridscale storage. Since Li-ion cells were first commercialized in the early 1990’s, there has been tremendous progress in increasing their energy-density, power-density, and cycle-life, much of which was achieved by incremental improvements to the chemical, crystallographic, and morphological properties of active materials. Most positive electrodes that are used in the field today for applications like electric vehicles consist of polycrystalline particles with layered crystal structures such as $\mathsf { L i N i _ { x } M n _ { y } C o _ { z } O _ { 2 } }$ (NMC). The morphology and orientations of the crystals within those particles have been shown to greatly influence their performance. This is because each grain facilitates directional Li-ion transport along 2-dimensional planes. Relative orientation between grains determines the tortuosity of the sub-particle Li transport during intercalation and deintercalation, and the crystal’s anisotropic volume changes during operation also create complex strain profiles that can lead to particle cracking, shortening the cycle life of the cell.<sup>1–3</sup> Thus, control of the grain orientation presents an opportunity for improving the rate capability and cycle life of cells. For example, Ren et al.<sup>4</sup> showed dramatic increase in rate capability of NMC particles by radially aligning particles (e.g. from \~30 mAh/g to \~135 mAh/g at a discharge rate of 20C) and Kim et al.<sup>5</sup> showed that by radially aligning elongated grains and by selectively doping the crystal structure with other transition metals, the sub-particle strain profiles and consequent particle cracking can be suppressed, extending the life of the cathode considerably. Crack mitigation can also be achieved by manufacturing single crystal particles,<sup>6</sup> but the rate capability of single crystal particles has been shown to be inferior to their polycrystalline counterparts.<sup>7</sup>

The clear influence of sub-particle grain morphology and orientation of cell performance motivates further research and manufacturing optimization of grain properties. However, to achieve quantitative relationships between grain properties and cell performance, it is essential to be able to quantitatively measure grain shapes and orientation, not just at the single particle level but across many particles from a synthesized batch to achieve statistical representation. Electron backscatter difraction (EBSD) has emerged as the most promising method to measure grain properties and support development of positive electrode particles; this is due to it being a highly accessible lab-based technique that supports collection of high-resolution spatially-resolved crystallographic properties, including morphology, crystal orientation, as well as other metrics like misorientation and strain for many particles.<sup>8–10</sup> The use of focused-ion beam (FIB)-EBSD also enables extending the spatial mapping of sub-particle crystal properties into 3D.<sup>11</sup> However, being a rastering technique, where each pixel in a 2D inverse pole figure (IPF) requires a distinct EBSD measurement, the technique is relatively slow and collecting data across many particles for statistical representativeness is hindered. This poses a challenge for research and development, where high throughput automated measurements are needed to keep pace with material development efort. Furthermore, FIB-EBSD to acquire 3D data that is needed for important transport and morphological metrics in particles is even more time-constrained than simple 2D measurements, particularly considering that achieving statistical representativeness may require hundreds of particles. Even assuming the rollout of 2D to 3D generative modeling<sup>12,13</sup> and large language models,<sup>14</sup> a substantial amount of 2D IPF data is still expected to be needed. However, 2D to 3D generation is expected to be key to enabling statistical representation in particle metrics which will require acquisition of 2D data within a favorable time-frame (e.g. less than half a day to characterize a batch of particles). Accelerating the acquisition of EBSD across 2D cross sections of many particles is the first step towards achieving this. High-throughput EBSD would facilitate better statistics, the possibility of generating representative 3D particles, and would enable faster feedback during material development stages that could accelerate advanced manufacturing. Faster image acquisition than what today’s hardware alone can achieve is needed, which necessitates solutions in software.

Super-resolution refers to the process of computationally enhancing image data to increase the pixel resolution, ideally by injecting meaningful sub-pixel details to the image.<sup>15</sup> Classical approaches to super-resolution (e.g., nearest neighbor, linear/cubic interpolation, etc.) are computationally eficient and do not require any training data. However, these highly general approaches typically result in overly smoothed images that struggle to capture critical small-scale features in the enhanced image. More recently, machine learning methods have provided powerful new capabilities for performing image super-resolution. Specifically, deep learning models comprised of convolutional neural networks (CNNs) constitute the state of the art for image superresolution due to their ability to efectively process image-based data and learn complex mappings between low-resolution (LR) inputs and their high-resolution (HR) counterparts. While simple CNN architectures were originally used for these models,<sup>16,17</sup> research has also explored the use of more sophisticated frameworks, such as generative adversarial networks (GANs)<sup>18,19</sup> and difusion models.<sup>20</sup>

The original application of these super-resolution models focused on photographic images from large-scale curated datasets (e.g., ImageNet<sup>21</sup>) that generally include images of objects, animals, cars, etc. Follow on work has explored the use of super-resolution for scientific data, such as climate and weather data,<sup>22,23</sup> medical imaging,<sup>24,25</sup> astronomical images,<sup>26,27</sup> and microscopy. Research into data-driven super-resolving microscopy data has focused primarily on the SEM imaging.<sup>28,29</sup> For EBSD imaging, physics-based approaches have been proposed to enhance orientation maps.<sup>30</sup> Jung, et al. explored the use of a super-resolution residual network (SRResNet) model to enhance grain orientation and phase data from EBSD imaging.<sup>31</sup> They train models to perform 4x, 8x, and 16x super-resolution on synthetic datasets and examine the results of finite element simulations using true high-resolution and super-resolved data. However, their work does not consider more powerful deep learning approaches, such as GANs, which have been demonstrated to outperform models trained using traditional losses. Additionally, these models were not trained on real datasets that include challenges that would arise in real imaging process, such as noise and stage drift.

In this work, we explore the use of GANs for super-resolution to accelerate EBSD analysis by enabling the use of coarser imaging processes with minimal impact to the overall quality of the results. Specifically, this work examines questions related to the trade-ofs between increasing the degree of synthetic resolution enhancement and the quality of the resulting image and microstructural analysis. To do this, we train super-resolution GAN (SRGAN) models to perform from 2× to 12× pixel resolution enhancement and study the quality of the resulting images. Note that a 4× coarsening of the imaging process results in a 16× speed-up in the overall imaging time. The super-resolution is performed on a combination of band contrast and labeled boundary data, requiring a mixed modeling approach to handle both grayscale and binary outputs simultaneously. We validate the performance of the trained SRGANs for various resolution enhancement across both pixel-based and grain-based metrics and demonstrate a 25× increase in acquisition rate is possible without losing important detail in quantitative microstructural metrics.

## Study overview and general methodology

The goal of this work is to examine the performance and trade-ofs of SRGAN-based data enhancement of EBSD images of NMC particles across a range of resolution enhancement levels. Figure 1 provides a general overview of the data collected and studies performed in this work. EBSD data was gathered for 15 cross-sectioned particles at two diferent resolutions. The high-resolution imaging used 25 nm step sizes (pixel size), and the low-resolution was imaged using 100 nm step sizes. The high-resolution imaging took \~16× the time as the low-resolution imaging (i.e., 16 hours vs. 1 hour). To obtain training and testing data across multiple resolution enhancements, we artificially downscaled, or coarsened, the high-resolution data to varying degrees. The imaged 1/4 resolution data was used as a point of comparison for this downscaling procedure. The lowresolution data was then upscaled, or enhanced, using the SRGAN approach as well as multiple classical approaches. The quality of the enhanced data was examined by comparing to the imaged high-resolution data. The remainder of this section discusses the background and technical details for each of these steps before we explore the results in the next section.

![](images/609ea3ee0382a936316f87680cc65ec3896e5a4fdecc92afc05b22e5651a3edb.jpg)  
Figure 1. A general schematic of the data and processes used in this study. Ground truth imaged data were acquired at full and ¼ resolution. Downscaled data were generated from the full resolution with downscaling factors ranging from 2× to 12×. All downscaled data, as well as the imaged ¼ resolution data, were subsequently upscaled back to full resolution using various methods.

## Experimental data

## Electrode materials:

The electrode used in this work consisted of Targray $\mathsf { L i N i } _ { 0 . 8 } \mathsf { M n } _ { 0 . 1 } \mathsf { C o } _ { 0 . 1 } \mathsf { O } _ { 2 }$ (NMC811) particles. The NMC811 was mixed with Timcal C45 carbon, and solvay 5130 PVDF binder to the weight percentages of 90 wt%, 5 wt%, and 5 wt% respectively. The electrodes were coated to achieve 9.08 mg $\mathsf { c m } ^ { - 2 }$ and calendered to a coating thickness of \~53 µm with a porosity of \~33%. The electrodes were prepared by the Cell Analysis, Modeling, and Prototyping (CAMP) Facility at Argonne National Laboratory.

## Sample preparation and imaging:

NMC811 electrode samples were cut down to approximately 1 cm<sup>2</sup> using scissors and then adhered to a similarly sized Si wafer, to act as a rigid handle, using conductive Ag paint (Ted Pella Inc, Prod. #16062). Smooth cross-sectional faces were then obtained by polishing one edge of the electrode using a JEOL broad beam Ar cross-section polisher (IB-19550CCP) for up to 6 hours with 5 kV Ar ions and \~120 µA current at ambient temperature.

Electron backscatter difraction (EBSD) of the polished cross-sectional faces was acquired using an Oxford Instruments Symmetry EBSD detector mounted on a FEI Nova NanoSEM 630. The samples were mounted on a 20° pre-tilt holder such that the cross-sectional face was pre-tilted to 70° and aligned facing the EBSD detector. An accelerating voltage of 20 kV and beam current of 6.4 nA was used with the sample set at approximately 11 mm working distance. Oxford Instruments AZtec software was used to acquire the EBSD data and export the processed map data (band contrast, Euler angles) and metadata in an HDF5 file format (.h5oina) for further processing and use in training and validating the super-resolution framework. Oxford Instruments AZtecCrystal software was also used to clean up the map data using standard methods and export grain boundary maps.

## Numerical methods

## Image resolution mapping

## Super resolution generative adversarial network

In this work, we implement a super-resolution generative adversarial network (SRGAN) framework to enhance low-resolution EBSD data. In general, the GAN approach trains two competing networks: a generator � that creates new data and a discriminator � that attempts to distinguish between real and synthetic data samples.<sup>32</sup> When performing super-resolution, the generator takes in low-resolution input data and maps it to a corresponding high-resolution output image,

$G \colon \mathcal { X } _ { l o w }  \mathcal { X } _ { h i g h }$ , where $\mathcal { X } _ { l o w }$ and $\mathcal { X } _ { h i g h }$ represent the space of low- and high-resolution images, respectively. The generator model is trained on a combination of two loss functions: the content loss and the adversarial loss. The content loss compares the generated and real high-resolution images in a pixel-wise manner. Here, we are performing super-resolution on images containing both band contrast and boundary data, which mixes continuous and binary data types. To

accommodate this, our content loss combines the mean squared error over the band contrast data with the binary cross entropy loss over the boundary data. The adversarial loss is computed by feedback from the discriminator model. The discriminator is provided with examples of real highresolution images and artificial images generated by � and attempts to classify them appropriately. The adversarial training process drives � to better distinguish between real and fake data and � to produce more realistic images. Mathematically, this is expressed as a minmax optimization,

$$
\operatorname* { m i n } _ { G } \operatorname* { m a x } _ { D } \Big [ \mathbb { E } \log D ( x _ { H R } ) + \mathbb { E } \log \Big ( 1 - D \big ( G ( x _ { L R } ) \big ) \Big ) \Big ]\tag{[1]}
$$

where $x _ { L R }$ and $x _ { H R }$ are low- and high-resolution images. Figure 2 provides a schematic overview of this training process.

![](images/027b7d7b441165e4cd872a4e985150bd438f55745af21c10b8f87b9df4db64c2.jpg)  
Figure 2. Process flow showing how low-resolution data is super-resolved and compared against the ground truth high-resolution data to train the SRGAN model.

For this work, the generator model contains 10 convolutional layers with a 3x3 kernel, step size of one, and hyperbolic tangent activations. The resolution is enhanced the appropriate amount using a depth-to-width pixel shufle layer. The discriminator model contains 7 convolutional layers with the same properties as the generator. The processed image is then flattened and fed through three fully-connected layers with widths of 128, 64, and 1 to predict whether or not a given image is real or generated. The model parameters are updated in an alternating fashion using the Adam optimizer.<sup>33</sup>

## Image downscaling:

One critical feature of SRGAN models is their reliance on representative training data. That is, the model requires corresponding pairs of low-resolution and high-resolution data to learn a statistical mapping between them. In this work, we seek to explore the trade-ofs between the degree of data enhancement (upscaling factor) and the quality of the generated data by training SRGAN models to perform resolution enhancements ranging from 2× to 12× upscaling factors. To avoid collecting paired low- and high-resolution images across all levels of enhancement studied here, we look to artificially downscale, or coarsen, the ground truth high-resolution image data. A common approach in SRGANs literature is to convolutionally average blocks of high-resolution pixels to obtain low-resolution counterparts. However, in the context of EBSD imaging, this procedure does not accurately reflect the nature of collecting coarse data. Instead, reducing acquisition resolution (i.e., increasing step size) merely increases the spacing between acquired pixels without incorporating averaged information from surrounding pixels. To more realistically mimic this process, we take a centered 2x2 or 3x3 block (for even or odd resolution enhancements, respectively) and randomly select a pixel from this block. Figure 3a provides an overview of this process. In Figure 3b, we examine the quality of the downscaling process the imaged low-resolution data. The figure contains a semivariogram, which measures the relative variance in an image as a function of the distance from each pixel. In this plot, we can see that the block-averaged data overly efectively smooths the downscaled imaged, which would result in the SRGAN model learning an incorrect relationship between low- and high-resolution data. The randomly sampled approach matches the level of variance we see in the imaged low-resolution data.

![](images/77504bbd0f2c4613c93eebbf45edc9112f44e1d7759a1305d29d86fd2ef35541.jpg)  
(b)

![](images/6018b567f98a8663092dd5eecd3d4db1390f3bcdbab6235a218e028d5b019a39.jpg)  
Figure 3. (a) Illustration qualitatively showing the artificial downscaling methods and comparison to imaged true low-resolution data. (b) Semivariogram showing the average variance of each lowresolution image as a function of distance from each pixel. Averaged downscaling artificially smooths the resulting low-resolution data and does not as accurately represent the true lowresolution data.

## Grains and grain boundaries identification:

Grains and grain boundaries have been segmented starting from the raw grain boundary map (cf. Fig. 4a). The workflow is illustrated in Figure 4 and explained here.

The field of view (FOV) may contain several particles. To avoid truncated particles that bias the particle size and shape analysis, only the particle located at the FOV’s center must be kept. Therefore, a particle identification workflow is applied. First, the background domain is identified. To do this, the clusters of the complementary domain of the grain boundaries are identified (cf. Fig. 4b) using the MATLAB built-in function bwlabel. Then, the truncated clusters are removed. The union of the grain boundaries and of these removed clusters represent the background (in grey, cf. Fig. 4c). Second, the main particle centered in the FOV is identified. The particle domain is retrieved dilating the grains using the spherebased local dilation method (cf., Fig. 4d).<sup>34</sup> This method ensures the exterior boundary of the particles present in the FOV are not dilated (unlike morphology closing<sup>35</sup>), which prevents creating irrelevant connections between nearby particles. Then, the largest

cluster is identified as the particle of interest (cf. Fig. 4e). Particle shape (circularity and solidity) and size (area-equivalent diameter) are then calculated. The grains of the particle of interest (cf. Fig. 4f) are then identified multiplying the grain cluster map (cf. Fig. 4b) with the particle binary image (i.e., Fig. 4e). The grain boundaries of the particle of interest (cf. Fig. 4g) are then identified multiplying the grain boundary map (cf. Fig. 4a) with the particle binary image (i.e., Fig. 4e). However, the raw boundary map shows variable widths with some nearby grain boundaries being merged together due to the limited image resolution. This prevents calculating the length of the grain boundaries. To remedy this, the grain boundary image is skeletonize using the MATLAB built-in function bwskel, which is based on the medial axis transform.<sup>36,37</sup> Coarse boundaries are either converted to several interfaces or to a single intersection (respectively, red and green rectangles in Figures 4g and h).

![](images/9e22be4a675d973f1fb39ee31f000621eac79e485c649b0264af2bd606be3a1b.jpg)  
Figure 4. Grain and grain boundary identification workflow. From top left to bottom right: (a) raw grain boundary map, (b) black domain clusters (i.e., connectivity), (c) after removing the truncated clusters, (d) cluster local dilation, (e) after keeping only the main particle at the center, (f) grains within the particle of interest (b × e), (g) grain boundaries within the particle of interest (a × e), and (h) grain boundary skeleton. Skeletonization of coarse grain boundary intersections is illustrated for two examples on the bottom right.

[3]

## Image analysis

The subscripts $i , j ,$ and k refer, respectively, to the pixel i, the grain $j ,$ and the particle k. The number of pixels and grains within a particle k is noted, respectively, $N _ { p , k }$ and $N _ { g , k }$ . The number of particles is noted $N _ { P }$

Image comparison (context-agnostic, before segmentation)

This section details generic metrics typically used to compare images which are not necessarily related to the battery field. All these metrics are defined per image (that is per particle k) and are a comparison between the ground truth high resolution image (superscript gt) and the low-resolution image upscaled using one the method investigated in this work (superscript up). Therefore, that can only be calculated for the upscaled images using the ground truth as a reference image. The grey level value of pixel i in particle k is noted $x _ { i , k }$

• Normalized root mean squared error (NRMSE): A typical metric used in many regression-type problems that provides a pixel-per-pixel comparison. For this study, we normalize the error according to dynamic range of the image to aggregate the errors across all of the tested particles more easily (see Eq. 2).

$$
N R M S E _ { k } ^ { u p } = \frac { 1 } { \operatorname* { m a x } \left( x _ { i , k } ^ { g t } \right) - \operatorname* { m i n } ( x _ { i , k } ^ { g t } ) } \sqrt { \frac { \sum _ { i = 1 } ^ { N _ { p , k } } \bigl ( x _ { i , k } ^ { u p } - x _ { i , k } ^ { g t } \bigr ) ^ { 2 } } { N _ { p , k } } }\tag{[2]}
$$

Peak signal-to-noise ratio (PSNR): It is defined as the logarithm of the ratio between the maximum range of the data and the reconstruction error (see Eq. 3). PSNR is usually expressed as a logarithmic quantity as signals may have a very wide range. The signal is interpreted as being the ground truth data, and the noise is the error introduced by the upscaling process. The python scikit-image built-in function peak\_signal\_noise\_ratio was used with default parameters.

$$
P S N R _ { k } ^ { u p } = 1 0 l o g _ { 1 0 } \left( \frac { \operatorname* { m a x } \bigl ( x _ { i , k } ^ { g t } \bigr ) - \operatorname* { m i n } ( x _ { i , k } ^ { g t } ) } { \sum _ { i = 1 } ^ { N _ { p , k } } \bigl ( x _ { i , k } ^ { u p } - x _ { i , k } ^ { g t } \bigr ) ^ { 2 } / N _ { p , k } } \right)
$$

Structural similarity index measure (SSIM): It captures the perceived similarities between two images by comparing the luminance, contrast, and structures (correlation of pixel patterns), which are denoted by $l , c ,$ and $s ,$ respectively. These quantities are weighted by exponents $\alpha , \beta$ , and $\gamma _ { : }$ , and use regularization constants $c _ { 1 } , c _ { 2 }$ , and $c _ { 3 }$ ,

respectively. This is shown in Eq. 4 with � and � the upscaled and ground truth local images, $\mu _ { x } , \mu _ { y } , \sigma _ { x } , \sigma _ { y }$ and $\sigma _ { x y }$ the local means, standard deviations, and crosscovariance).<sup>38</sup> SSIM ranges from -1 to 1, where -1 indicates perfect anti-correlation, 0 no similarity, and 1 perfect similarity. SSIM is calculated using a moving window and is then averaged on the whole image to provide a single value. The python scikit-image built-in function structural\_similarity was used with default parameters.

$$
S S I M _ { k } ^ { u p } = l ( x , y ) ^ { \alpha } c ( x , y ) ^ { \beta } s ( x , y ) ^ { \gamma } \quad \mathrm { w i t h } \quad \left\{ c ( x , y ) = \frac { 2 \mu _ { x } \mu _ { y } + C _ { 1 } } { \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } } \right. \nonumber\tag{[4]}
$$

Normalized mutual information (NMI): It quantifies the information shared between two images based on the statistical dependence of their distributions of pixel values. Exact definition is provided in.<sup>39</sup> It ranges from 1 (perfectly uncorrelated image values) to 2 (perfectly correlated image values, whether positively or negatively). The python scikitimage built-in function normalized\_mutual\_information was used with default parameters.

Note that lower values correspond to better performance for the NRMSE, while higher values correspond to better performance for the other metrics. SSIM and NMI metrics have finite bounds unlike NRMSE and PSNR: $\mathsf { N R M S E } > 0 , \mathsf { P S N R } \in \mathsf { \Gamma } ] - \infty , + \infty [ \mathsf { , S S } \mathsf { I M } \in [ - 1 , 1 ]$ , and NMI ∈ [1,2].

## Image quality (NMC-specific, after segmentation)

This section details quality metrics specific for segmented images with grains and grain boundaries both labelled. It is defined per image (that is per particle k).

In a 2D cross section representation, grain boundaries are a union of lines (excluding the unlikely case for which a grain boundary plane would be perfectly aligned with the point of view and localized exactly on the same depth), while grain and particle areas are, by definition, a surface. However, due to the digitalization of the EBSD measurement, both are represented with pixels, that is surface elements. The grain boundary width in this pixel discretization is expected to be in the same order of the pixel length (the actual width would be the true resolution of the imaging, that is ideally the pixel length but is usually

larger). That is, for an infinite image resolution the cumulative area assigned to all the grain boundaries would be infinitely smaller than the area assigned to the whole particle (grain area plus grain boundary area). Therefore, a simple metric to estimate if the resolution is high enough is to calculate the ratio between these two areas (see Eq. 5, with $\Omega _ { G B , k }$ and $\Omega _ { G , k }$ , respectively, the grain boundary and the grain domains for the particle k). As defined, $I Q _ { k }$ ranges from 0 (most pixels belong to the grain boundaries: low image quality) to 1 (most pixels belong to the grains: high image quality). The image quality average among all the investigated particles, for a given image resolution and upscale method, is noted 〈��〉 (see Eq. 6).

$$
\begin{array}{c} \underbrace { I Q _ { k } = 1 - \sum _ { i = 1 } ^ { N _ { p , k } } G B ( x _ { i } ) } _ { \sum _ { i = 1 } ^ { N _ { p , k } } G B ( x _ { i } ) + G ( x _ { i } ) } \in [ 0 , 1 ] \quad \mathrm { w i t h } \quad \left\{ \begin{array} { l l l } { G B ( x _ { i } ) = 1 } & { \mathrm { i f } x _ { i } \in \Omega _ { G B , k } } \\ { G B ( x _ { i } ) = 0 } & { \mathrm { o t h e r w i s e } } & { \forall k \in [ 1 , N _ { P } ] } \\ { G ( x _ { i } ) = 1 } & { \mathrm { i f } x _ { i } \in \Omega _ { G , k } } \end{array} \right. \qquad [ 5 ]  & { \qquad [ 5 ] } \\ { G ( x _ { i } ) = 0 } & { \mathrm { o t h e r w i s e } } & \\ { \langle I Q \rangle = \frac { \sum _ { k = 1 } ^ { N _ { P } } I Q _ { k } } { N _ { P } } } & { \qquad [ 6 ] } \end{array}
$$

Grain boundaries width $W _ { i , k }$ is calculated using the continuum particle size distribution (c-PSD), or maximum inscribed sphere method.<sup>40</sup> For each pixel of the grain boundary, the diameter of the largest sphere that contains it and does not overlap with the

complementary domain of the grain boundary $( \Omega _ { G B , k } { } ^ { C } )$ is assigned to it (see Eq. 7, with $S _ { d }$ a disc of diameter d). Applied to the channel-like domain such as the grain boundaries, this method provides the channel width (cf. Fig. 5a). As discussed in the previous paragraph, the grain boundary width should be in the same range of the pixel length, while higher widths would indicate a degraded image. The grain with map $W _ { i , k }$ is averaged to provide a mean width for each particle k, $\langle W \rangle _ { k }$ (see Eq. 8). Lastly, the average grain width among all the particles 〈〈�〉〉 is obtained using Eq. 9.

<table><tr><td rowspan=1 colspan=1> $W _ { i , k } = d \Leftrightarrow \left\{ \ L _ { \mathrm { a n d } } \ L _ { \mathrm { \# } } \right. \ L _ { x _ { i } } \in S _ { d }$ and xi ∈ ΩGB,kd such as xi ∈ Sd, ∀i ∈ [1, Np,k] and $\forall k \in [ 1 , N _ { P } ]$ with   ${ S _ { d } } \cap { \Omega _ { G B , k } } ^ { C } = \emptyset \forall d$ </td><td rowspan=1 colspan=1>[7]</td></tr><tr><td rowspan=1 colspan=1>∑iDk ik(W)k =Np,k</td><td rowspan=1 colspan=1>[8]</td></tr><tr><td rowspan=1 colspan=1>∑Ne1(W)k《《w》=    Np</td><td rowspan=1 colspan=1>[9]</td></tr></table>

Except for pulverized particles (for which cracks efectively separate particles in smaller particles), the grain boundary network should be percolating as all grains are in contact with at least another grain. The grain boundary network connectivity is calculated (using the built-in MATLAB function bwlabel) and the number of clusters (ideally 1) is deduced. As well, the ratio of pixels that belong to the largest grain boundary cluster is calculated (ideally 100%).

![](images/ed2bc7d033d49819b7bc2395fa1ce0503f501485ddf3259a9efe3c0b6d0c20e5.jpg)

![](images/420a856bb24a92658ae203631ca7542378fa9c8e1245cb5f55b8be8118dee6c0.jpg)

![](images/f7e5471d471eedd78bcdba7abca5d39073a9c56a1d0756bf43cbd3913e3c21ef.jpg)  
Figure 5. Illustration of (a) grain boundary width $W _ { i , k }$ , (b) grain area-equivalent diameter $D A _ { j , k }$ , and (c) grain maximum inscribed sphere diameter $D S _ { i , k }$

## Grains and grain boundaries quantification (NMC-specific, after segmentation):

This section details metrics specific to the particle/battery field. They provide insights into the performances and degradations of the electrode.

The area of a grain is noted $A _ { j , k }$ . Similarly, for any quantity $\chi$ defined per grain, its value is noted $\chi _ { j , k }$ . The average of $\chi$ for the particle k is noted $\langle \chi \rangle _ { k }$ and deduced using a rule of mixture with $\chi _ { j , k }$ weighted by the grain area $A _ { j , k }$ (cf. Eq. 9). The average of $\chi$ for all the particles investigated is noted $\langle \langle \chi \rangle \rangle$ , and calculated using an unweighted rule of mixture (cf. Eq. 10), and its associated 95% confidence interval, noted $C I ,$ is calculated (cf. Eq. 11, with $\mathsf { r } { = } 0 . 0 2 5$ and $t _ { s }$ the T-score, that is CI is calculated for a +/- 2.5%). The unweighted averaging is justified as EBSD imaging is typically centered in a single particle, and thus it allows comparing metrics calculated on a single particle with the average and CI calculated for the particle library analyzed in this work. $\langle \langle \chi \rangle \rangle$ is calculated separately for each resolution and for each upscale method. The diferent metrics $\chi$ used to characterize the grain size and shape are detailed below.

[13]

<table><tr><td rowspan="2">(x)k =</td><td colspan="2">Ng,k 1 Xj,kAj,k ∑j=1</td><td rowspan="2">[10]</td></tr><tr><td>Aj,k</td><td>∀k ∈ [1, Np]</td></tr><tr><td></td><td colspan="2">∑N=1(x)k 《〈x》&gt;= Np</td><td>[11]</td></tr><tr><td> $C I = t _ { s } ( r , N _ { P } - 1 ) \frac { s t d ( \langle \chi \rangle ) } { \sqrt { N _ { P } } }$ </td><td>with std((x) =</td><td>1 NP ((x&gt;k−《〈x》》)² Np − 1 k=1</td><td>[12]</td></tr></table>

Area-equivalent diameter (DA): For each grain j that belongs to a particle k, a diameter $D A _ { j , k }$ is calculated using a disc-equivalence approach (cf. Eq. 12). This metric is illustrated in Figure 5b.

$$
D A _ { j , k } = 2 \sqrt { \frac { A _ { j , k } } { \pi } } \quad \forall j \in \left[ 1 , N _ { g , k } \right] \mathrm { a n d } \forall k \in \left[ 1 , N _ { P } \right]
$$

Continuous grain size distribution (DS): The same expression defined in Eq. 6 is reused, except it is applied on $\Omega _ { G , k }$ instead of $\Omega _ { G B , k }$ . This diameter represents a characteristic distance that controls the (de)lithiation dynamics within each grain. However, macroscopic Pseudo two-dimensional (P2D) battery models, based on the framework proposed by Doyle and Newman<sup>41</sup>, use the particle diameter to model the (de)lithiation dynamics within each particle, and not the grain diameter. Nevertheless, modelers often fit the particle diameter in a P2D to match experimental data, which results in lower diameters<sup>42</sup> compared to what is calculated for the particle from imaging.<sup>43</sup> For NMC, this can be explained by electrolyte infiltration within the grain boundaries. That is, the lower bound for the NMC diameter to use in a P2D model is the primary grain average diameter.<sup>44</sup> This lower bound diameter is only relevant if particle inter-grain cracking is high enough to have wide enough cracks, and open-porosity cracks, so that electrolyte infiltration can happen. Unlike the area-equivalent diameter DA, DS is not calculated per grain, but per pixel. For non-circular grain DS<DA. This metric is illustrated in Figure 5c.

Diameter distribution: In addition to calculating the average grain diameter per particle, for all the particles, the cumulative function and probability density function are also calculated for $D A _ { j , k = 1 }$ , using the grain area $A _ { j , k = 1 }$ as weights, and for $D S _ { j , k = 1 }$ . The grain diameter distribution is only performed for a single particle for simplicity. It provides a higher granularity for the comparison between the diferent upscale methods.

Grain circularity (�): The shape that has the minimum specific surface area for a given volume is the sphere. Similarly, the shape that has the minimum specific perimeter length for a given area is the disc. Therefore, the area/perimeter ratio indicates the proximity of a 2D shape with a perfect disc. The roundness Θ is used to quantify this proximity and is scaled accordingly so that it is equal to one when applied to a disc, and lower otherwise. Here, the grain circularity $\xi _ { j , k }$ is calculated using the built-in MATLAB function regionprops that is a modification of the roundness numerically more suitable for small objects (cf. Eq. 14, with $P _ { j , k }$ the grain perimeter). A high circularity (sphericity in 3D) is desirable to improve the (de)lithiation homogeneity at constant grain area (volume in 3D), as Lithium concentration within a circular (spherical in 3D) particle is invariant with the azimuth (and also the elevation in 3D), assuming the surrounding electrolyte domain is homogeneous. P2D models assume a maximum sphericity (i.e., spherical particles) which allows to calculate Lithium concentration in the active material particles only through one axis (particle radius) and not three. On the contrary, a low circularity can be beneficial to reduce kinetic resistance as it maximizes interfacial perimeter for the same grain area. Note that for non-ideal geometries (i.e., domains with a non-zero surface roughness) such as the grain shapes of real NMC particles, circularity can be very low despite exhibiting an apparent circular shape due to a high surface roughness that increases the perimeter. However, in this work, this systematic underestimation is solved as circularities are normalized with the value calculated for the initial resolution.

$$
\begin{array}{c} \xi _ { j , k } = \Theta _ { j , k } ( 1 - \frac { 0 . 5 } { r } ) ^ { 2 } \in [ 0 , 1 ] \quad \mathrm { w i t h } \quad \{ \begin{array} { l l } { \Theta = \frac { 4 \pi A _ { j , k } } { { P _ { j , k } } ^ { 2 } } } & \\ { r = \frac { P _ { j , k } } { 2 \pi } + 0 . 5 } \end{array}  \forall j \in [ 1 , N _ { g , k } ] \mathrm { a n d } \forall k \in [ 1 , N _ { P } ]  \end{array}\tag{[14]}
$$

Grain solidity (�): Solidity is defined as the ratio of the grain area over the convex area of the grain and ranges from 0 to 1. The convex area is the area of the convex hull, which is the smallest convex set that contains the grain. A set is convex if any doublets of points taken within the set can be connected with a straight line that only intersects with the domain itself and not with its complementary. Here, the grain solidity $\psi _ { j , k }$ is calculated using the built-in MATLAB function regionprops. Convex shapes such as disc have a solidity of 1. Unlike circularity, non-circular shape can reach the maximum unit value. For instance, a square is not circular but is convex $( \mathfrak { i . e . , \xi } = \pi / 4 < 1 , \psi = 1 )$ . Similarly with circularity, a high solidity is

desirable to improve the (de)lithiation homogeneity. Some shape with a low solidity (and low circularity) can be decomposed into smaller higher solidity shapes, which for electrode microstructure can be used to identify particles.<sup>34</sup> The combination of circularity and solidity provides an exhaustive characterization of the grain shape.

Lastly, grain boundaries are the preferential locations for crack initiation during cycling.<sup>1</sup> Grain boundary density could be evaluated directly by normalizing the grain boundary area (cf. Fig. 4g) with the particle area (cf. Fig. 4e). However, this area-based approach significantly overestimates the grain boundaries for the coarse resolutions in addition to being dimensionally incorrect. Instead, we choose to normalize the grain boundaries cumulative length using the skeleton representation. However, the length-based approach is biased by the pixel discretization of the skeleton image. That is, a pixel attributed to the skeleton can represent diferent lengths. To remedy this, a length-calculation algorithm based on the pattern recognition of the point of contact between adjacent grain boundaries is reused from previous work<sup>45</sup> and is illustrated in Figure 6.

![](images/5ea504452e41e8953c4e47041a8e8ee045953eda1e5508a24f3ef9d60fa5cc9f.jpg)  
Figure 6. Grain boundary length calculation from the skeleton representation. Lengths are expressed in voxel length for this illustration.

## Results & Discussion

## Super resolved image comparison

In this section, we examine the trade-ofs between the level of resolution enhancement from 2× to 12× and the quality of the resulting images. We compare the performance of the SRGAN-based enhancement against several classical approaches, including nearest neighbor, linear, and cubic interpolation as well as an unsharp masking approach, which attempts to enhance image data while preserving hard edges and textures. We aggregate metrics for the quality of the enhancement over all 15 particles for robustness. Since the SRGAN model requires training data, we train 15 distinct SRGAN models with each model holding out a diferent particle for testing. Each model is trained for 2,000 epochs using the Kestrel high-performance computing system at the National Laboratory of the Rockies.

Figures 7 and 8 provide qualitative comparisons between the resulting band contrast and boundary images, respectively, from the various resolution enhancement methods for a representative particle alongside the true high-resolution image. The band contrast images in Figure 7 show that every method studied here produces relatively high-quality images at a 2× resolution enhancement. For a 4× enhancement, the classical methods begin to show non-physical artifacts, such as pixelization or blurring, while the SRGAN image continues to maintain a high quality. By 8×, all grain structure has been lost in the classical approaches. The 8× SRGAN output still manages to produce reasonable grain structures, although some of the smaller grains have lost their shape or disappeared entirely. For the boundary data in Figure 8, we see that the classical super-resolution methods begin to falter as early as 2×. While these methods still produce grain structures in the super-resolved data, the boundaries between the grains are uncharacteristically thick and the grains themselves have already lost some of their actual shape. Similar to the band contrast data, all grain structure has been lost by 8× using these methods. The SRGAN approach manages to produce reasonable boundary data for 2× and 4× enhancements and the 8× enhancement still manages to produce decent reconstructions of the high-resolution boundaries. Further, comparing the band contrast data in Figure 7 and the boundary data in Figure 8, we can see that the SRGAN method produces consistent structures that match grains with boundaries between the two images. Conversely, there is little or no correlation in the structures obtained between the two images for the traditional super-resolution approaches. This highlights the added value of generating both super-resolved images simultaneously using the SRGAN approach.

![](images/cfd536ebbfe2b020c4400b26939b68c6f8f31188dd9030c8ea2e3b06642b00de.jpg)  
Figure 7. Side-by-side comparison of the super-resolved band contrast images using various methods for a 2×, 4×, and 8× enhancement. Red boxes provide a zoomed-in view of the particle grain structure. The imaged high-resolution particle is shown on the right for comparison.

![](images/a479a9712bb12a24fb83d667072552b4ba907c1b21a27cc4c92724537e78e977.jpg)  
Figure 8. Side-by-side comparison of the super-resolved boundary data using various methods for a 2×, 4×, and 8× enhancement. Red boxes provide a zoomed-in view of the particle grain structure. The imaged high-resolution particle is shown on the right for comparison.

## Generic image metrics

Next, we provide a more quantitative examination of the various super-resolution methods across the range of enhancements. These analyses include both image-based metrics, which focus on the enhanced band contrast data, and morphological criteria, which

![](images/9915569b75b7618bf8b5a9241df779f0b0d5e49de62a3dc8198d9b3c06d09ecf.jpg)

examine the grain structures obtained from the boundary data. The pixel-wise metrics we consider are the normalized root mean squared error (NRMSE), the peak signal-to-noise ratio (PSNR, the structural similarity index (SSIM), and the normalized mutual information (NMI). The NRMSE is a typical metric used in many regression-type problems. For this study, we normalize the error according to dynamic range of the image in order to enable us to aggregate the errors across all of the tested particles more easily. The PSNR is a popular metric for image analysis studies. It is defined as the logarithm of the ratio between the maximum range of the data and the reconstruction error. The SSIM captures the perceived similarities between two images by comparing the luminance, contrast, and structures. NMI quantifies the information shared between two images based on the statistical dependence of their distributions of pixel values. Figure 9 shows the averages of these metrics across the various methods and resolution enhancements. Note that lower values correspond to better performance for the NRMSE, while higher values correspond to better performance for the other metrics.

Figure 9. The average of various image-based metrics for each of the super-resolution methods as a function of the level of resolution enhancement. Note that we also provide the result of applying the SRGAN approach to the imaged low-resolution data for the 4x enhancement, denoted by the × in each plot.

Figure 9 shows mean and standard deviation of these image-level metrics for each of the super-resolution approaches across the range of enhancement levels we considered. Across all metrics, we see that the SRGAN outperforms the classical methods for nearly every level of enhancement. At a 2× resolution enhancement, the performance converges slightly across the methods, especially for the SSIM. For the small amount of resolution enhancement being performed here, interpolation is a reasonable approach. However, as the resolution enhancement increases, the SRGAN quickly stands out as the best approach. Notably, there is no specific level of enhancement at which the super-resolution process suddenly drops of despite the clear breakdown in grain structures that we observed in the qualitative results from Figure 7. This highlights the limitation of the imagelevel metrics to characterize the quality of physically relevant features in data. We also include the result from applying the real imaged low-resolution data to the trained SRGAN model to further validate the downscaling process discussed earlier. Here, we see that the enhanced output conditioned on the imaged low-resolution data aligns well with the quality of the results obtained using the artificially downscaled images at the same enhancement level. This consistency provides confidence that the proposed methodology is robust and can be deployed in real workflows.

## Image quality

The image quality metric (cf. Eq. 5,6) defined specifically for segmented particles with their grain boundaries identified is shown in Figure 10a. The ground truth images have on average an image quality of 0.645 +/- 0.019 (95% confidence interface). SRGAN systematically significantly outperforms all the other upscaling methods considered in this work. With SRGAN, upscaling first causes a slight decrease of the image quality. A lower plateau, \~0.59, is reached for the 5× to 7× upscale levels. This initial decrease is expected as the grain width (and thus the grain cumulative area) is overestimated at low resolution, and the upscaling can only partly retrieve the ground truth thinner with (cf. Fig. 10c). For higher upscaling levels (8× and above), the image quality is increasing and eventually surpasses the ground truth value. This increase is, however, misleading: it does not correspond to an actual improvement of the image but is due to the grain boundary

domains being removed as the image is severely degraded (cf. Fig. 10d), to the point metrics defined for an NMC particle are getting meaningless. The resolution for which this degeneration of the image occurs is also indicated by the number of grain boundary clusters, and especially the relative size of the largest one. Ground truth images have roughly 10 grain boundary clusters on average, however the largest one represents more than 99% of the total grain boundary area, that is, the grain boundary network is fully connected, even when observed in a cross section (cf. Fig. 10b). SRGAN shows the share of the largest grain boundary cluster starts to decline for upscaling higher than 6×. This analysis indicates that an upscaling from a resolution up to between 125 and 150nm (that is 5× and 6×) preserves the image quality with SRGAN.

The other upscaling methods can be categorized into two groups based on these image quality metrics. First, the linear and cubic interpolations, and the unsharp mask methods provide very similar image results. Second, the nearest neighbor method provides the worst outcome. They all have in common a sharp decline in quality from the first level of upscaling (2×): from 0.645 (ground truth) down to 0.443-0.427 for the first group, and down to 0.379 for the nearest neighbor method, way lower than what is achieved with SRGAN (down to 0.624). Poor performance for the baseline methods is due to a significant coarsening of the grain boundaries (cf. Fig. 10d), which result in very wide, unrealistic, grain boundary widths (cf. Fig. 10c), and thus a low image quality.

(a)  
![](images/af1da57e7108e784b335b0840891180e78e30b6eaba3f6a7cf72d974d78732d5.jpg)

![](images/500fda1a69d3ccc12f8d804c2ad354cd3bcdfb70884e2090e92beebfd8b888f2.jpg)

![](images/b70e2330868d3d63af0139bf92beb7599cf81171e74a0d21424127e5ee99968f.jpg)

![](images/0e19057f2aecabb1f86e6d6d1b434fc5e4daada8d3846f97940863403e0acfab.jpg)  
Figure 10. (a) Image quality averaged among all the particles calculated for the ground truth images and the images upscaled with diferent methods, for degraded resolution ranging from 2× to 12×. (b) Number of grain boundary clusters, and ratio of the largest one. (c) Grain boundary diameter (maximum sphere-inscribed method). (d) Visualization of the grain boundary coarsening and disconnection occurring at high upscaling levels (videos available in Supplementary Information show all particles, with all resolutions, and all methods).

## Quantification of morphological properties

## Grain size and numbers:

Figure 11a shows the number of grains normalized with the ground truth images for the diferent upscaling methods. All methods, except for SRGAN, exhibit a sharp reduction with the first downscaling-upscaling level (2×), retaining \~48% of the initial number of grains. On the contrary, SRGAN managed to preserve 80% of the grain number at this resolution. SRGAN’s advantage is systematic for all the resolutions investigated, with for instance a 44% retention at 6× compared to 16% for all the other methods. Among the baseline methods considered in this work, the nearest neighbor approach is the worst, while the three other approaches (linear and cubic interpolations, and unsharp mask) are roughly similar.

Despite SRGAN losing 20% of grains at the first upscaling level, the penalty is much more contained for the grain average diameters: only 5.1% and 1.02% higher, respectively, for the maximum sphere-inscribed method (cf. Fig. 11b) and the area-equivalent method (cf. Fig. 11c,d). This indicates that most of the grain information lost during the downscalingupscaling scheme is concentrated on the smallest grains. These small features correspond either to the cross sections of small grains or out-of-plane tips of larger grains. If a 10% degradation threshold is considered, SRGAN preserves the maximum sphere-inscribed diameter up to a 6× scaling (+8%) and the area-equivalent diameter up to a 5× scaling (+5.5%). Above these scaling, the error is above 10% and degrades further as the grain boundary disconnection occurring at these degraded resolution result in larger grains (cf. Fig. 10b,d).

While SRGAN tend to systematically overestimate the diameters, the other methods first underestimate them before eventually also overestimating them. This diferent trend is due to the grain boundary coarsening, absent in SRGAN, but systematic in the other approaches (cf. Fig. 10c,d). Indeed, as the boundaries are getting wider, the grains in between are shrinking, and the shrinking of the large grain overshadow the removal of the smallest grains. Additionally, the higher the upscaling is, the higher are the variations between each upscaled particles for all the methods considered in this work.

![](images/674feae768d0f10726c4b65a2e6233c4cbfcbdb67e5bc337fde78d02e3b0e678.jpg)

![](images/486f1b11950b10f39df2bf4ca4d9357217725ca79dfe054ba9b7cfbbc208b831.jpg)

![](images/6d8b669db28f491c02024e38da5c4d294f083acf2d0134fc378d296a9056ebd9.jpg)

![](images/9f610b8b339727cccc6772acf07edfae0afbfb3c0f40a78078b9b65520c2bcf3.jpg)  
Figure 11. (a) Number of grains, grain average diameter calculated with (b) the maximum inscribed sphere method and (c, d) the area-equivalent method, for images upscaled with diferent methods, for degraded resolution ranging from 2× to 12× and normalized with the values of the ground truth image.

To further evaluate the loss of information induced by the coarse resolutions, the grain size distributions are plotted for a single representative dataset for all the methods in Figure 12 for a selection of resolutions. Up to 4×, all methods show similar area-equivalent diameters (cf. Fig. 12, top row), except for the small diameters for which SRGAN outperforms the other upscaling approaches. At 5×, SRGAN is still relatively close with the ground truth while all other methods start to deviate significantly in favor of larger diameters. At 6×, SRGAN is significantly of. At 7× and above, the grain boundary network is losing its connectivity (cf. Fig. 10b) and the grains are merging even more (cf. Fig. 10a), which results in unrealistic large grain area-equivalent diameters. The maximum sphere inscribed diameter distributions are much better preserved compared to the areaequivalent diameter distributions, even at very high upscaling (cf. Fig. 12, bottom row). This is because the sphere inscribed method is hardly afected by the grain boundary percolation. The main issue lies in the strong underestimation of the grain volume assigned with the small diameters, systematic for all methods but especially substantial for the linear and cubic interpolation approaches. Here again, SRGAN provides the best

estimation of the small diameters among all the methods investigated. Figure 13 shows the cumulative and probability density functions for the same data set, but only for SRGAN. The area-equivalent diameters distributions are preserved up to a 5× upscaling. The maximum sphere inscribed diameter distributions are mostly a rebalancing between the smaller and the larger diameters.

![](images/249d0e690b458910eed766adae1c9c03d9dadb1ec8d8c0974cb201d551252b7d.jpg)  
Figure 12. Cumulative functions of (top row) grain area-equivalent diameter and (bottom row) grain maximum sphere inscribed diameter calculated for a representative dataset, for the ground truth image and for images upscaled with all the upscaling methods. Inserts show the distributions for the 10% grain domain associated with the smallest diameters.

![](images/5f33c66f8b71fe656f75b782066a98c54d3b54234b4e51dbe48d925b3b5ccce4.jpg)  
Grain equivalent diameter (nm)

![](images/868eac113ad01fbc4d672d35cd3a6d299163487abf364084a1a5e3a4a5d5e108.jpg)

![](images/a832ace70067ba6d193dc49f20e484af5e84de0e0c61b63102f47e7255d9bb94.jpg)

![](images/d016434dcda56669b5cb48d06b09f16cbdfae4ba8a61aa19affc964391474f89.jpg)  
Grain maximum sphere inscribed diameter (nm) Grain maximum sphere inscribed diameter (nm)  
Figure 13. Grain area-equivalent diameter (a) cumulative functions and (b) probability density functions, and grain maximum sphere inscribed diameter (c) cumulative functions and (d) probability density functions, calculated for a representative dataset, for the ground truth image and for images upscaled with the SRGAN method only. (a, c) Inserts show the distributions for the 10% grain domain associated with the smallest diameters.

## Grain shape:

Figure 14 shows the grain shape metrics calculated for both methods and normalized with the ground truth images. SRGAN preserves shape (both circularity and solidity with an error below \~10%) for an upscaling up to 5× to 6×. SRGAN slightly overestimates these two metrics (i.e., the methods tend to recreate slightly more convex grain shapes). Above a 6× scaling, solidity starts to degrade significantly and circularity variability increases. The sharp decrease calculated for solidity is coherent with the associated sharp increase in the grain area equivalent diameter (cf. Fig. 11c) and the grain boundary connectivity loss (cf.

Fig. 10b). Indeed, as the grain boundary network is losing its percolation, adjacent grains previously separated by a grain boundary get in contact which results in a new grain, much larger (area-wise) and with a highly concave shape. For low to intermediate upscaling (up to 4×), linear and cubic interpolations and unsharp methods provide a better representation of the grain shape compared to SRGAN. The nearest neighbor method provides the worst shape degradation.

![](images/0cde23dd80f43a0ea9e589d6bce617df3674c8671c2402e8496b58ba577552cd.jpg)

![](images/2fc5c2cc7a9ccd0726d632788667810338a21411c4817b34066a66496e4a5717.jpg)  
Figure 14. Grain shape analysis performed with two metrics, (a) grain circularity and (b) grain solidity, calculated for images upscaled with diferent methods, for degraded resolutions ranging from 2× to 12×, and normalized with the ground truth image. Note that both metrics can go higher than 1 as the values reported here are normalized with the values calculated for the ground truth images.

## Grain boundary length density:

Figure 15 shows the grain boundary cumulative length density calculated for all methods and normalized with the ground truth images. A significant reduction is calculated for all approaches as the number of grains is decreasing with the upscaling level (cf. Fig. 11a). The reduction is also expected as surface roughness increases with finer resolutions. SRGAN outperforms the other methods (except for very high upscaling, above 8×). SRGAN manages to get 93.0% and 80.1% of the initial grain boundary length density, respectively, at 2× and 6×. If a 10% degradation threshold is considered, SRGAN preserves the grain boundary length density up to 3×. Above this scaling, the error is above 10% and degrades further linearly, with the degradation slope increasing significantly at $5 \times$ to 6×.

Among the baseline methods considered in this work, the nearest neighbor approach is the best, while the three other approaches (linear and cubic interpolations, and unsharp mask) are roughly similar. Unlike the other metrics previously investigated, the nearest neighbor method is a worthy choice, especially considering its simplicity. This is due to its excellent preservation of the grain boundary network, that stays connected even at very high upscaling (cf. Fig. 10b). Especially, the nearest neighbor method overcomes SRGAN above 8× (albeit, for such high upscaling, the error is very large, with a -40% underestimation of the ground truth value), which corresponds to the upscaling level when SRGAN no more preserves the connectivity of the grain boundary network (cf. Fig. 10b).

![](images/71e8c9d40368e7e4aa6d3e593d8fb9e4103aea8d50e1e2ded9276cf4f8034b01.jpg)  
Figure 15. Grain boundary length calculated for particles upscaled with diferent methods, for degraded resolution ranging from 2× to 12×, and normalized with the ground truth images.

## Demonstration of high-throughput wide-area EBSD

To demonstrate the capability and throughput enhancement of the SRGAN model developed in this work, a large-area EBSD dataset was acquired at 100 nm/pixel resolution, encompassing approximately 35 cathode particles contained within an electrode crosssection. The total acquisition time for the dataset was 40 minutes. For reference, the same dataset acquired with a resolution of 25 nm/pixel would take approximately 11 hours. The performance of the SRGAN model, shown qualitatively in Figure 16, demonstrates notable enhancement in the clarity of the band contrast map along with a significant improvement in the resolution of the segmented grain boundary map, which is critical for extracting quantitative microstructural metrics from these EBSD datasets. The time savings aforded by the SRGAN framework enables the acquisition and analysis of a substantially greater

number of data points, and therefore microstructural features, thereby providing more robust and statistically accurate representations of the material being studied in less time than traditional methods.

Band Contrast Image

Imaged Low-Resolution  
![](images/a33ac3be6a1f65580f0cf37b6862f6a5059d12f74bcda9ad1086a8a0292549d5.jpg)

SRGAN High-Resolution  
![](images/5d6c3e28041bcf26b7ec248e696deb0089b51301ebe55b38541b0cdaf6e8aee1.jpg)  
50 μm

Segmented Grain Boundaries  
![](images/137a884d5f4eda7e9808d8f11624151f4fcd45cef5f8372bd0acbf8887ef9617.jpg)

![](images/e8a7f4ddf60a12f7e26e1f78f8c1ad679955676f9b0c4382017ca0395837dbf4.jpg)  
Figure 16. Representative large field of view cross-sectional EBSD map of the same NMC material demonstrating the SRGAN methodology developed in this work. Qualitative comparison between the band contrast (top row) and grain boundary (bottom row) maps of imaged low-resolution (left) and super-resolved (right) datasets demonstrates the improvement in microstructural clarity, notably in the segmented grain boundary map.

## Conclusions

This study demonstrates the application of a super-resolution generative adversarial network model trained on electron backscatter difraction imaging data to significantly improve throughput of EBSD characterization. The developed SRGAN model was applied to 15 EBSD datasets across super-resolution upscaling factors ranging from 2× to 12× and compared to classical superresolution methods, such as linear interpolation. Quantitative microstructural analysis verified that the SRGAN systematically outperformed the other methods considered in this work. Especially,

SRGAN better preserves the small grains, the thin width of the grain boundaries, and the grain boundary network percolation. For an arbitrary 10% error threshold, SRGAN preserves grain size and shape up to a 5x upscaling, and the grain boundary length density up to a 3x upscaling. The latter being more penalized as surface roughness is poorly described at low resolutions. Since the grain size is related to the characteristic difusion length used in P2D representations to model the (de)lithiation dynamics within the active material particles (assuming electrolyte infiltration within the grain boundaries and/or cracks), it can be considered that the errors are acceptably constrained up to 5× upscaling. That is, the model systematically predicts reasonably accurate metrics across all datasets for SRGAN images generated from 125 nm/pixel resolution images when compared to corresponding 25 nm/pixel ground truth images. These results highlight the potential of deep learning–based GAN models to significantly reduce acquisition times (25× faster) and/or increase practical fields of view (25× larger) without substantial loss of quantitative accuracy.

Beyond the demonstrated case, this methodology is broadly adaptable to other material systems and EBSD modalities where high temporal or spatial throughput is critical. Potential applications include accelerated mm-scale characterization of hydrogen fuel cell electrode architectures, thinfilm photovoltaic microstructures (e.g., CdTe), and complex microstructures of additively manufactured components. In situ EBSD experiments stand to benefit in particular, as the enhanced temporal resolution aforded by super-resolution would enable more detailed monitoring of microstructural evolution during environmental stressing. This framework would also greatly speed up the EBSD acquisition steps in 3D-FIB-EBSD workflows, which are typically the ratelimiting steps, thereby enabling larger volumes to be studied more eficiently.

In summary, this work demonstrates that the adoption of GAN-based super-resolution methods could substantially advance EBSD as a high-throughput, high-fidelity microstructural characterization tool for materials science research and industrial process monitoring.

## Acknowledgements

This work was authored by the National Laboratory of the Rockies (NLR), for the U.S. Department of Energy (DOE) under Contract No. DE-AC36–08GO28308. A portion of this research was performed using computational resources sponsored by the U.S. Department of Energy's Ofice of Critical Minerals and Energy Innovation and located at the National Laboratory of the Rockies. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

## References

1. Allen, J. M. et al. Quantifying the influence of charge rate and cathode-particle architectures on degradation of Li-ion cells through 3D continuum-level damage models. J. Power Sources 512, 230415 (2021).

2. Parks, H. C. W. et al. Non-linear cracking response to voltage revealed by operando Xray tomography in polycrystalline NMC811. EES Batter. 1, 482–494 (2025).

3. Heenan, T. M. M. et al. Identifying the Origins of Microstructural Defects Such as Cracking within Ni-Rich NMC811 Cathode Particles for Lithium-Ion Batteries. Adv. Energy Mater. 10, 2002655 (2020).

4. Ren, D. et al. Ultrahigh Rate Performance of a Robust Lithium Nickel Manganese Cobalt Oxide Cathode with Preferentially Orientated Li-Difusing Channels. ACS Appl. Mater. Interfaces 11, 41178–41187 (2019).

5. Kim, U.-H. et al. Heuristic solution for achieving long-term cycle stability for Ni-rich layered cathodes at full depth of discharge. Nat. Energy 5, 860–869 (2020).

6. Zhang, N. et al. Long-Term Cycling and Mechanisms of Cell Degradation of Single Crystal LiNi0.95Mn0.04Co0.01O2/Graphite Cells. J. Electrochem. Soc. 171, 010520 (2024).

7. Li, J. et al. Comparison of Single Crystal and Polycrystalline LiNi0.5Mn0.3Co0.2O2 Positive Electrode Materials for High Voltage Li-Ion Cells. J. Electrochem. Soc. 164, A1534 (2017).

8. Quinn, A. et al. Electron Backscatter Difraction for Investigating Lithium-Ion Electrode Particle Architectures. Cell Rep. Phys. Sci. 1, 100137 (2020).

9. Fink, K. et al. Application of electron backscatter difraction techniques to quantify efects of aging on sub-grain and spatial heterogeneity in NMC cathodes. Energy Storage Mater. 44, 342–352 (2022).

10. Wang, W. et al. Visualization and quantification of lattice strain in battery cathode particles through electron backscatter difraction imaging. Nat. Commun. 17, 1421 (2026).

11. Furat, O. et al. Mapping the architecture of single lithium ion electrode particles in 3D, using electron backscatter difraction and machine learning segmentation. J. Power Sources 483, 229148 (2021).

12. Kench, S. et al. Li-ion battery design through microstructural optimization using generative AI. Matter 7, 4260–4269 (2024).

13. Dahari, A., Docherty, R., Kench, S. & Cooper, S. J. Prediction of Microstructural Representativity From A Single Image. Adv. Sci. 12, e14149 (2025).

14. Lei, G., Docherty, R. & Cooper, S. J. Materials science in the era of large language models: a perspective. Digit. Discov. 3, 1257–1272 (2024).

15. Anwar, S., Khan, S. & Barnes, N. A Deep Journey into Super-resolution: A Survey. ACM Comput Surv 53, 60:1-60:34 (2020).

16. Dong, C., Loy, C. C., He, K. & Tang, X. Learning a Deep Convolutional Network for Image Super-Resolution. in Computer Vision – ECCV 2014 (eds Fleet, D., Pajdla, T., Schiele, B. & Tuytelaars, T.) 184–199 (Springer International Publishing, Cham, 2014). doi:10.1007/978-3-319-10593-2\_13.

17. Dong, C., Loy, C. C., He, K. & Tang, X. Image Super-Resolution Using Deep Convolutional Networks. IEEE Trans. Pattern Anal. Mach. Intell. 38, 295–307 (2016).

18. Ledig, C. et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. 2017 IEEE Conf. Comput. Vis. Pattern Recognit. CVPR 105–114 (2017) doi:10.1109/CVPR.2017.19.

19. Wang, X. et al. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. in Computer Vision – ECCV 2018 Workshops (eds Leal-Taixé, L. & Roth, S.) 63–79 (Springer International Publishing, Cham, 2019). doi:10.1007/978-3-030-11021-5\_5.

20. Li, H. et al. SRDif: Single image super-resolution with difusion probabilistic models. Neurocomputing 479, 47–59 (2022).

21. Deng, J. et al. ImageNet: A large-scale hierarchical image database. in 2009 IEEE Conference on Computer Vision and Pattern Recognition 248–255 (2009). doi:10.1109/CVPR.2009.5206848.

22. Stengel, K., Glaws, A., Hettinger, D. & King, R. N. Adversarial super-resolution of climatological wind and solar data. Proc. Natl. Acad. Sci. 117, 16805–16815 (2020).

23. Buster, G., Benton, B. N., Glaws, A. & King, R. N. High-resolution meteorology with climate change impacts from global climate model data using generative machine learning. Nat. Energy 9, 894–906 (2024).

24. Chaudhari, A. S. et al. Super-resolution musculoskeletal MRI using deep learning. Magn. Reson. Med. 80, 2139–2154 (2018).

25. You, C. et al. CT Super-Resolution GAN Constrained by the Identical, Residual, and Cycle Learning Ensemble (GAN-CIRCLE). IEEE Trans. Med. Imaging 39, 188–203 (2020).

26. Dou, F., Xu, L., Ren, Z., Zhao, D. & Zhang, X. Super-resolution of Solar Magnetograms Using Deep Learning. Res. Astron. Astrophys. 22, 085018 (2022).

27. Sweere, S. F. et al. Deep learning-based super-resolution and de-noising for XMMnewton images. Mon. Not. R. Astron. Soc. 517, 4054–4069 (2022).

28. de Haan, K., Ballard, Z. S., Rivenson, Y., Wu, Y. & Ozcan, A. Resolution enhancement in scanning electron microscopy using deep learning. Sci. Rep. 9, 12050 (2019).

29. Hagita, K., Higuchi, T. & Jinnai, H. Super-resolution for asymmetric resolution of FIB-SEM 3D imaging using AI with deep learning. Sci. Rep. 8, 5877 (2018).

30. Jangid, D. K. et al. Adaptable physics-based super-resolution for electron backscatter difraction maps. Npj Comput. Mater. 8, 255 (2022).

31. Jung, J. et al. Super-resolving material microstructure image via deep learning for microstructure characterization and mechanical behavior analysis. Npj Comput. Mater. 7, 96 (2021).

32. Goodfellow, I. J. et al. Generative Adversarial Nets. in Advances in Neural Information Processing Systems vol. 27 (Curran Associates, Inc., 2014).

33. Kingma, D. P. & Ba, J. Adam: A Method for Stochastic Optimization. Preprint at https://doi.org/10.48550/arXiv.1412.6980 (2017).

34. Usseglio-Viretta, F. L. E. et al. Concavity-based local erosion and sphere-size-based local dilation applied to lithium-ion battery electrode microstructures for particle identification. Comput. Mater. Sci. 251, 113758 (2025).

35. Haralick, R. M., Sternberg, S. R. & Zhuang, X. Image Analysis Using Mathematical Morphology. IEEE Trans. Pattern Anal. Mach. Intell. PAMI-9, 532–550 (1987).

36. Kerschnitzki, M. et al. Architecture of the osteocyte network correlates with bone material quality. J. Bone Miner. Res. 28, 1837–1845 (2013).

37. Lee, T. C., Kashyap, R. L. & Chu, C. N. Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms. CVGIP Graph. Models Image Process. 56, 462–478 (1994).

38. Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. 13, 600–612 (2004).

39. Studholme, C., Hill, D. L. G. & Hawkes, D. J. An overlap invariant entropy measure of 3D medical image alignment. Pattern Recognit. 32, 71–86 (1999).

40. Gelb, L. D. & Gubbins, K. E. Pore Size Distributions in Porous Glasses: A Computer Simulation Study. Langmuir 15, 305–308 (1999).

41. Doyle, M. & Newman, J. The use of mathematical modeling in the design of lithium/polymer battery systems. Electrochimica Acta 40, 2191–2196 (1995).

42. Colclasure, A. M. et al. Electrode scale and electrolyte transport efects on extreme fast charging of lithium-ion cells. Electrochimica Acta 337, 135854 (2020).

43. Usseglio-Viretta, F. L. E. et al. Quantitative Relationships Between Pore Tortuosity, Pore Topology, and Solid Particle Morphology Using a Novel Discrete Particle Size Algorithm. J. Electrochem. Soc. 167, 100513 (2020).

44. Allen, J. M., Weddle, P. J., Smith, K., Usseglio-Viretta, F. L. E. & Finegan, D. P. (Invited) Using a Chemo-Mechanical Model to Investigate Electrolyte Infiltration and Surface Reconstruction within Cracked Cathode Material. ECS Meet. Abstr. MA2024-01, 505 (2024).

45. Usseglio-Viretta, F. L. E. et al. Optimizing Fast Charging and Wetting in Lithium-Ion Batteries with Optimal Microstructure Patterns Identified by Genetic Algorithm. J. Electrochem. Soc. 170, 120506 (2023).