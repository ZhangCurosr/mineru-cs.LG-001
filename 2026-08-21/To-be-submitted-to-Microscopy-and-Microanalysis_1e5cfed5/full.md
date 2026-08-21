# To be submitted to Microscopy and Microanalysis

# Composition-Driven Phase Evolution in Sm-Doped BiFeO₃ via Latent-Field Reconstruction of Atomically Resolved STEM Data

Newsha Javanmardi¹\*, Christopher T. Nelson², Anna N. Morozovska³, Eugene A. Eliseev⁴, Ichiro Takeuchi⁵, and Sergei V. Kalinin¹\*

¹ Department of Materials Science and Engineering, University of Tennessee, Knoxville, Tennessee 37996, USA

² Center for Nanophase Materials Sciences, Oak Ridge National Laboratory, Oak Ridge, Tennessee 37831, USA

³ Institute of Physics, National Academy of Sciences of Ukraine, 46 Nauky Avenue, Kyiv 03028, Ukraine

⁴ Frantsevich Institute for Problems of Materials Science, National Academy of Sciences of Ukraine, 3 Omeliana Pritsaka Street, Kyiv 03142, Ukraine

⁵ Department of Materials Science and Engineering, University of Maryland, College Park, Maryland 20742, USA

\* Corresponding authors: Newsha Javanmardi and Sergei V. Kalinin; sergei2@utk.edu

Functionalities of ferroelectric materials are governed by the spatial organization and coupling of polarization, strain, lattice rotation, and structural order accessible via atomically resolved scanning transmission electron microscopy (STEM) images. Quantitative interpretation of atomicresolution STEM data has conventionally relied on locating atomic columns and converting their fitted coordinates into local structural descriptors. Here, we develop a field-based approach in which atomic-resolution images are represented by spatially varying latent Bragg fields, whose amplitudes and phases provide continuous maps of crystalline order, lattice displacement, strain, rotation, and mode-specific residual structure. The observed atomically resolved images are decoded from the latent fields. We apply this framework to image series of Sm-substituted ${ \mathrm { B i F e O } } _ { 3 }$ spanning 0-20% Sm and crossing the composition-driven boundary between the R3c ferroelectric phase and the orthorhombic, nonpolar Pnma phase. Conventional atom-resolved parameterization is used as an independent validation, showing that reconstructed Bragg amplitude tracks local atomic-column intensity and that field-derived shear reproduces unit-cell angular distortions obtained from atom fitting. The combined analysis reveals a systematic evolution from extended ferroelectric domains at low Sm concentration, through the appearance and growth of localized regions with period-doubled Pnma order at intermediate compositions, to a connected Pnma dominated state at high Sm content. The period-doubled order is accompanied by enhanced shear and lattice rotation and by progressive reorganization of the ferroelectric domain structure. These results establish latent-field reconstruction as a physically interpretable complement to atom finding and provide a unified framework for resolving composition-driven phase evolution in ferroic materials.

## I. Introduction

Ferroelectrics are a materials family in which symmetry breaking is associated with the emergence of spontaneous electric polarization that can be reoriented by an applied electric field.<sup>1</sup> In conventional proper ferroelectrics, polarization is the primary order parameter, but its spatial organization and response cannot be understood independently of the other structural and electrostatic degrees of freedom.<sup>2</sup> Polarization couples to strain through electrostriction and to electric fields through electrostatic energy.<sup>3</sup> It also couples to chemical composition, charged defects, surfaces, interfaces, and, in many perovskites, oxygen-octahedral rotations and tilts.<sup>4</sup> These interactions establish characteristic length scales extending from atomic displacements within a unit cell to domain walls and mesoscale domain patterns.<sup>5</sup> All of these structures contribute to macroscopic electromechanical response.<sup>6</sup> Consequently, ferroelectric behavior is determined not only by the symmetry or lattice parameters of an average structure, but also by the spatial distribution of local order parameters and the compatibility conditions that couple them.

This multiscale character becomes especially pronounced in thin films, heterostructures, compositionally complex materials, and systems near structural phase transition boundaries.<sup>2</sup> Epitaxial strain can stabilize phase states absent in bulk crystals and produce strain-driven morphotropic-like boundaries.<sup>7</sup> Electrostatic and chemical boundary conditions can stabilize conductive domain walls <sup>8</sup> and chemically reconstructed interfaces.<sup>9</sup> Related ferroic heterostructures host flux-closure quadrants,<sup>10</sup> polar vortices,<sup>11</sup> and skyrmions.<sup>12</sup> In these systems, nominally small changes in local lattice distortion may correspond to qualitatively different physical states. Establishing structure-property relationships therefore requires methods capable of identifying not only the average phase but also spatial variations in polarization, strain, rotation, chemical site occupancy, and interfacial structure.

Aberration-corrected STEM provides direct access to projected atomic-column positions and intensities with picometer-scale precision.<sup>13</sup> In perovskite ferroelectrics, displacement of a transition-metal or B-site cation relative to the surrounding A-site sublattice can be used as a local structural proxy for polarization.<sup>13</sup> Measurements of local lattice vectors provide tetragonality, shear, rotation, and strain, whereas column-shape and oxygen-sensitive imaging can make octahedral distortions accessible.<sup>14</sup> Interface-sensitive STEM measurements can relate structural distortions to charge compensation<sup>15</sup> and polarization-controlled screening.<sup>9</sup> These capabilities enabled unit-cell-scale mapping of ferroelectricity and tetragonality,<sup>13</sup> determination of dipole configurations at charged and uncharged walls,<sup>16</sup> mapping of coupled polarization and octahedral tilts,<sup>14</sup> and observation of emergent polar textures.<sup>10</sup>

However, an atomic-resolution STEM image is not itself a direct map of polarization or any other thermodynamic field. It is a two-dimensional projection generated by electron scattering through a specimen of finite thickness and is influenced by probe shape, crystal tilt, scan distortions, drift, dose, and noise.<sup>17</sup>,<sup>18</sup> The structural polarization inferred from projected cation displacements is also distinct from a rigorous three-dimensional polarization calculated using the complete ionic configuration and the corresponding Born effective charges. Moreover, displacement fields of physical interest are often on the order of only a few picometers, comparable to the apparent displacements that can be introduced by scan instabilities or imperfect image registration.<sup>19</sup>,<sup>20</sup> Quantitative interpretation therefore depends as strongly on the image-analysis pathway as on the nominal spatial resolution of the microscope.

The first step in most atomic-resolution analyses is correction of instrumental distortions and establishment of a reliable coordinate system. Revolving or orthogonal scan acquisition can suppress drift-induced distortions.<sup>20</sup>,<sup>21</sup> Non-rigid registration improves frame averaging and removes time-dependent image deformation.<sup>22</sup>,<sup>23</sup> These approaches enable picometer-precision coordinate measurements when combined with appropriate uncertainty controls.<sup>19</sup> Once corrected, the image can be analyzed through pathways that differ in the quantities assumed to be observable, the structural priors imposed, and whether the output is a deterministic estimate or a probability distribution over possible structures.

The most direct pathway is real-space atom finding. In this case, atomic columns are identified as local intensity maxima and their positions are refined using Gaussian, centroid, or template-based fitting.<sup>24</sup>,<sup>25</sup> The resulting coordinates can be indexed by sublattice or unit cell and converted into local lattice vectors, bond lengths, bond angles, cation displacements, strain tensors, and polarization proxies.<sup>13</sup> This representation is physically intuitive and highly effective when atomic columns are well separated, image contrast is strong, and local lattice topology is known. Its limitations become important in low-signal regions, at interfaces and defects, in overlapping projections, and in strongly distorted or multiphase materials, where errors in peak assignment propagate nonlinearly into subsequently calculated fields.

A second pathway operates in reciprocal space. Geometric phase analysis isolates selected Bragg components and uses their spatially varying phases to determine displacement and strain relative to a reference lattice.<sup>26</sup> Related peak-pair methods provide a real-space route to local strain measurement.<sup>27</sup> These approaches are naturally sensitive to periodic order and can yield smoothly varying displacement fields without explicitly locating every atom. However, the result depends on the selected reciprocal-lattice vectors, Fourier masks, reference region, and assumptions concerning phase continuity. Fourier filtering can also suppress nonperiodic information associated with interfaces, defects, chemistry, and local motif changes. Real-space atom fitting and Fourier-phase analysis should therefore be viewed as complementary projections of the image rather than interchangeable measurements.

A third class of methods seeks to infer local structure directly from image neighborhoods or learned representations. Local-crystallography approaches identify phases, symmetries, and defects from atomic neighborhoods.<sup>28</sup> Deep-learning methods can identify chemistry and track local transformations,<sup>29</sup> while statistical representations can extract physically relevant descriptors from atomically resolved images.<sup>30</sup> Bayesian analysis enables uncertainty-aware comparison of physical models for ferroelectric domain walls.<sup>31</sup> Polarization distributions can be learned with or without an explicit atom-finding intermediate.<sup>32</sup> Four-dimensional STEM retains a diffraction pattern at every probe position and extends the available reciprocal-space observables.<sup>33</sup> Electron ptychography provides a related reconstructive route to phase and scattering-potential information.<sup>34</sup>

Despite this progress, most current analysis workflows remain sequential. Images are first corrected, atomic columns are then fitted, atomic coordinates are converted into displacement or polarization vectors, and these vectors are subsequently interpreted using a physical model. Each stage produces a point estimate that is treated as exact by the next stage, even though the underlying quantities may be incompletely observed or mutually coupled. Fourier-based workflows similarly require the reference lattice and relevant Bragg modes to be specified before the physical fields are reconstructed. These choices are frequently reasonable, but they obscure the relationship between assumptions, observability, and uncertainty. They are especially restrictive for systems containing multiple phases, spatially varying strain, diffuse interfaces, weak or missing atomic columns, and coupled polar and nonpolar structural modes.

Here, we propose a field-first reconstruction in which atomic-resolution images are represented by physically interpretable latent Bragg fields and the observed images are decoded from these latent fields. We apply the method to image series of Sm-doped BiFeO₃ spanning the R3c-to-Pnma composition series. Conventional atom coordinates and unit-cell descriptors independently validate the reconstructed amplitude and deformation fields and provide motif-level polar-displacement and Pnma-related quantities not determined by the Bragg fields alone. We combine the two representations to follow the composition-dependent emergence, growth, and connectivity of period-doubled Pnma order across the morphotropic boundary.

## II. Latent Field Analysis

Quantitative analysis of atomic-resolution STEM images is commonly formulated as a sequence of object-detection operations: atomic columns are identified, their coordinates and intensities are refined, atoms are assigned to crystallographic sublattices, and local structural descriptors are calculated from the resulting coordinate set. This representation has enabled unitcell-scale measurements of ferroelectric displacements, lattice distortions, strain, and polarizationrelated order and remains the conventional reference pathway for atomic-resolution ferroelectric microscopy.<sup>13</sup> Here, we adopt a complementary field-based representation in which the image is treated as an observation generated by a small number of spatially modulated crystalline modes.

## II.A. Field-based representations of atomic structures

The fundamental variables of the analysis are the continuous complex fields rather than discrete atomic coordinates. For an image $I ( \mathbf { r } )$ , where $\mathbf { r } = ( x , y )$ denotes position in the image plane, we write the observation model as

$$
\begin{array} { r } { I ( \mathbf { r } ) = B ( \mathbf { r } ) + \sum _ { j = 1 } ^ { N _ { q } } \mathrm { R e } \left[ \eta _ { j } ( \mathbf { r } ) \mathrm { e x p } \big ( i \mathbf { q } _ { j } \cdot \mathbf { r } \big ) \right] + \epsilon ( \mathbf { r } ) . } \end{array}\tag{1}
$$

Here, $B ( \mathbf { r } )$ represents a slowly varying background, $\mathbf { q } _ { j }$ are reference reciprocal-lattice vectors, $\eta _ { j } ( \mathbf { r } )$ are slowly varying complex latent fields, and $\epsilon ( \mathbf { r } )$ describes image noise and structural components not represented by the selected modes. Each latent field is written as

$$
\eta _ { j } ( \mathbf { r } ) = A _ { j } ( \mathbf { r } ) \mathrm { e x p } \big [ i \phi _ { j } ( \mathbf { r } ) \big ] ,\tag{2}
$$

where $A _ { j } ( \mathbf { r } )$ is the local mode amplitude and $\phi _ { j } ( \mathbf { r } )$ is its phase. This form is closely related to geometric-phase and local lock-in descriptions of crystalline images,<sup>26</sup> but is used here as a reconstructive latent representation rather than only as a strain-measurement transform.

The decomposition separates two physically distinct forms of image variation. Changes in $A _ { j }$ reflect changes in the strength and coherence of the corresponding periodic image component and can arise from local atomic-column intensity, structural disorder, thickness, occupancy, defects, or loss of crystalline coherence. Variations in $\phi _ { j } ,$ , by contrast, describe local translations of periodic lattice relative to the reference state. When several non-collinear reciprocal vectors are available, the phase fields jointly define a two-dimensional displacement field and, through its spatial derivatives, local strain, and lattice rotation.

The representation is latent because the complex fields are not measured directly by the detector. They are inferred from the image under the assumption that atomic-resolution contrast can be represented locally by modulated crystalline modes. Unlike a generic learned latent vector, these fields retain an explicit relation to reciprocal-space structure, lattice displacement, and continuum deformation. They therefore provide a physically constrained intermediate representation between the raw image and atom-derived structural descriptors. This complements local-crystallography methods<sup>28</sup> and machine-learning approaches that extract phases, motifs, and transformations from atomic-resolution images.<sup>29</sup>

## II.B. Identification and demodulation of crystalline modes

The images were first restricted to the measured contrast-bearing support, and a missingaware low-pass background was removed before Fourier analysis. Full preprocessing, cropping, masking, and parameter settings are provided in the Supplementary Methods and in the accompanying analysis notebook. The resulting high-pass image is

$$
I _ { \mathrm { h p } } ( \mathbf { r } ) = I ( \mathbf { r } ) - B ( \mathbf { r } ) .\tag{3}
$$

The resulting high-pass image contains the periodic lattice contrast used for Bragg-mode identification and complex demodulation. Candidate reciprocal vectors are identified from local maxima in the magnitude of the Fourier transform,

$$
\tilde { I } ( \mathbf { q } ) = \mathcal { F } \big [ I _ { \mathrm { h p } } ( \mathbf { r } ) \big ] .\tag{4}
$$

The central low-frequency region is excluded, and candidate peaks are ranked by intensity, separation from the origin, angular diversity, and consistency with approximately orthogonal or symmetry-related lattice directions. For the two-dimensional displacement reconstruction used here, at least two non-collinear reciprocal vectors are required.

For each selected reciprocal vector $\mathbf { q } _ { j }$ , the image is shifted to baseband and low-pass filtered:

$$
\eta _ { j } ( \mathbf { r } ) = G _ { \sigma _ { j } } * \big [ I _ { \mathrm { h p } } ( \mathbf { r } ) \mathrm { e x p } \big ( - i \mathbf { q } _ { j } \cdot \mathbf { r } \big ) \big ] ,\tag{5}
$$

where $G _ { \sigma _ { j } }$ is a Gaussian envelope and the asterisk denotes convolution. This operation is equivalent to a local lock-in measurement of the crystalline component associated with $\mathbf { q } _ { j }$ . The filter width controls the compromise between spatial resolution and reciprocal-space selectivity. A narrow real-space kernel retains sharp interfaces but mixes nearby reciprocal components, whereas a broad kernel isolates the Bragg mode more cleanly but smooths domain walls and localized defects.

For image contrasts that exhibit an approximate sign or sublattice ambiguity, the raw phase can be doubled before unwrapping. The doubled phase is unwrapped, divided by two, and corrected for a fitted affine background. Removal of the affine component eliminates the arbitrary global phase, constant translation, and small mismatch between the selected reciprocal vectors and the mean experimental lattice:

$$
\phi _ { j } ^ { \mathrm { r e s } } ( \mathbf { r } ) = \phi _ { j } ( \mathbf { r } ) - \big ( a _ { j } x + b _ { j } y + c _ { j } \big ) .\tag{6}
$$

The residual phase then contains local deviations from the best-fit reference lattice.

## II.C. Displacement, strain, and rotation fields

For a smoothly displaced lattice, the phase of the �th crystalline mode is related to the displacement field �(�) by

$$
\phi _ { j } ^ { \mathrm { r e s } } ( \mathbf { r } ) \approx - \mathbf { q } _ { j } \cdot \mathbf { u } ( \mathbf { r } ) .\tag{7}
$$

Writing the reciprocal vectors as rows of a matrix,

$$
\mathbf { Q } = \left[ \begin{array} { c c } { q _ { 1 x } } & { q _ { 1 y } } \\ { q _ { 2 x } } & { q _ { 2 y } } \\ { \vdots } & { \vdots } \end{array} \right] ,\tag{8}
$$

and the residual phases as

$$
\Phi = \left[ { \phi } _ { 2 } ^ { \mathrm { r e s } } \right] ,\tag{9}
$$

the displacement is obtained through an amplitude-weighted least-squares inversion,

$$
\begin{array} { r } { \mathbf { u } ( \mathbf { r } ) = - [ \mathbf { Q } ^ { \mathrm { T } } \mathbf { W } ( \mathbf { r } ) \mathbf { Q } ] ^ { - 1 } \mathbf { Q } ^ { \mathrm { T } } \mathbf { W } ( \mathbf { r } ) \phi ( \mathbf { r } ) . } \end{array}\tag{10}
$$

The diagonal matrix �(�) weights each phase equation according to the local Bragg amplitude. Modes with weak local amplitude therefore contribute less strongly to the displacement estimate, reducing phase instabilities in defects, interfaces, and low-signal regions.

The displacement field is smoothed using an amplitude-weighted local estimator before differentiation. Its gradient is decomposed into symmetric and antisymmetric parts:

$$
\nabla \mathbf { { u } } = \big [ \partial _ { x } u _ { x } \quad \partial _ { y } u _ { x } \big ] ,\tag{11}
$$

$$
\begin{array} { r } { \pmb { \mathrm { \pmb { \varepsilon } } } = \frac { 1 } { 2 } [ \nabla  { \mathbf { u } } + ( \nabla  { \mathbf { u } } ) ^ { \mathrm { T } } ] , \qquad { \mathbf { \ddot { u } } } = \frac { 1 } { 2 } [ \nabla  { \mathbf { u } } - ( \nabla  { \mathbf { u } } ) ^ { \mathrm { T } } ] . } \end{array}\tag{12}
$$

The corresponding in-plane components are

$$
\begin{array} { r } { \varepsilon _ { x x } = \frac { \partial u _ { x } } { \partial x } , \qquad \varepsilon _ { y y } = \frac { \partial u _ { y } } { \partial y } , } \end{array}\tag{13}
$$

$$
\begin{array} { r } { \varepsilon _ { x y } = \frac { 1 } { 2 } \Big ( \frac { \partial u _ { x } } { \partial y } + \frac { \partial u _ { y } } { \partial x } \Big ) , \qquad \omega = \frac { 1 } { 2 } \Big ( \frac { \partial u _ { y } } { \partial x } - \frac { \partial u _ { x } } { \partial y } \Big ) . } \end{array}\tag{14}
$$

Because differentiation amplifies high-frequency noise, the strain and rotation fields are more sensitive than displacement to scan-line artifacts, phase-unwrapping errors, missing support, and filter selection. Quantitative interpretation is therefore restricted to regions with sufficient Bragg amplitude and adequate distance from crop boundaries and missing-data masks.

## II.D. Residual phases and image reconstruction

The displacement field accounts for phase variation that can be represented as a common translation of the selected lattice modes. The remaining mode-specific phase,

$$
\chi _ { j } ( \mathbf { r } ) = \phi _ { j } ^ { \mathrm { r e s } } ( \mathbf { r } ) + \mathbf { q } _ { j } \cdot \mathbf { u } ( \mathbf { r } ) ,\tag{15}
$$

contains information not captured by a single displacement field. This residual can arise from local changes in the crystallographic basis or motif, phase-slip structures, coexistence of structural variants, errors in the selected reference lattice, or image components outside the assumed observation model.

Because $\chi _ { j }$ is angular, ordinary linear correlations with the wrapped phase are not well defined. It is represented through its circular components,

$$
s _ { j } ( \mathbf { r } ) = \mathrm { s i n } \chi _ { j } ( \mathbf { r } ) , \qquad c _ { j } ( \mathbf { r } ) = \mathrm { c o s } \chi _ { j } ( \mathbf { r } ) ,\tag{16}
$$

which preserve the periodic character of the variable and permit comparison with other local descriptors. The adequacy of the latent representation is evaluated by reconstructing the image:

$$
\begin{array} { r } { I _ { \mathrm { r e c } } ( \mathbf { r } ) = B ( \mathbf { r } ) + \sum _ { j } \mathrm { R e } \big [ \eta _ { j } ( \mathbf { r } ) \mathrm { e x p } \big ( i \mathbf { q } _ { j } \cdot \mathbf { r } \big ) \big ] . } \end{array}\tag{17}
$$

The reconstruction residual,

$$
R ( { \bf r } ) = I ( { \bf r } ) - I _ { \mathrm { r e c } } ( { \bf r } ) ,\tag{18}
$$

provides a measure of model incompleteness. A small residual indicates that the selected modes reproduce the dominant periodic contrast but does not establish that every derived field is physically correct. Conversely, spatially structured residuals can identify defects, nonperiodic motif variations, interfaces, or additional reciprocal modes that require a richer representation.

## II.E. Validation against atom-resolved parameterization

The latent-field reconstruction does not require atom finding for its construction. Conventional atom-resolved parameterization is used both as an independent validation representation and as the source of motif-level quantities that are not uniquely available from the Bragg fields. Bayesian analysis of atomically resolved ferroelectric domain-wall data established the use of unit-cell descriptors as inputs to physical inference.<sup>31</sup> The same Sm-doped $\mathrm { B i F e O } _ { 3 }$ composition series was subsequently analyzed through causal relations among local structural, chemical, and polarization descriptors.<sup>35</sup> Polarization distributions from this system were also learned both with and without an explicit atom-finding stage.<sup>32</sup> Here, atomic-column coordinates and intensities are grouped into local unit cells, from which cation displacement, lattice parameters, unit-cell angle, area, and intensity descriptors are calculated.

The atom-derived unit-cell centers are registered to the cropped image without resizing either representation. Because microscopy files can store coordinates in Cartesian $\left( x , y \right)$ order or array (row, column) order, both conventions are evaluated using image-bound occupancy and agreement between local image intensity and atom-derived intensity. The selected coordinate transformation is applied consistently to cropping, interpolation, visualization, and correlation analysis. Latent fields are sampled at registered unit-cell centers. Cells outside the measured image support, within the missing-data buffer, or close to the crop boundary are excluded. Pearson and Spearman correlations are then calculated between image-derived latent fields and atom-derived descriptors. This comparison tests whether a field reconstruction inferred independently of atomic coordinates recovers physically meaningful structural variations. The most direct expected correspondences are between local Bragg amplitude and atomic-column intensity, and between field-derived shear and atom-derived unit-cell angular distortion.

The validation also defines the limits of the representation. Strong agreement for amplitude and shear does not imply that every motif-level degree of freedom can be inferred from a small set of Bragg modes. Rather, it establishes which physical quantities are observable in the selected field representation and which require additional structural information.

## II.F. Relation between lattice displacement and polarization

The displacement field reconstructed from Bragg phases describes translation and deformation of the selected periodic lattice. It is not generally equivalent to ferroelectric polarization. Structural polarization in a perovskite is associated with relative displacements of different sublattices within the unit cell, whereas the Bragg-phase displacement describes motion of the periodic image motif as a whole. Methods that infer polarization directly from image neighborhoods without an explicit atom-finding stage nevertheless require training labels or an equivalent structural reference.<sup>32</sup> Schematically, u<sub>Bragg</sub> is not equal to P.

The measured atom-derived polar displacement is more appropriately written as

$$
\mathbf { P } _ { \mathrm { a p p } } ( \mathbf { r } ) = \mathbf { P } _ { \mathrm { s t r u c t } } ( \mathbf { r } ) + \mathbf { b } _ { \mathrm { t i l t } } ( \mathbf { r } ) + \mathbf { b } _ { \mathrm { s c a n } } ( \mathbf { r } ) + \boldsymbol { \epsilon } ( \mathbf { r } ) .\tag{19}
$$

Here, $\mathbf { b } _ { \mathrm { t i l t } }$ includes apparent displacement generated by specimen mistilt, thicknessdependent channeling, and dynamical scattering, while $\mathbf { b } _ { s c a \mathbf { n } }$ represents residual instrumental distortion. These common-mode contributions can produce a nonzero apparent polarization even in a weakly polar or nonpolar structure. <sup>17</sup>,<sup>18</sup>,<sup>32</sup>

For this reason, atom-derived polarization is analyzed separately from the Bragg displacement reconstruction. Established unit-cell analyses infer polarization-related order from relative sublattice displacements<sup>13</sup> and probabilistic analyses treat these local descriptors as distinct observables.<sup>31</sup> Polarization-vector populations are modeled in the $( \mathrm { P _ { x } , P _ { y } ) }$ plane, and an inversioncenter or common-mode offset is estimated when symmetry-related domain variants provide sufficient information. The corrected displacement is

$$
\mathbf { P } _ { \mathrm { c o r r } } ( \mathbf { r } ) = \mathbf { P } _ { \mathrm { a p p } } ( \mathbf { r } ) - \mathbf { b } _ { \mathrm { c o m m o n } } ( \mathbf { r } ) .\tag{20}
$$

When the data contains only one domain variant and no independent nonpolar reference, common-mode bias and true polarization are not uniquely identifiable. In such cases, the analysis reports apparent polar displacement and marks the correction as weakly constrained rather than assigning an absolute polarization magnitude.

This separation establishes a hierarchy of observables. Bragg amplitudes describe the strength of periodic order; Bragg phases describe average-lattice displacement and deformation; residual mode structure describes deviations from the common displacement model; and motiflevel relative displacements are required to determine polarization and antipolar order.

## II.G. Opportunities and limitations of the latent-field representation

The latent-field approach does not require every atomic column to be detected and assigned before a displacement field can be constructed. It retains a continuous representation across the image, permits direct reconstruction of the observation, and provides a common framework in which amplitude, displacement, strain, rotation, residual phase, and phase identity can be analyzed jointly. It is also naturally extensible to probabilistic inference, where the latent fields and model parameters are represented by posterior distributions rather than single optimized values.

The representation nevertheless depends on the selected reciprocal vectors, demodulation length scales, reference-lattice gauge, and validity of the slowly varying envelope approximation. It can be affected by scan distortion, specimen bending, local changes in thickness, missing data, and overlap of neighboring Fourier components. A small number of Bragg modes cannot uniquely determine all motif-level degrees of freedom. Polarization, oxygen-octahedral rotations, chemical order, and antipolar displacements can require additional modes or an explicit crystallographicbasis model.

The present formulation should therefore be viewed as complementary to atom-resolved analysis rather than as a universal replacement. Atom finding provides direct local crystallographic measurements when the motif is known and well resolved. Latent-field reconstruction provides a continuous, reconstructive description that is advantageous for extended deformation, phase coexistence, missing regions, and structural patterns whose natural description is field-like rather than atom-discrete. The same conceptual framework can be expanded to 4D-STEM, where reciprocal-space information is retained at every probe position<sup>33</sup> and additional latent fields can be constrained by nanodiffraction or ptychographic observables.<sup>34</sup>

## III. Phase evolution in Sm-doped BiFeO3

Samarium substitution provides a compositionally controlled path from the rhombohedral polar R3c phase of ${ \mathrm { B i F e O } } _ { 3 }$ to an orthorhombic antipolar or non-ferroelectric Pnma phase. The morphotropic boundary occurs near 14% Sm and is accompanied by enhanced dielectric and electromechanical response.<sup>36</sup> Electron diffraction showed short-range antiparallel cation displacements below the boundary, a complex nanoscale phase mixture at the critical composition, and orthorhombic twin structures at Sm concentrations above the boundary.<sup>37</sup> The high-Sm Pnma state is associated with antipolar A-site displacements and an octahedral-tilt superstructure that introduces half-order reflections and a doubled projected periodicity.<sup>37</sup> Near the boundary, additional quarter-order and incommensurate superstructure components reveal a modulated Pnma-related bridging state rather than a spatially uniform direct R3c-to-Pnma transformation.<sup>37</sup> In projected angle-distortion and shear maps, this period-doubled order appears as alternating stripe-like bands. The stripe-like appearance is descriptive only; throughout this paper, the scientific designation is the period-doubled Pnma phase or, for the near-boundary intermediate state, period-doubled Pnma-related order.

Synchrotron diffraction established coexistence of competing structures near the boundary<sup>38</sup> and composition-dependent measurements mapped polarization rotation<sup>39</sup> and structural evolution with temperature and rare-earth substitution.<sup>40</sup> Phenomenological modeling connected the polar-to-antipolar transition to shallow competition between the R3c and Pnma states,<sup>41</sup> while microstructure-electromechanical measurements linked the boundary morphology to enhanced response.<sup>42</sup>

The present dataset contains 14 paired HAADF-STEM and unit-cell-parameterization datasets at nominal Sm concentrations of 0, 7, 10, 13, and 20%.<sup>43</sup> Earlier atom-resolved analysis of this system revealed spatially modulated order-parameter fields near the ferroelectric-antiferroelectric boundary and attributed their stabilization to flexoelectric coupling.<sup>44</sup> The same composition series was later used for causal analysis of competing atomistic mechanisms<sup>35</sup> and for learning polarization distributions with and without atom finding.<sup>32</sup> Here, the conventional descriptors validate the field reconstruction and provide the motif-level quantities used to identify Pnmarelated colonies and ferroelectric walls. The colony score is obtained from directional band-pass structure in the atom-derived angle-distortion field using one threshold set at the pooled 0% Sm 98.5th percentile. Ferroelectric walls are obtained from the mistilt-corrected apparent polardisplacement direction, and their charged or uncharged labels use a geometric bound-charge proxy.

## III.A. Continuous fields in the rhombohedral R3c ferroelectric phase

The 0% Sm-free dataset provides the reference R3c ferroelectric state (Figure 1). The Fourier transform contains a sharp and nearly orthogonal set of lattice reflections, and the selected Bragg modes reconstruct the dominant atomic contrast with R² = 0.951 for the high-pass lattice contrast and 0.955 for the full cropped image (NRMSE = 0.211). The reconstruction residual is concentrated near the support boundary and the localized interior defect, whereas the latent amplitude is comparatively uniform throughout the crystalline region. This establishes that a small set of complex fields captures the periodic lattice contrast without requiring atomic-column localization.

The reconstructed displacement components vary smoothly over the field of view and contain broad regions that coincide qualitatively with atom-derived polar-displacement domains. This correspondence should not be interpreted as equality between Bragg displacement and polarization: the former describes translation of the selected periodic lattice, whereas the latter is obtained from relative sublattice displacement. The atom-derived angle-distortion map contains only weak fluctuations and isolated defect-associated features, and neither the angle field nor the latent shear displays extended Pnma-related period doubling. The derivative fields do contain a diagonal line feature and high-frequency scan texture, demonstrating why local strain and rotation must be interpreted together with the amplitude mask, residual image, and composition controls.

a

0% Sm (Sm\_0\_1): latent-field reconstruction and atom-based validation

e  
![](images/28c3f71e72c4ae98cad516a3152d4af36d43da88c43228cebcd089179a1bd0ca.jpg)

![](images/e7bfb46cd504e323934981d2140f0e7604f1711fff0dfe54f64040ba8a0e300e.jpg)

![](images/3ca4d9b2e687962d4cae1060c42f36884fc85a5612b1a60727472da49ca28424.jpg)

d  
![](images/9116d3748e215300fdb5dccefd64a48701c80a5039c97831677cb15881644e9b.jpg)

![](images/0a3c93ed13ce6c64ed7ce08672aadc5d45132397e12a7243cad99b1b3ed4c131.jpg)

![](images/339e148cfa6a0a59c22a8425f32c000bd1716385d09f842d645b4a9f3f5ba9d4.jpg)

![](images/ccb5e6ec6fb6971d07d58b44b830acf39a36ee00b57432d9bc1ce66e45b3cc75.jpg)

![](images/8e2d9c253896bab921b618ab79fbdc483024da74a737655405a5b34303dd6a50.jpg)

![](images/e06da0978f39e0bf9d5bec0e81f39e9d9a5c8cf347445523b67533228d98f909.jpg)

![](images/c92d4bb8dada354fb5bf2eddd94b8b9083e173f54b7630cfd2fe5c8c576b183b.jpg)

![](images/2b9df19caae2cfc9e754103cccd0af45885c511c7b157e917308d68a8e6fabb4.jpg)

![](images/628621fba0d2780b191738611a7e6c5aeaf513ae402d6e0cb44951aff31bce76.jpg)  
F field analysis A atom-based validation  
Figure 1. Latent-field reconstruction and conventional atom-based quantities for Sm\_0\_1 (0% Sm). F denotes latent-field analysis and A denotes conventional atom-column analysis used for validation; the input and Fourier-selection diagnostic are unbadged. (a) Cropped HAADF-STEM image; only the analyzed crop is shown. (b) Fourier magnitude; circles mark the two primary non-collinear reciprocal vectors and squares mark additional reconstruction modes. (c) Bragg reconstruction and (d) residual. (e, f) Latent Bragg amplitude and the corresponding atomderived mean column intensity. (g, h) Field-derived shear and atom-derived unit-cell angle distortion. All shear maps in Figures 1–3 are displayed within quantitative support on a common −0.035 to +0.035 strain scale; gray indicates unsupported pixels. (i, j) In-plane latent displacement components. (k) Mistilt-corrected apparent polar-displacement direction and (l) lattice rotation. The panel order and real-space orientation are identical in Figures 1 and 2.

## III.B. Continuous fields near the morphotropic phase boundary

A representative 13% Sm dataset is shown in Figure 2. This composition lies immediately below the nominal 14% morphotropic boundary and exhibits coexistence between slowly varying R3c-like lattice deformation and a localized region of period-doubled Pnma-related order. The selected Bragg fields reproduce the dominant image contrast with $\mathrm { R } ^ { 2 } = 0 . 9 5 3$ for the high-pass lattice contrast and 0.974 for the full cropped image $( \mathrm { N R M S E } = 0 . 1 6 1 )$ . The latent amplitude remains continuous across most of the image, while the displacement fields contain broad gradients together with a periodic component in the lower part of the field of view.

The period-doubled Pnma-related region is most clearly identified by alternating positive and negative unit-cell angle distortion. The same spatial region appears independently in fieldderived shear and rotation, demonstrating that the superstructure is encoded in the image phase rather than introduced solely by atom finding. Atom-derived polar-displacement components also reorganize across this region, but their absolute magnitude is not used because a common mistilt or channeling contribution can shift the apparent displacement vector. The central observation is therefore the co-localization of atom-derived angular modulation with independently reconstructed shear and rotation.

## 13% Sm (Sm\_13\_0): latent-field reconstruction and atom-based validation

![](images/f8e153dd4c4856a9a3f9e142fe0122998fb67380cd4353639751b07157fd8d61.jpg)

![](images/43e0ae1d76946988c89331119bcd1f475b554b403f2e7c5b40004d06dae72ea8.jpg)

![](images/21dc7804611adc389a04ce41bd3636d94a9430e28177ece6b90ba29748c75648.jpg)

d  
![](images/6d002024562906f90a17faf86e023f486a8f32343d67cec4f6fd8ee10f2accda.jpg)

![](images/150d6cd60d78fd79cdc4a7d92e085afac230af90dba590057d11f2ebe1ef7acb.jpg)

![](images/3a05ac4e1ba71c5fdb70209e1f95f8777364436ad65c296c8b330d9590ceb78b.jpg)

![](images/222da82584b3908606a7da1010b7930afe518729162c1a0ec3d8c8932608b006.jpg)

![](images/332614d005cbe8dff130bf1efaed0d11edd105e96675dd9b641af94e7b3b7818.jpg)

![](images/0138be1e027f6130c35474c5a33f10655bfa158828ca3abaca8cea2682430c39.jpg)

![](images/2749554593178f9a93e97c0c9adc322dc926e49b6a71b0896d7f8260370ab09c.jpg)

![](images/a8d63166becaa791eb009cf5e96c5711e3c19c4050df20bce71740c97a461b3a.jpg)

![](images/790938144a5fcc185978bebe4c46f44aa6da2ac0b176c5c299ccd50d22da97db.jpg)  
Figure 2. Latent-field reconstruction and conventional atom-based quantities for Sm\_13\_0 (13% Sm). F denotes latent-field analysis and A denotes conventional atom-column analysis used for validation; the input and Fourier-selection diagnostic are unbadged. (a) Cropped HAADF-STEM image; only the analyzed crop is shown. (b) Fourier magnitude; circles mark the two primary non-collinear reciprocal vectors and squares mark additional reconstruction modes. (c) Bragg reconstruction and (d) residual. (e, f) Latent Bragg amplitude and the corresponding atomderived mean column intensity. (g, h) Field-derived shear and atom-derived unit-cell angle distortion. All shear maps in Figures 1–3 are displayed within quantitative support on a common −0.035 to +0.035 strain scale; gray indicates unsupported pixels. (i, j) In-plane latent displacement components. (k) Mistilt-corrected apparent polar-displacement direction and (l) lattice rotation. The panel order and real-space orientation are identical in Figures 1 and 2.

The validation was performed over all 14 datasets rather than only for the representative 13% image. Fisher-weighted repository correlations show that latent Bragg amplitude tracks atomderived mean column intensity (Pearson ${ \bf r } = 0 . 8 4 4 )$ and column-intensity dispersion $( \mathbf { r } = 0 . 7 5 2 )$ . The geometric validation is similarly strong: field-derived $\varepsilon _ { \mathrm { y y } }$ correlates with atom-derived a-axis strain at ${ \bf r } = 0 . 8 5 2 , { \tt \varepsilon } _ { \mathrm { x x } }$ correlates with b-axis strain at $\mathrm { r } = 0 . 7 9 5$ , and field-derived shear correlates with unit-cell angle distortion at $\mathbf { r } = - 0 . 8 3 4$ (95% interval $- 0 . 9 1 3 \mathrm { \ t o \ } - 0 . 6 9 4 )$ . The shear sign follows the stored coordinate convention; its magnitude and consistency show that the phasederived deformation tensor recovers the same local angular distortion measured by conventional unit-cell geometry. Bragg displacement and atom-derived polar displacement show weaker and less consistent component-wise correlations, as expected for observables defined at different structural levels.

## III.C. Composition-dependent evolution of the period-doubled Pnma phase

Figure 3 compares representative images at all five compositions in a fixed-orientation four-row layout. The first row shows the direction of the continuous latent lattice-displacement field, the second row shows latent shear within the quantitative-support mask on one common scale, the third row shows detected period-doubled Pnma-related colonies, and the fourth row shows the relationship of those colonies to ferroelectric walls classified by the geometric boundcharge proxy. The undoped R3c image has only sparse detector responses; the 7 and 10% images contain isolated, replicate-dependent colonies; the 13% image contains a coherent colony; and the 20% image is dominated by one connected colony. Charged and uncharged wall segments both occur near colony boundaries, so the comparison supports local coexistence rather than a universal charged-wall nucleation rule.

Composition evolution: field results and atom-based validation

![](images/6b272e67d7674bf6d43f916305312faf345f0e6b1aa28ea3d9ad143cc0eb190f.jpg)  
Figure 3. Composition-dependent comparison of latent-field results and conventional atombased validation for representative 0, 7, 10, 13, and 20% Sm images. F denotes latent-field analysis and A denotes conventional atom-column analysis used for validation. (a–e) Direction of the continuous Bragg-phase displacement field; this relative lattice-translation field is not polarization. $( \mathrm { f - j } )$ Latent shear within the quantitative-support mask. All five shear maps use the same −0.035 to +0.035 strain scale used in Figures 1 and 2; gray indicates unsupported pixels. (k–o) Detected period-doubled Pnma-related colonies. (p–t) Pnma-related colony mask with the ferroelectric wall skeleton overlaid; green and orange identify walls classified as uncharged and charged, respectively. All maps use stored crop coordinates, so orientation is identical within each column and directly comparable with Figures 1 and 2. The wall charge label is a sign-based structural proxy, not a calibrated bound-charge density.

The composition-level metrics are summarized in Figure 4 using the stored full-resolution arrays. The mean detected Pnma-related area fractions are $0 . 0 1 5 \pm 0 . 0 1 9$ at 0% Sm, $0 . 0 6 7 \pm 0 . 0 2 9$ at $7 \% , 0 . 0 3 2 \pm 0 . 0 4 3$ at $1 0 \% , 0 . 1 8 0 \pm 0 . 0 5 3$ at 13%, and $0 . 6 6 5 \pm 0 . 1 2 3$ at 20%. The small nonzero low-composition values are treated as a detector baseline rather than a physical bulk Pnma fraction. The robust change is the increase between 10 and 13% Sm, followed by formation of an extended Pnma-dominated region at 20% Sm. The ferroelectric wall-density proxy increases from $0 . 0 3 0 5 \pm$ 0.0090 in the undoped material to $0 . 0 6 2 9 ~ \pm ~ 0 . 0 1 6 4$ at 20% Sm, indicating progressive reorganization of the slowly varying polar-displacement field.

Within detected Pnma-related regions, the modulation period is approximately two pseudocubic unit cells at 7, 10, 13, and 20% Sm. The 0% estimate of $2 . 3 3 \pm 0 . 5 8$ cells is produced by sparse detector responses and is not interpreted as a bulk R3c modulation. Composition therefore changes the area and connectivity of the Pnma-related phase much more strongly than its characteristic wavelength. The twofold diffuse-FFT anisotropy A2 is not monotonic and has large replicate scatter at 20% Sm; the complete A2, A4, angular-streak, and spectral-entropy comparison is retained in Figure S4. Global reciprocal-space streaking cannot by itself distinguish the Pnma superstructure from scan-axis artifacts, making the local real-space correspondence between angle distortion and shear the more reliable signature.

![](images/0bd37250600d2c0bd459b61e1c322b494e14306c948677e537f0c259bc47d359.jpg)

![](images/c685c82e344eeed88594e0171e4783b4303ee5ceafd460eac3c925e778d63503.jpg)

![](images/796e9708244a253c82309bda490c8f44f2bb71c61bb8bad764f837b6c4a35431.jpg)

![](images/b40593d559b27afc8e1f9de17915fcdd99c5c41fec2edd2c545902407c34d763.jpg)  
Figure 4. Composition-dependent metrics computed from the stored full-resolution arrays for all 14 images. Open symbols are individual images; filled symbols and error bars are the composition mean and standard deviation between images. (a) Detected period-doubled Pnmarelated area fraction. (b) Ferroelectric wall length per ferroelectric-like unit cell. (c) Modulation period within detected Pnma-related regions. (d) Twofold anisotropy A2 of the Bragg-excluded diffuse Fourier intensity.

## III.D. Nucleation, growth, and coalescence of period-doubled Pnma order

The combined field and metric analysis define a simple morphology sequence. At low Sm concentration, the material is dominated by extended R3c ferroelectric domains and only sparse detector responses to Pnma-related modulation. At 7–10% Sm, localized period-doubled Pnma regions appear, but their fraction is small and varies substantially between images. Near 13% Sm, these regions develop into coherent Pnma-related colonies. By 20% Sm, the detected Pnma-related area fraction is $0 . 6 6 5 \pm 0 . 1 2 3$ and the largest connected component occupies $0 . 6 6 4 \pm 0 . 1 2 5$ of the usable field, showing that nearly all detected Pnma-related area has coalesced into one component. The transition is therefore best described as nucleation, growth, and coalescence of period-doubled Pnma order rather than as a continuous reduction of modulation wavelength.

This evolution is consistent with earlier diffraction and atom-resolved observations of Pnma-related modulated phases near the ferroelectric-antiferroelectric boundary.<sup>37</sup> In the earlier atomic-scale analysis, the near-boundary phase displayed modulated structural and polarization order parameters, and flexoelectric coupling was proposed to reduce the effective wall energy and stabilize the modulated state. <sup>44</sup> The present analysis does not independently determine whether flexoelectricity, elastic compatibility, electrostatic boundary conditions, or chemical heterogeneity supplies the dominant microscopic stabilization. It does show that the Pnma-related superstructure is simultaneously encoded in atom-derived unit-cell geometry and continuous Bragg-phase fields, and that its principal composition-dependent change is growth and connectivity together with an increasing ferroelectric wall-density proxy.

## IV. Summary

Methodologically, we developed a field-first reconstruction of atomically resolved STEM images in which a small set of complex Bragg envelopes generates continuous maps of crystalline amplitude, displacement, strain, rotation, and residual phase. The reconstruction is performed without atomic coordinates, while conventional unit-cell parameterization provides independent validation and the motif-level Pnma and polar descriptors not uniquely available from the Bragg fields. Agreement between latent amplitude and atom-column intensity, and between phasederived shear and unit-cell angular distortion, identifies the physical content of the latent representation and clarifies which motif-level quantities, particularly polarization, still require sublattice-resolved analysis.

For the Sm-doped BiFeO₃ series, the combined latent-field and atom-resolved analysis reveals systematic evolution from an undoped R3c ferroelectric state without coherent period doubling, through sparse and replicate-dependent Pnma-related nuclei at 7–10% Sm, to coherent period-doubled Pnma colonies near 13% Sm and a connected Pnma-dominated state at 20% Sm. The mean detected Pnma-related fraction increases sharply between 10 and 13% Sm and reaches $0 . 6 6 5 \pm 0 . 1 2 3$ at 20% Sm. Once established, the modulation period remains approximately two unit cells, whereas the ferroelectric wall-density proxy increases and the period-doubled angle field becomes strongly co-localized with shear and rotation. The composition dependence is therefore governed primarily by the area and connectivity of Pnma-related order rather than by a continuously changing modulation wavelength.

The practical advantage of the latent representation is most apparent relative to peak-bypeak atom finding and in machine-learning-enabled microscopy. Peak fitting remains indispensable when sublattice identity and absolute motif geometry are required, but it is brittle when columns are missing, overlapping, weak, or locally rearranged. Continuous latent fields use all image pixels, preserve spatial continuity, provide physically interpretable channels for learning and classification, and can be evaluated even where an atomic list is incomplete. They therefore offer a natural intermediate representation for automated microscopy: more directly connected to image formation than a table of fitted coordinates, but substantially more interpretable and transferable than an unconstrained neural-network embedding. Recent diffraction-based work by Latypov and co-workers represents crystalline heterogeneity through spatially resolved latent descriptors. <sup>45</sup> A complementary continuous-field formulation treats diffraction intensity as a differentiable field for reconstructing patterns and localizing boundaries and heterogeneity. 46

## V. Acknowledgements

This material (NJ, IT, SVK) is based upon work supported by the National Science Foundation under Award No. NSF 2523284.

The work of A.N.M. and E.A.E. is primarily supported by the DOE Software Project on “Computational Mesoscale Science and Open Software for Quantum Materials,” under Award Number DE-SC0020145 as part of the Computational Materials Sciences Program of US Department of Energy, Office of Science, Basic Energy Sciences, and also partially supported by the National Academy of Sciences of Ukraine.

9. Y. M. Kim, A. Morozovska, E. Eliseev, M. P. Oxley, R. Mishra, S. M. Selbach, T. Grande, S. T.

## References

1. M. E. Lines and A. M. Glass, Principles and Applications of Ferroelectrics and Related Materials. (Clarendon Press, Oxford, 1977).

2. M. Dawber, K. M. Rabe and J. F. Scott, Reviews of Modern Physics 77 (4), 1083-1130 (2005).

3. L.-Q. Chen, Journal of the American Ceramic Society 91 (6), 1835-1844 (2008).

4. R. Ramesh and N. A. Spaldin, Nature Materials 6, 21-29 (2007).

5. G. Catalan, J. Seidel, R. Ramesh and J. F. Scott, Reviews of Modern Physics 84 (1), 119-156 (2012).

6. L. W. Martin and A. M. Rappe, Nature Reviews Materials 2, 16087 (2017).

7. R. J. Zeches, M. D. Rossell, J. X. Zhang, A. J. Hatt, Q. He, C. H. Yang, A. Kumar, C. H. Wang, A.

Melville, C. Adamo, G. Sheng, Y. H. Chu, J. F. Ihlefeld, R. Erni, C. Ederer, V. Gopalan, L. Q. Chen, D. G. Schlom, N. A. Spaldin, L. W. Martin and R. Ramesh, Science 326 (5955), 977-980 (2009).

8. J. Seidel, L. W. Martin, Q. He, Q. Zhan, Y. H. Chu, A. Rother, M. E. Hawkridge, P. Maksymovych, P. Yu, M. Gajek, N. Balke, S. V. Kalinin, S. Gemming, F. Wang, G. Catalan, J. F. Scott, N. A. Spaldin, J. Orenstein and R. Ramesh, Nat Mater 8 (3), 229-234 (2009).

10. Y. L. Tang, Y. L. Zhu, X. L. Ma, A. Y. Borisevich, A. N. Morozovska, E. A. Eliseev, W. Y. Wang, Y. J. Wang, Y. B. Xu, Z. D. Zhang and S. J. Pennycook, Science 348 (6234), 547-551 (2015).

Shafer, E. Arenholz, L. R. Dedon, D. Chen, A. Vishwanath, A. M. Minor, L. Q. Chen, J. F. Scott, L. W. Martin and R. Ramesh, Nature 530 (7589), 198-201 (2016).

12. S. Das, Y. L. Tang, Z. Hong, M. A. P. Goncalves, M. R. McCarter, C. Klewe, K. X. Nguyen, F. Gomez-Ortiz, P. Shafer, E. Arenholz, V. A. Stoica, S. L. Hsu, B. Wang, C. Ophus, J. F. Liu, C. T. Nelson, S. Saremi, B.

Prasad, A. B. Mei, D. G. Schlom, J. Iniguez, P. Garcia-Fernandez, D. A. Muller, L. Q. Chen, J. Junquera, L. W. Martin and R. Ramesh, Nature 568 (7752), 368-372 (2019).

13. C. L. Jia, V. Nagarajan, J. Q. He, L. Houben, T. Zhao, R. Ramesh, K. Urban and R. Waser, Nat Mater 6 (1), 64-69 (2007).

15. M. F. Chisholm, W. Luo, M. P. Oxley, S. T. Pantelides and H. N. Lee, Phys Rev Lett 105 (19), 197602 (2010).

16. C. L. Jia, S. B. Mi, K. Urban, I. Vrejoiu, M. Alexe and D. Hesse, Nat Mater 7 (1), 57-61 (2008).

17. J. Cui, Y. Yao, Y. G. Wang, X. Shen and R. C. Yu, Ultramicroscopy 182, 156-162 (2017).

18. P. Gao, A. Kumamoto, R. Ishikawa, N. Lugg, N. Shibata and Y. Ikuhara, Ultramicroscopy 184, 177-187 (2018).

19. A. B. Yankovich, B. Berkels, W. Dahmen, P. Binev, S. I. Sanchez, S. A. Bradley, A. Li, I. Szlufarska and P. M. Voyles, Nat Commun 5, 4155 (2014).

20. C. Ophus, J. Ciston and C. T. Nelson, Ultramicroscopy 162, 1-9 (2016).

21. X. Sang and J. M. LeBeau, Ultramicroscopy 138, 28-35 (2014).

22. B. Berkels, P. Binev, D. A. Blom, W. Dahmen, R. C. Sharpley and T. Vogt, Ultramicroscopy 138, 46-56 (2014).

23. L. Jones, H. Yang, T. J. Pennycook, M. S. J. Marshall, S. Van Aert, N. D. Browning, M. R. Castell and P. D. Nellist, Advanced Structural and Chemical Imaging 1, 8 (2015).

24. X. Sang, A. A. Oni and J. M. LeBeau, Microscopy and Microanalysis 20 (6), 1764-1771 (2014).

25. M. Nord, P. E. Vullum, I. MacLaren, T. Tybell and R. Holmestad, Advanced Structural and Chemical Imaging 3, 9 (2017).

26. M. J. Hÿtch, E. Snoeck and R. Kilaas, Ultramicroscopy 74 (3), 131-146 (1998).

27. P. L. Galindo, S. Kret, A. M. Sanchez, J.-Y. Laval, A. Yáñez, J. Pizarro, E. Guerrero, T. Ben and S. I. Molina, Ultramicroscopy 107 (12), 1186-1193 (2007).

28. A. Belianinov, Q. He, M. Kravchenko, S. Jesse, A. Borisevich and S. V. Kalinin, Nature Communications 6, 7801 (2015).

29. M. Ziatdinov, O. Dyck, A. Maksov, X. Li, X. Sang, K. Xiao, R. R. Unocic, R. Vasudevan, S. Jesse and S. V. Kalinin, ACS Nano 11 (12), 12742-12752 (2017).

30. L. Vlcek, A. Maksov, M. Pan, R. K. Vasudevan and S. V. Kalinin, ACS Nano 11 (10), 10313-10320 (2017).

31. C. T. Nelson, R. K. Vasudevan, X. Zhang, M. Ziatdinov, E. A. Eliseev, I. Takeuchi, A. N. Morozovska and S. V. Kalinin, Nature Communications 11, 6361 (2020).

32. C. T. Nelson, A. Ghosh, M. P. Oxley, X. Zhang, M. Ziatdinov, I. Takeuchi and S. V. Kalinin, npj Computational Materials 7, 149 (2021).

33. C. Ophus, Microsc Microanal 25 (3), 563-582 (2019).

34. Y. Jiang, Z. Chen, Y. Han, P. Deb, H. Gao, S. Xie, P. Purohit, M. W. Tate, J. Park, S. M. Gruner, V. Elser and D. A. Muller, Nature 559 (7714), 343-349 (2018).

35. M. Ziatdinov, C. T. Nelson, X. Zhang, R. K. Vasudevan, E. A. Eliseev, A. N. Morozovska, I. Takeuchi and S. V. Kalinin, npj Computational Materials 6 (1), 127 (2020).

36. S. Fujino, M. Murakami, V. Anbusathaiah, S.-H. Lim, V. Nagarajan, C. J. Fennie, M. Wuttig, L. G.

Salamanca-Riba and I. Takeuchi, Applied Physics Letters 92 (20), 202904 (2008).

37. C.-J. Cheng, D. Kan, S.-H. Lim, W. R. McKenzie, P. R. Munroe, L. G. Salamanca-Riba, R. L. Withers, I. Takeuchi and V. Nagarajan, Physical Review B 80 (1), 014109 (2009).

38. S. B. Emery, C.-J. Cheng, D. Kan, F. J. Rueckert, S. P. Alpay, V. Nagarajan, I. Takeuchi and B. O. Wells, Applied Physics Letters 97 (15), 152902 (2010).

39. D. Kan, V. Anbusathaiah and I. Takeuchi, Advanced Materials 23 (15), 1765-1769 (2011).

40. D. Kan, C.-J. Cheng, V. Nagarajan and I. Takeuchi, Journal of Applied Physics 110 (1), 014106 (2011).

41. F. Xue, L. Liang, Y. Gu, I. Takeuchi, S. V. Kalinin and L.-Q. Chen, Applied Physics Letters 106 (1), 012903 (2015).

42. C.-J. Cheng, D. Kan, V. Anbusathaiah, I. Takeuchi and V. Nagarajan, Applied Physics Letters 97 (21), 212905 (2010).

43. A. Ghosh, C. Nelson, M. Ziatdinov and S. V. Kalinin, Zenodo (2021).

44. A. Y. Borisevich, E. A. Eliseev, A. N. Morozovska, C. J. Cheng, J. Y. Lin, Y. H. Chu, D. Kan, I. Takeuchi, V. Nagarajan and S. V. Kalinin, Nat Commun 3, 775 (2012).

45. J. Wang, M. Calvat, J. C. Stinville and M. I. Latypov, arXiv:2606.09611 [cond-mat.mtrl-sci] (2026).

46. I.-T. Huang and M. I. Latypov, arXiv:2606.10352 [cond-mat.mtrl-sci] (2026).