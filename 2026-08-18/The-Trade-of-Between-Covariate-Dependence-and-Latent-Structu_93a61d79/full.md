# The Trade-of Between Covariate Dependence and Latent Structure in Representation Learning

Mał<sub>g</sub>orzata Łaz<sub>ę</sub>cka <sub>m.</sub>l<sub>azec</sub>k<sub>a@uw.e</sub>d<sub>u.p</sub>l Fac<sub>u</sub>lt<sub>y</sub> of Mathematics<sub>,</sub> Informatics<sub>,</sub> and Mechanics<sub>,</sub> Uni<sub>v</sub>ersit<sub>y</sub> of Warsa<sub>w</sub> W<sub>arsaw,</sub> P<sub>o</sub>l<sub>an</sub>d

## Abstract

Disentan<sub>g</sub>led re<sub>p</sub>resentation learnin<sub>g</sub> seeks latent re<sub>p</sub>resentations whose indicid<sub>u</sub>al dimensions each ali<sub>g</sub>n with a distinct covariate. Unsu<sub>p</sub>ervised a<sub>pp</sub>roaches t<sub>yp</sub>icall<sub>y</sub> tar<sub>g</sub>et latent dimension inde-<sub>p</sub>endence<sub>, y</sub>et this <sub>g</sub>ives no <sub>g</sub>uarantee that the resultin<sub>g</sub> dimensions ali<sub>g</sub>n with semanticall<sub>y</sub> meanin<sub>g</sub>ful covariates. Su<sub>p</sub>ervised a<sub>pp</sub>roaches structure the latent s<sub>p</sub>ace usin<sub>g</sub> observed covariates<sub>,</sub> b<sub>u</sub>t <sub>un</sub>d<sub>er corre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> th<sub>ey canno</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y con</sub>t<sub>ro</sub>l <sub>one-</sub>t<sub>o-one</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e a</sub>li<sub>gnmen</sub>t <sub>an</sub>d l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub> W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>a</sub> <sub>un</sub>ifi<sub>e</sub>d<sub>,</sub> <sub>superv</sub>i<sub>se</sub>d f<sub>ramewor</sub>k th<sub>a</sub>t <sub>coup</sub>l<sub>es</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence w</sub>ith <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> Withi<sub>n</sub> thi<sub>s</sub> f<sub>ramewor</sub>k<sub>, we s</sub>h<sub>ow an</sub> i<sub>n</sub>h<sub>eren</sub>t t<sub>ra</sub>d<sub>e-o</sub>f<sub>,</sub> <sub>w</sub>h<sub>ere</sub> <sub>en</sub>f<sub>orc</sub>i<sub>ng</sub> l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>or</sub> <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub> <sub>one-</sub>t<sub>o-one</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence comes a</sub>t <sub>a prova</sub>bl<sub>e cos</sub>t i<sub>n</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> ali<sub>g</sub>nment. We <sub>p</sub>rove that the resultin<sub>g</sub> disentan<sub>g</sub>lement re<sub>g</sub>imes <sub>are or</sub>d<sub>ere</sub>d b<sub>y</sub> th<sub>e s</sub>t<sub>reng</sub>th <sub>o</sub>f th<sub>a</sub>t <sub>a</sub>li<sub>gnmen</sub>t<sub>.</sub> E<sub>ac</sub>h <sub>reg</sub>i<sub>me a</sub>d<sub>-</sub> <sub>m</sub>it<sub>s</sub> <sub>a</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space.</sub> W<sub>e</sub> <sub>app</sub>l<sub>y</sub> th<sub>ese</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons pos</sub>t<sub>-</sub>h<sub>oc</sub> t<sub>o rea</sub>li<sub>gn</sub> th<sub>e represen</sub>t<sub>a</sub>ti<sub>ons o</sub>f <sub>p</sub>r<sub>e</sub>tr<sub>a</sub>in<sub>e</sub>d m<sub>o</sub>d<sub>e</sub>l<sub>s suc</sub>h <sub>as</sub> CLIP<sub>,</sub> DINO<sub>v</sub>2<sub>, a</sub>nd ViT<sub>, a</sub>nd <sub>we</sub> f<sub>o</sub>ld them into the inference of informed factor anal<sub>y</sub>sis (iFA), a <sub>p</sub>robabili<sub>s</sub>ti<sub>c mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>covar</sub>i<sub>a</sub>t<sub>e-</sub>i<sub>n</sub>f<sub>orme</sub>d f<sub>ac</sub>t<sub>ors.</sub> O<sub>n s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>an</sub>d <sub>rea</sub>l <sub>mu</sub>lti<sub>-om</sub>i<sub>cs</sub> d<sub>a</sub>t<sub>a, we s</sub>h<sub>ow</sub> th<sub>a</sub>t b<sub>o</sub>th <sub>os</sub>t<sub>-</sub>h<sub>oc a</sub>li <sub>nmen</sub>t <sub>an</sub>d iFA <sub>ena</sub>bl<sub>e con</sub>t<sub>ro</sub>ll<sub>a</sub>bilit<sub>y o</sub>f <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>ons.</sub>

## CCS Concepts

• Computing methodologies → Learning latent representations; Supervised learning; Latent variable models; Factor analysis.

## Keywords

disentan<sub>g</sub>lement<sub>,</sub> su<sub>p</sub>ervised latent re<sub>p</sub>resentation learnin<sub>g,</sub> factoranal<sub>y</sub>sis

## 1 Introduction

E<sub>wa</sub> Sz<sub>c</sub>z<sub>u</sub>r<sub>e</sub>k <sub>em.szczure</sub>k<sub>@uw.e</sub>d<sub>u.p</sub>l I<sub>ns</sub>tit<sub>u</sub>t<sub>e o</sub>f AI f<sub>or</sub> H<sub>ea</sub>lth<sub>,</sub> H<sub>e</sub>l<sub>m</sub>h<sub>o</sub>lt<sub>z</sub> M<sub>un</sub>i<sub>c</sub>h M<sub>u</sub>nich<sub>,</sub> German<sub>y</sub> Fac<sub>u</sub>lt<sub>y</sub> of Mathematics<sub>,</sub> Informatics<sub>,</sub> and Mechanics<sub>,</sub> Uni<sub>v</sub>ersit<sub>y</sub> of Warsa<sub>w</sub> W<sub>arsaw,</sub> P<sub>o</sub>l<sub>an</sub>d

Latent re<sub>p</sub>resentation learnin<sub>g</sub> ma<sub>p</sub>s a hi<sub>g</sub>h-dimensional data s<sub>p</sub>ace t<sub>o a</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>one, enco</sub>di<sub>ng eac</sub>h <sub>o</sub>b<sub>serva</sub>ti<sub>on as a compac</sub>t latent representation capturing its essentia<sup>l</sup> c<sup>h</sup>aracteristics. A <sup>l</sup>ongstan<sup>d</sup>ing o<sup>b</sup>jective o<sup>f</sup> <sup>l</sup>atent representation <sup>l</sup>earning is to o<sup>b</sup>tain re<sub>p</sub>resentations <sub>w</sub>hose indi<sub>v</sub>id<sub>u</sub>al dimensions ali<sub>g</sub>n <sub>w</sub>ith distinct<sub>,</sub> semantica<sup>ll</sup>y meaning<sup>f</sup>u<sup>l</sup> <sup>f</sup>actors o<sup>f</sup> variation [6]. Suc<sup>h</sup> disentangled representations promise interpretabi<sup>l</sup>ity, contro<sup>ll</sup>ab<sup>l</sup>e generation and im<sub>p</sub>roved <sub>g</sub>eneralization<sub>,</sub> because the<sub>y</sub> seek to ensure that mani<sub>p</sub>ulatin<sub>g</sub> a sin<sub>g</sub>le latent dimension chan<sub>g</sub>es onl<sub>y</sub> one underl<sub>y</sub>in<sub>g</sub> factor of variation [54]. In man<sub>y</sub> real-world a<sub>pp</sub>lications, the factors of interest are not <sup>l</sup>atent abstractions, but observed covariates suc<sup>h</sup> as <sub>c</sub>li<sub>n</sub>i<sub>ca</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>es,</sub> <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>con</sub>diti<sub>ons,</sub> <sub>or</sub> <sub>ca</sub>t<sub>egory</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s,</sub> <sub>an</sub>d the <sub>p</sub>ractical aim is a re<sub>p</sub>resentation in which each latent dimension is ali<sub>g</sub>ned one-to-one with its corres<sub>p</sub>ondin<sub>g</sub> covariate [23].

![](images/112eb6e8b859d9d5f3458ac661ec3f3afac8383286ae999c38267d6b4b842115.jpg)  
Figure 1: Overview of the proposed framework. Correlated covariates (Σ<sub>�</sub>) shape the latent structure (Σ<sub>�</sub>). The transformations impose diferent constraints on the latent space, trading of structural constraints (exclusive latent-covariate dependence, latent inter-independence) against the strength of pairwise latent-covariate dependence $( \Sigma _ { Y , \tilde { Z } } )$

Thi<sub>s</sub> <sub>assump</sub>ti<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> <sub>are</sub> i<sub>n</sub>d<sub>e-</sub> <sub>pen</sub>d<sub>en</sub>t<sub>,</sub> b<sub>u</sub>t f<sub>a</sub>il<sub>s</sub> i<sub>n</sub> th<sub>e</sub> <sub>prac</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>re</sub>l<sub>evan</sub>t <sub>reg</sub>i<sub>me</sub> <sub>o</sub>f <sub>corre</sub>l<sub>a</sub>t<sub>e</sub>d covariates [16, 52], as exem<sub>p</sub>lified b<sub>y</sub> Σ<sub>�</sub> in Fi<sub>g</sub>. 1. This correlation among covariates creates a tension <sup>b</sup>etween two o<sup>b</sup>jectives: ma<sup>k</sup>- in<sub>g</sub> each latent dimension stron<sub>g</sub>l<sub>y</sub> ali<sub>g</sub>ned with its covariate<sub>,</sub> and im<sub>p</sub>osin<sub>g</sub> structural constraints on the latent s<sub>p</sub>ace<sub>,</sub> such as inde<sub>p</sub>endence amon<sub>g</sub> the latent dimensions or exclusive de<sub>p</sub>endence (each dimension aligning only with its own covariate). This trade-of between latent-covariate dependence and structural constraints on the latent space has not been systematically studied.

The <sub>p</sub>rinci<sub>p</sub>le of latent inde<sub>p</sub>endence is common in full<sub>y</sub> unsu<sub>p</sub>ervised a<sub>pp</sub>roaches<sub>,</sub> and the wa<sub>y</sub> it is a<sub>pp</sub>lied varies: classical linear methods such as PCA [25, 42] and ICA [27] assume and recover i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>sources o</sub>f <sub>var</sub>i<sub>a</sub>ti<sub>on, w</sub>hil<sub>e mo</sub>d<sub>ern non</sub>li<sub>near mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>encourage</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>e.g.</sub> th<sub>roug</sub>h f<sub>ac</sub>t<sub>or</sub>i<sub>ze</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>represen</sub>t<sub>a-</sub> tions [12, 23, 32]. Enforcin<sub>g</sub> inde<sub>p</sub>endence alone, however, does not in <sub>g</sub>eneral ensure that latent dimensions ali<sub>g</sub>n with inter<sub>p</sub>retable or meanin<sub>g</sub>ful covariates, nor does it <sub>g</sub>uarantee identifiabilit<sub>y</sub> [38]. C<sub>on</sub>diti<sub>on</sub>i<sub>ng</sub> l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e mo</sub>d<sub>e</sub>l<sub>s on o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es un</sub>d<sub>er</sub> additional assum<sub>p</sub>tions restores identifiabilit<sub>y</sub> u<sub>p</sub> to sim<sub>p</sub>le transformations [31], and a broad famil<sub>y</sub> of nonlinear [22, 28, 34, 50, 56] and linear [2, 4, 5, 11, 21, 41, 48, 58] su<sub>p</sub>ervised re<sub>p</sub>resentation learni<sub>ng me</sub>th<sub>o</sub>d<sub>s uses o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> t<sub>o s</sub>t<sub>ruc</sub>t<sub>ure</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space.</sub> Yet all existing approaches leave open the problem of the simultaneous control of the one-to-one alignment of each latent dimension with its covariate and of the structural constraints on the latent space, particularly in the general, correlatedcovariate setting.

W<sub>e c</sub>l<sub>ose</sub> thi<sub>s gap w</sub>ith <sub>a superv</sub>i<sub>se</sub>d f<sub>ramewor</sub>k th<sub>a</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er-</sub> i<sub>zes</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> and latent structural constraints<sub>,</sub> <sub>p</sub>rovidin<sub>g</sub> a <sub>p</sub>rinci<sub>p</sub>led a<sub>pp</sub>roach to controllin<sub>g</sub> latent re<sub>p</sub>resentations. Our contributions are both theoretical and <sub>p</sub>ractical. First, (i) levera<sub>g</sub>in<sub>g</sub> the non-identifiabilit<sub>y</sub> of latent re<sub>p</sub>resentations<sub>,</sub> we formulate o<sub>p</sub>timization <sub>p</sub>roblems over invertible latent space transformations (� in Fig. 1) whose sol<sub>u</sub>ti<sub>ons y</sub>i<sub>e</sub>ld l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence across</sub> dif<sub>eren</sub>t di<sub>sen-</sub> tan<sub>g</sub>lement re<sub>g</sub>imes: inde<sub>p</sub>endent latent dimensions (� ), tunable intermediate latent inter-de<sub>p</sub>endencies (�<sub>�</sub>), unconstrained latent inter-de<sub>p</sub>endencies (�<sub>1</sub>), and a variant with exclusive one-to-one l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> $( T _ { \mathrm { e x } } )$ . Second, (ii) we <sub>p</sub>rove that these regimes are strictly ordered by latent-covariate alignment strength, and that there exists a fundamental trade-of between structural latent constraints and latent-covariate dependence. Finall<sub>y</sub>, (iii) we demonstrate that the <sub>p</sub>ro<sub>p</sub>osed transformations can be applied post hoc to re-align pretrained representations or integrated into a probabilistic graphical model, informed Factor Analysis (iFA). Through experiments on simulated covariate-<sub>covar</sub>i<sub>ance scenar</sub>i<sub>os an</sub>d <sub>rea</sub>l <sub>mu</sub>lti<sub>-om</sub>i<sub>cs</sub> d<sub>a</sub>t<sub>a, we s</sub>h<sub>ow</sub> th<sub>a</sub>t b<sub>o</sub>th <sub>approac</sub>h<sub>es ena</sub>bl<sub>e con</sub>t<sub>ro</sub>ll<sub>a</sub>bilit<sub>y o</sub>f <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>ons.</sub>

## 2 Related works

Independence-based disentanglement methods. Many disentang<sup>lement</sup> <sup>a</sup>pp<sup>roaches</sup> <sup>assume</sup> <sup>or</sup> <sup>encoura</sup>g<sup>e</sup> <sup>inde</sup>p<sup>endence</sup> <sup>amon</sup>g th<sub>e</sub> di<sub>mens</sub>i<sub>ons o</sub>f th<sub>e represen</sub>t<sub>a</sub>ti<sub>on, an</sub> id<sub>ea roo</sub>t<sub>e</sub>d i<sub>n c</sub>l<sub>ass</sub>i<sub>ca</sub>l latent variable methods such as <sub>p</sub>rinci<sub>p</sub>al com<sub>p</sub>onent anal<sub>y</sub>sis (PCA) [25, 42], inde<sub>p</sub>endent com<sub>p</sub>onent anal<sub>y</sub>sis (ICA) [27], factor anal-<sub>y</sub>sis (FA) [51], and extended to <sub>p</sub>robabilistic latent variable mod els, includin<sub>g</sub> multi-view factor models, e.<sub>g</sub>. MOFA [3] and others [33, 44, 57]. The most <sub>p</sub>rominent dee<sub>p</sub> <sub>g</sub>enerative exam<sub>p</sub>les are

VAE<sub>-</sub>b<sub>ase</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s regu</sub>l<sub>ar</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> t<sub>owar</sub>d inde<sub>p</sub>endence, e.<sub>g</sub>. �-VAE [23], FactorVAE [32], or �-TCVAE [12]. Si<sub>m</sub>il<sub>ar</sub> id<sub>eas</sub> h<sub>ave a</sub>l<sub>so</sub> b<sub>een exp</sub>l<sub>ore</sub>d b<sub>eyon</sub>d VAE<sub>s,</sub> f<sub>or examp</sub>l<sub>e</sub> throu<sub>g</sub>h conce<sub>p</sub>t whitenin<sub>g</sub> in convolutional neural networks [14], InfoGAN [13], a <sub>g</sub>enerative GAN model which <sub>p</sub>romotes inde<sub>p</sub>endence usin<sub>g</sub> mutual information, or non-linear ICA extension [9].

Supervision in disentanglement and latent representation models. A<sub>ssum</sub>i<sub>ng</sub> l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence a</sub>l<sub>one</sub> d<sub>oes no</sub>t <sub>guaran</sub>t<sub>ee</sub> id<sub>en</sub>tifi<sub>a-</sub> bilit<sub>y</sub> [38]. Conse<sub>q</sub>uentl<sub>y</sub>, man<sub>y</sub> successful a<sub>pp</sub>roaches incor<sub>p</sub>orate weak-su<sub>p</sub>ervision, such as <sub>g</sub>rou<sub>p</sub>ed observations [8], tem<sub>p</sub>oral structure [26], or <sub>p</sub>aired observations in which some factors of variation are shared [39]. A second line is su<sub>p</sub>ervised in the stron<sub>g</sub>er sense th<sub>a</sub>t <sub>covar</sub>i<sub>a</sub>t<sub>es are</sub> di<sub>rec</sub>tl<sub>y o</sub>b<sub>serve</sub>d<sub>, e.g. use</sub>d t<sub>o con</sub>diti<sub>on</sub> th<sub>e</sub> latent re<sub>p</sub>resentation, as in CVAE [50], iVAE [31], or SE-VAE [56], amon<sub>g</sub> others [22, 28, 34]. The Identifiable VAE (iVAE) [31] tar-<sub>g</sub>ets identifiabilit<sub>y</sub> b<sub>y</sub> conditionin<sub>g</sub> a factorized ex<sub>p</sub>onential-famil<sub>y</sub> <sub>pr</sub>i<sub>or</sub> <sub>on</sub> th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>up</sub> t<sub>o</sub> <sub>a</sub> li<sub>near</sub> <sub>an</sub>d <sub>componen</sub>t<sub>-w</sub>i<sub>se</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>can</sub> b<sub>e</sub> <sub>un</sub>d<sub>ers</sub>t<sub>oo</sub>d <sub>as a</sub> f<sub>orm o</sub>f di<sub>sen</sub>t<sub>ang</sub>l<sub>emen</sub>t<sub>.</sub>

Supervised linear latent variable models. T<sup>h</sup>e use o<sup>f</sup> covariates to <sub>gu</sub>id<sub>e represen</sub>t<sub>a</sub>ti<sub>ons</sub> h<sub>as</sub> b<sub>een a</sub>l<sub>so use</sub>d i<sub>n</sub> li<sub>near mo</sub>d<sub>e</sub>l<sub>s.</sub> E<sub>ar</sub>l<sub>y</sub> exam<sub>p</sub>les include su<sub>p</sub>ervised extensions of PCA (SPCA), such as b<sub>y</sub> Bair et al. [4], which focus onl<sub>y</sub> on features that are individuall<sub>y</sub> associated with the covariates, and Barshan et al. [5], which estimates a se<sub>q</sub>uence of <sub>p</sub>rinci<sub>p</sub>al com<sub>p</sub>onents that have maximal de<sub>p</sub>endence <sub>w</sub>ith <sub>covar</sub>i<sub>a</sub>t<sub>es.</sub> Oth<sub>er re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e regress</sub>i<sub>on- an</sub>d correlation-based a<sub>pp</sub>roaches, such as <sub>p</sub>artial least s<sub>q</sub>uares (PLS) [55] and canonical correlation anal<sub>y</sub>sis (CCA), as well as su<sub>p</sub>ervised variants of FA [2, 21, 41, 48, 58]. These methods levera<sub>g</sub>e covariates to im<sub>p</sub>rove <sub>p</sub>rediction or inter<sub>p</sub>retabilit<sub>y,</sub> but <sub>g</sub>enerall<sub>y</sub> do not <sub>en</sub>f<sub>orce a one-</sub>t<sub>o-one</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e a</sub>li<sub>gnmen</sub>t<sub>.</sub> T<sub>o</sub> th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>f our knowled<sub>g</sub>e, the onl<sub>y</sub> exce<sub>p</sub>tion is SOFA [11], which ex<sub>p</sub>licitl<sub>y</sub> associates indi<sub>v</sub>id<sub>u</sub>al latent dimensions <sub>w</sub>ith s<sub>p</sub>ecific co<sub>v</sub>ariates.

Disentanglement under correlated covariates. Disentang<sup>l</sup>ing t<sup>h</sup>e l<sub>a</sub>t<sub>en</sub>t <sub>space un</sub>d<sub>er corre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> i<sub>s</sub> i<sub>mpor</sub>t<sub>an</sub>t b<sub>u</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> underex<sub>p</sub>lored [52]. Em<sub>p</sub>irical studies that introduce correlated covariates [16, 52] show that the roblem difers fundamentall from the usual assum<sub>p</sub>tion of inde<sub>p</sub>endent covariates. Enforcin<sub>g</sub> inde-<sub>pen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>ons un</sub>d<sub>er corre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> de<sub>g</sub>rades the model b<sub>y</sub> e.<sub>g</sub>. <sub>p</sub>reventin<sub>g</sub> it from attainin<sub>g</sub> the o<sub>p</sub>timal likelihood [52] or discardin information relevant to the covariates and thus harmin<sub>g p</sub>redictive <sub>p</sub>erformance [19]. Exact inde<sub>p</sub>endence is t<sup>h</sup>ere<sup>f</sup>ore t<sup>h</sup>e wrong o<sup>b</sup>jective, motivating re<sup>l</sup>axations. One <sup>l</sup>ine of work re<sub>q</sub>uires onl<sub>y</sub> the latent s<sub>p</sub>ace’s su<sub>pp</sub>ort to factorize [47], <sub>w</sub>hil<sub>e</sub> <sub>ano</sub>th<sub>er</sub> <sub>en</sub>f<sub>orces</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>con</sub>diti<sub>ona</sub>ll<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>covar</sub>i<sub>-</sub> ates [19]. The ver<sub>y</sub> definitions and metrics of disentan<sub>g</sub>lement must also be rethou<sub>g</sub>ht to remain valid under de<sub>p</sub>endence [1].

## 3 Proposed framework

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce a un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k th<sub>a</sub>t f<sub>orma</sub>ll<sub>y</sub> <sub>exp</sub>l<sub>ores</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> <sub>on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space.</sub>

## 3.1 Notation and problem setup

Let $Z \in \mathbb { R } ^ { P }$ b<sub>e a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>ran</sub>d<sub>om vec</sub>t<sub>or w</sub>ith <sub>covar</sub>i<sub>ance ma</sub>t<sub>r</sub>i<sub>x</sub> $\Sigma _ { Z } : =$ Cov(�) and unit mar<sub>g</sub>inal variances $\mathrm { V a r } ( Z _ { p } ) = 1$ f<sub>or</sub> $p = 1 , \ldots , P .$ Let $Y \in \mathbb { R } ^ { P }$ b<sub>e a vec</sub>t<sub>or o</sub>f <sub>o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es, w</sub>ith V<sub>ar</sub> $( Y _ { p } ) = 1$ f<sub>or</sub> $p = 1 , \ldots , P ;$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e cross-covar</sub>i<sub>ance ma</sub>t<sub>r</sub>i<sub>x</sub> $\Sigma _ { Z Y } : =$ $\operatorname { C o v } ( Z , Y )$ . Our o<sup>b</sup>jective is a <sup>l</sup>atent representation in w<sup>h</sup>ic<sup>h</sup> every dimension is <sub>p</sub>aired with the covariate<sub>,</sub> so that we obtain stron<sub>g</sub> ali<sub>g</sub>nment in <sub>p</sub>airs $( Z _ { p } , Y _ { p } )$ f<sub>or</sub> $p = 1 , \ldots , P .$

W<sub>e assume</sub> th<sub>a</sub>t th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> $Z$ i<sub>s non-</sub>id<sub>en</sub>tifi<sub>a</sub>bl<sub>e,</sub> i<sub>n</sub> th<sub>e sense</sub> th<sub>a</sub>t f<sub>or any</sub> i<sub>nver</sub>tibl<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $T \in \mathbb { R } ^ { P \times P }$ <sub>,</sub> th<sub>e</sub> li<sub>near</sub>l<sub>y</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>vec</sub>t<sub>or</sub> $\tilde { Z } = T Z$ <sub>p</sub>rovides an e<sub>q</sub>uivalent re<sub>p</sub>resentation <sub>o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> �<sub>.</sub> W<sub>e exp</sub>l<sub>o</sub>it thi<sub>s</sub> f<sub>ree</sub>d<sub>om</sub> t<sub>o se</sub>l<sub>ec</sub>t <sub>a</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> � th<sub>a</sub>t i<sub>mposes a</sub> d<sub>es</sub>i<sub>ra</sub>bl<sub>e s</sub>t<sub>ruc</sub>t<sub>ure on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space,</sub> th<sub>ere</sub>b<sub>y</sub> definin<sub>g</sub> a canonical structured re<sub>p</sub>resentation and resolvin<sub>g</sub> re<sub>p</sub>- resentation ambi<sub>g</sub>uit<sub>y</sub>. Here � ma<sub>y</sub> be a sin<sub>g</sub>le vector $X \in \mathbb { R } ^ { d } \mathrm { o r }$ i<sub>n</sub> th<sub>e mu</sub>lti<sub>-mo</sub>d<sub>a</sub>l <sub>case, a co</sub>ll<sub>ec</sub>ti<sub>on o</sub>f <sub>vec</sub>t<sub>ors</sub> $\{ X ^ { ( m ) } \in \mathbb { R } ^ { d ^ { ( m ) } } \} _ { m = 1 } ^ { M }$ <sub>ac</sub>r<sub>oss</sub> � m<sub>o</sub>d<sub>a</sub>liti<sub>es.</sub>

In particular, we consider three disentanglement regimes: (i) strong latent dependence on covariates, which maximizes the <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween eac</sub>h l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on an</sub>d it<sub>s covar</sub>i<sub>a</sub>t<sub>e w</sub>hil<sub>e</sub> l<sub>eav</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t d<sub>epen</sub>d<sub>ence</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>o</sub>th<sub>erw</sub>i<sub>se</sub> <sub>uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d<sub>,</sub> (ii) decorrelated/independent latent dimensions, which enforces uncorrelatedness amon<sub>g</sub> the latent dimensions (inde<sub>p</sub>endence under a joint Gaussian assumption) while aligning each latent dimension with its covariate, and (iii) intermediate latent inter-dependence, which weights the alignment objective against <sub>prox</sub>i<sub>m</sub>it<sub>y</sub> t<sub>o</sub> th<sub>e</sub> d<sub>ecorre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>so</sub>l<sub>u</sub>ti<sub>on an</sub>d th<sub>ere</sub>b<sub>y</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>t<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo ex</sub>t<sub>remes.</sub> W<sub>e a</sub>dditi<sub>ona</sub>ll<sub>y exam</sub>i<sub>ne a</sub> f<sub>our</sub>th <sub>sce-</sub> nario, (iv) exclusive latent-dimension-covariate dependence, i<sub>n w</sub>hi<sub>c</sub>h <sub>eac</sub>h l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> i<sub>s requ</sub>i<sub>re</sub>d t<sub>o corre</sub>l<sub>a</sub>t<sub>e on</sub>l<sub>y w</sub>ith its own covariate.

We <sub>p</sub>ose the <sub>p</sub>roblem as <sub>p</sub>airwise latent-covariate de<sub>p</sub>endence under structural constraints on the latent s<sub>p</sub>ace, b<sub>y</sub> framin<sub>g</sub> (i)-(iv) within a sin<sub>g</sub>le unified setu<sub>p</sub>. In <sub>p</sub>articular<sub>,</sub> in followin<sub>g</sub> section we <sub>p</sub>rove that re<sub>g</sub>ime (iii) arises as a continuous inter<sub>p</sub>olation between (i) and (ii), and an orderin<sub>g</sub> orderin<sub>g</sub> of all four re<sub>g</sub>imes b<sub>y</sub> their ali<sub>g</sub>nment stren<sub>g</sub>th.

## 3.2 Optimization-based formulations and solutions

We be<sub>g</sub>in with the inde<sub>p</sub>endence re<sub>g</sub>ime (ii), since its whitenin<sub>g</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> fi<sub>xes a canon</sub>i<sub>ca</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> $\overline { Z }$ <sub>on w</sub>hi<sub>c</sub>h <sub>a</sub>ll <sub>re-</sub> <sub>ma</sub>i<sub>n</sub>i<sub>ng so</sub>l<sub>u</sub>ti<sub>ons are expresse</sub>d<sub>,</sub> th<sub>en we so</sub>l<sub>ve</sub> th<sub>e uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d re<sub>g</sub><sup>i</sup>me $( \mathrm { i } ) ,$ followed b<sub>y</sub> the inter<sub>p</sub>olation (iii), and finall<sub>y</sub> the exclusivede<sub>p</sub>endence re<sub>g</sub>ime (iv).

3.2.1 Independent latent dimensions. We first consider the problem <sub>o</sub>f fi<sub>n</sub>di<sub>ng a w</sub>hit<sub>en</sub>i<sub>ng</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> � <sub>suc</sub>h th<sub>a</sub>t th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es are uncorre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>ze</sub>d $\mathrm { C o v } ( T Z ) = I ;$ , or e<sub>q</sub>uivalentl<sub>y</sub> $T \Sigma _ { Z } T ^ { \prime } = I $ <sub>.</sub> Thi<sub>s cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>a</sub>l<sub>one</sub> d<sub>oes no</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> � uniquely: if � satisfies the above condition, then so does �� for an<sub>y</sub> ortho<sub>g</sub>onal matrix � [30], so whitenin<sub>g</sub> is defined onl<sub>y</sub> u<sub>p</sub> to an ortho<sub>g</sub>onal rotation. A canonical choice for � is the s<sub>y</sub>mmetric whitenin<sub>g</sub> transformation $\Sigma _ { 7 } ^ { - 1 / 2 }$ <sub>,</sub> the <sub>u</sub>ni<sub>qu</sub>e s<sub>y</sub>mmetric <sub>p</sub>ositi<sub>v</sub>ed<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> th<sub>a</sub>t <sub>w</sub>hit<sub>ens</sub> ${ \bar { Z } } .$ Th<sub>e</sub> <sub>genera</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> h<sub>as</sub> <sub>a</sub> f<sub>orm</sub>

$$
T = Q \Sigma _ { Z } ^ { - 1 / 2 } , \quad Q ^ { \prime } Q = I .
$$

To resol<sub>v</sub>e the remainin<sub>g</sub> ambi<sub>gu</sub>it<sub>y, w</sub>e incor<sub>p</sub>orate the co<sub>v</sub>ariates � and seek a transformation that<sub>,</sub> in addition to whitenin<sub>g,</sub> maximizes the ali<sub>g</sub>nment in latent dimension-co<sub>v</sub>ariate <sub>p</sub>airs $( \tilde { Z } _ { p } , Y _ { p } )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub>

$$
\operatorname* { m a x } _ { T } \mathrm { t r } ( T \Sigma _ { Z Y } ) \mathrm { s . t . } T \Sigma _ { Z } T ^ { \prime } = I .\tag{1}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> � <sub>w</sub>ith $Q \Sigma _ { Z } ^ { - 1 / 2 }$ <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> t<sub>o</sub>

$$
\operatorname* { m a x } _ { Q ^ { \prime } Q = I } \mathrm { t r } \bigl ( Q \Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z Y } \bigr ) ,
$$

which is the ortho<sub>g</sub>onal Procrustes <sub>p</sub>roblem [49] (Section A.1). Let

$$
\Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z Y } = U D V ^ { \prime }
$$

be the sin<sub>g</sub>ular value decom<sub>p</sub>osition. Then the o<sub>p</sub>timal solution is

$$
Q ^ { * } = V U ^ { \prime } , \quad T _ { 0 } ^ { * } = Q ^ { * } \Sigma _ { Z } ^ { - 1 / 2 } .
$$

In terms o $\cdot _ { \Sigma _ { Z } }$ <sub>an</sub>d $\Sigma _ { Z Y }$ (Section A.2) $T _ { 0 } ^ { * }$ e<sub>q</sub>ua<sup>l</sup>s

$$
T _ { 0 } ^ { * } = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { - 1 / 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } .\tag{2}
$$

Note that (2) is well-defined whenever $\Sigma _ { Z }$ is <sub>p</sub>ositi<sub>v</sub>e definite.

I<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng sec</sub>ti<sub>ons, we wor</sub>k <sub>w</sub>ith l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> th<sub>a</sub>t h<sub>ave</sub> b<sub>een w</sub>hit<sub>ene</sub>d b<sub>y</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $T _ { 0 } ^ { * }$ <sub>, an</sub>d <sub>we</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d $Z$ <sup>b</sup><sub>y</sub> ${ \overline { { Z } } } ~ : = ~ T _ { 0 } ^ { * } Z$ <sub>.</sub> Thi<sub>s s</sub>i<sub>mp</sub>lifi<sub>es</sub> th<sub>e a</sub>l<sub>ge</sub>b<sub>ra w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> th<sub>e genera</sub>lit<sub>y o</sub>f th<sub>e resu</sub>lt<sub>s, as</sub> t<sub>o recover</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orm</sub> on the ori<sub>g</sub>inal $Z$ we sim<sub>p</sub>l<sub>y</sub> com<sub>p</sub>ose the o<sub>p</sub>timal transformations $T ( \overline { { Z } } )$ <sub>w</sub>ith $T _ { 0 } ^ { * }$ (justification that this yields the same solution as <sub>so</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e pro</sub>bl<sub>em</sub> f<sub>or a genera</sub>l $Z$ is <sub>g</sub>iven in Section A.4).

3.2.2 Strong latent dependence on covariates. We now consider a formulation that <sub>p</sub>rioritizes ali<sub>g</sub>nment with � while relaxin<sub>g</sub> the d<sub>eman</sub>d <sub>on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, we on</sub>l<sub>y en</sub>f<sub>orce</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> h<sub>ave un</sub>it <sub>marg</sub>i<sub>na</sub>l <sub>var</sub>i<sub>ances,</sub> th<sub>us</sub> th<sub>e pro</sub>bl<sub>em</sub> i<sub>s</sub>

$$
\operatorname* { m a x } _ { T } ~ \mathrm { t r } ( T \Sigma _ { \overline { { { Z } } } Y } ) \quad \mathrm { s . t . } \quad \mathrm { d i a g } ( T T ^ { \prime } ) = ( 1 , \dots , 1 ) ^ { \prime } .\tag{3}
$$

Let $\boldsymbol { T } = [ t _ { 1 } , \ldots , t _ { P } ] ^ { \prime }$ <sub>, w</sub>h<sub>ere</sub> $t _ { \mathcal { P } } \in \mathbb { R } ^ { P }$ <sub>are</sub> th<sub>e rows o</sub>f $T .$ <sub>.</sub> Th<sub>e con-</sub> <sub>s</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> $\left\| t _ { p } \right\| = 1$ f<sub>or a</sub>ll ${ \boldsymbol { \mathit { p } } } ,$ an<sup>d</sup> t<sup>h</sup>e o<sup>b</sup>jective term <sup>d</sup>ecomp<sup>oses</sup> <sup>as</sup>

$$
\mathrm { t r } ( T \Sigma _ { \overline { { { Z } } } Y } ) = \sum _ { p = 1 } ^ { P } t _ { p } ^ { \prime } ( \Sigma _ { \overline { { { Z } } } Y } ) . _ { \rlap / P } .
$$

D<sub>e</sub>n<sub>o</sub>tin<sub>g</sub> $a _ { p } : = ( \Sigma _ { \overline { { Z } } Y } ) _ { \cdot p } $ <sub>,</sub> the <sub>p</sub>roblem se<sub>p</sub>arates into � inde<sub>p</sub>endent <sub>su</sub>b<sub>pro</sub>bl<sub>ems:</sub>

$$
\operatorname* { m a x } _ { \| t _ { \pmb { \mathscr { p } } } \| = 1 } ~ t _ { \pmb { \mathscr { p } } } ^ { \prime } a _ { \pmb { \mathscr { p } } } .
$$

B<sub>y</sub> the Cauch<sub>y</sub>-Schwarz ine<sub>q</sub>ualit<sub>y,</sub>

$$
t _ { p } ^ { \prime } a _ { p } \leq \| t _ { p } \| \| a _ { p } \| ,
$$

with e<sub>q</sub>ualit<sub>y</sub> if and onl<sub>y</sub> i $\mathrm { f } t _ { p }$ i<sub>s</sub> <sub>co</sub>lli<sub>near</sub> <sub>w</sub>ith $a _ { p }$ . Hence<sub>,</sub> the o<sub>p</sub>timal <sub>so</sub>l<sub>u</sub>ti<sub>o</sub>n i<sub>s</sub> $\begin{array} { r } { t _ { \dot { p } } ^ { \ast } = \frac { { a } _ { \dot { p } } } { \left\| { a } _ { \dot { p } } \right\| } \operatorname { f o r } \left\| \dot { a } _ { \dot { p } } \right\| \neq 0 . } \end{array}$ or e<sub>q</sub>uivalentl<sub>y,</sub>

$$
t _ { \mathscr { p } } ^ { * } = \frac { \bigl ( \Sigma _ { \overline { { Z } } Y } \bigr ) _ { \cdot \mathscr { p } } } { \| \bigl ( \Sigma _ { \overline { { Z } } Y } \bigr ) _ { \cdot \mathscr { p } } \| } .\tag{4}
$$

For <sub>g</sub>eneral $Z ,$ the corres<sub>p</sub>ondin<sub>g</sub> transformation is $T _ { 1 } ^ { * } = T _ { 1 } ^ { * } ( \overline { { Z } } ) \ : T _ { 0 } ^ { * }$ Th<sub>e</sub> <sub>co</sub>nn<sub>ec</sub>ti<sub>o</sub>n t<sub>o</sub> r<sub>eg</sub>r<sub>ess</sub>i<sub>o</sub>n i<sub>s</sub> di<sub>scusse</sub>d in S<sub>ec</sub>ti<sub>o</sub>n A<sub>.</sub>3<sub>.</sub>

3.2.3 Intermediate scenarios. We now interpolate between the two r<sub>eg</sub>im<sub>es.</sub> L<sub>e</sub>t $\lambda \in [ 0 , 1 ]$ <sub>an</sub>d <sub>cons</sub>id<sub>er</sub>

max $\lambda \mathrm { t r } ( T \Sigma _ { \overline { { { Z } } } Y } ) - \frac { 1 - \lambda } { 2 } \| T - I \| _ { F } ^ { 2 } \mathrm { s . t . } \mathrm { d i a g } ( T T ^ { \prime } ) = ( 1 , \ldots , 1 ) ^ { \prime } ,$ (5) � F<sub>o</sub>r $\lambda = 0 , T ^ { * } = I ,$ <sub>w</sub>hil<sub>e</sub> f<sub>or</sub> $\lambda = 1$ we recover (3). Usin<sub>g</sub> the identit<sub>y</sub>

$$
\| T - I \| _ { F } ^ { 2 } = \sum _ { p = 1 } ^ { P } \| t _ { p } - e _ { p } \| ^ { 2 } ,
$$

<sub>w</sub>h<sub>ere</sub> $e _ { p }$ i<sub>s</sub> th<sub>e</sub> <sub>�-</sub>th <sub>canon</sub>i<sub>ca</sub>l b<sub>as</sub>i<sub>s</sub> <sub>vec</sub>t<sub>or,</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> d<sub>ecomposes</sub> into:

$$
\operatorname* { m a x } _ { \| t _ { p } \| = 1 } \lambda t _ { \hat { p } } ^ { \prime } a _ { \hat { p } } - \frac { 1 - \lambda } { 2 } \| t _ { \hat { p } } - e _ { \hat { p } } \| ^ { 2 } ,
$$

<sub>w</sub>h<sub>ere</sub> $a _ { p } : = ( \Sigma _ { \overline { { { Z } } } Y } ) _ { \cdot p } .$ Ex<sub>p</sub>andin<sub>g</sub> the <sub>q</sub>uadratic term and usin<sub>g</sub> $\| t _ { p } \| = 1$ , the objective is equivalent to (up to a constant independent <sub>o</sub>f $t _ { p } )$

$$
\operatorname* { m a x } _ { \| t _ { \hat { p } } \| = 1 } \lambda t _ { \hat { p } } ^ { \prime } a _ { \hat { p } } + \left( 1 - \lambda \right) t _ { \hat { p } } ^ { \prime } e _ { \hat { p } } .
$$

$\mathrm { B y }$ the Cauch<sub>y</sub>-Schwarz ine<sub>q</sub>ualit<sub>y,</sub> the o<sub>p</sub>timum is attained when $t _ { p }$ i<sub>s co</sub>lli<sub>near w</sub>ith $\lambda a _ { p } + ( 1 - \lambda ) e _ { p }$ <sub>, w</sub>hi<sub>c</sub>h <sub>resu</sub>lt<sub>s</sub> i<sub>n</sub>

$$
t _ { \pmb { \jmath } } ^ { * } = \frac { \lambda ( \Sigma _ { \overline { { { Z } } } Y } ) . _ { \pmb { \jmath } } + ( 1 - \lambda ) e _ { p } } { | | \lambda ( \Sigma _ { \overline { { { Z } } } Y } ) . _ { \pmb { \jmath } } + ( 1 - \lambda ) e _ { p } | | } .\tag{6}
$$

For <sub>g</sub>eneral $Z ,$ <sub>we o</sub>bt<sub>a</sub>i<sub>n</sub> $T _ { \lambda } ^ { \ast } = T _ { \lambda } ^ { \ast } ( \overline { { Z } } ) T _ { 0 } ^ { \ast } . T _ { \lambda } ^ { \ast }$ recovers the <sub>p</sub>revious <sub>so</sub>l<sub>u</sub>ti<sub>ons</sub> $T _ { 0 } ^ { * }$ <sub>an</sub>d $T _ { 1 } ^ { * }$ f<sub>or</sub> $\lambda = 0$ <sub>an</sub>d $\lambda = 1 ,$ res<sub>p</sub>ectivel<sub>y</sub> (see Table S1).

3.2.4 Exclusive latent dependence with covariates. We now require <sub>eac</sub>h l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> t<sub>o</sub> b<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith it<sub>s covar</sub>i<sub>a</sub>t<sub>e w</sub>hil<sub>e</sub> b<sub>e</sub>i<sub>ng cross-uncorre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>a</sub>ll <sub>o</sub>th<sub>ers,</sub> i<sub>.e.</sub> th<sub>e cross-covar</sub>i<sub>ance</sub> $\mathrm { C o v } ( T \overline { { Z } } , Y ) = T \Sigma _ { \overline { { Z } } Y }$ is dia<sub>g</sub>onal

max $\mathrm { t r } ( T \Sigma _ { \overline { { Z } } Y } )$ s.t. $( T \Sigma _ { \overline { { { Z } } } Y } ) _ { p q } = 0 ( p \neq q )$ , dia<sub>g</sub>(�� <sup>′</sup> ) = (1, . . . , 1)<sup>′</sup> . �

(7)

With $T = [ t _ { 1 } , \ldots , t _ { P } ] ^ { \prime }$ <sub>an</sub>d $a _ { p } : = ( \Sigma _ { \overline { { Z } } Y } ) _ { \cdot p } ,$ <sub>eac</sub>h <sub>row so</sub>l<sub>ves</sub>

$$
\operatorname* { m a x } _ { \| t _ { \rho } \| = 1 } ~ t _ { \rho } ^ { \prime } a _ { \rho } ~ \mathrm { s . t . } ~ t _ { \rho } ^ { \prime } a _ { q } = 0 ~ ( q \neq p ) .
$$

Th<sub>e</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> f<sub>orce</sub> $t _ { \ / p }$ t<sub>o</sub> b<sub>e</sub> <sub>or</sub>th<sub>ogona</sub>l t<sub>o</sub> <sub>a</sub>ll $a _ { q }$ f<sub>or</sub> $q \neq p ,$ <sub>so</sub> th<sub>e</sub> o<sub>p</sub>timum lies in a one-dimensional subs<sub>p</sub>ace s<sub>p</sub>anned b<sub>y</sub> $( \Sigma _ { \overline { { Z } } Y } ^ { - 1 } ) _ { \rlap { / } p }$ · <sup>as</sup> $\bigl ( \Sigma _ { \overline { { { Z } } } Y } ^ { - 1 } \bigr ) _ { \pmb { p } \cdot } \bigl ( \Sigma _ { \overline { { { Z } } } Y } \bigr ) _ { \cdot q } = \mathbb { I } ( \pmb { p } = q )$ <sub>.</sub> Th<sub>us,</sub> t<sub>a</sub>ki<sub>ng</sub> i<sub>n</sub>t<sub>o accoun</sub>t th<sub>e secon</sub>d constra<sup>i</sup>nt, we <sub>g</sub>et

$$
t _ { \hat { p } } ^ { * } = \frac { ( \Sigma _ { \overline { { Z } } Y } ^ { - 1 } ) _ { \hat { p } } ^ { \prime } . } { \left\| ( \Sigma _ { \overline { { Z } } Y } ^ { - 1 } ) _ { \hat { p } } . \right\| }\tag{8}
$$

<sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $T _ { \mathrm { e x } } ^ { \ast } = T _ { \mathrm { e x } } ^ { \ast } ( \overline { { Z } } ) T _ { 0 } ^ { \ast }$ for a <sub>g</sub>eneral �. Note that here invertibilit<sub>y</sub> o $\Sigma _ { Z Y }$ is re<sub>q</sub>uired for feasibilit<sub>y</sub>.

3.2.5 The comparison of the strength of the dependencies of the regimes. Theorem 3.1 orders the four regimes by latent-covariate de<sub>p</sub>endence stren<sub>g</sub>th (<sub>p</sub>roof in Section A.6): de<sub>p</sub>endence increases from the inde<sub>p</sub>endent re<sub>g</sub>ime (ii) throu<sub>g</sub>h the intermediate re<sub>g</sub>ime (iii) to the unconstrained re<sub>g</sub>ime (i), while the exclusive re<sub>g</sub>ime (iv) attains weaker ali<sub>g</sub>nment than all of (i)-(iii). The result makes the t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>con-</sub> straints on the latent <sub>p</sub>recise: in <sub>p</sub>articular latent inde<sub>p</sub>endence constraint in (ii) carries a cost in ali<sub>g</sub>nment relative to the unconstrained re<sub>g</sub>ime (i), and the intermediate re<sub>g</sub>ime (iii) traces the entire <sub>spec</sub>t<sub>rum</sub> <sub>o</sub>f <sub>so</sub>l<sub>u</sub>ti<sub>ons</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> <sub>ex</sub>t<sub>remes.</sub> F<sub>or</sub> <sub>genera</sub>l �<sub>,</sub> th<sub>e</sub> re<sub>g</sub>ime re<sub>g</sub>ularizes the transformation usin<sub>g</sub> $\lVert \boldsymbol { T } ( T _ { 0 } ^ { * } ) ^ { - 1 } - \boldsymbol { I } \rVert _ { F }$ <sub>,</sub> th<sub>e</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>ecorre</sub>l<sub>a</sub>ti<sub>ng so</sub>l<sub>u</sub>ti<sub>on.</sub> Thi<sub>s c</sub>h<sub>o</sub>i<sub>ce ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em separa</sub>bl<sub>e an</sub>d <sub>a</sub>d<sub>m</sub>it<sub>s</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm so</sub>l<sub>u</sub>ti<sub>ons</sub> $T _ { \lambda } ,$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s no</sub>t th<sub>e case</sub> f<sub>or</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-corre</sub>l<sub>a</sub>ti<sub>on pena</sub>lt<sub>y</sub> $\| \operatorname { C o v } ( \tilde { Z } ) - I \| _ { F }$ All f<sub>our</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons an</sub>d th<sub>e</sub>i<sub>r per-</sub>di<sub>mens</sub>i<sub>on covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen-</sub> d<sub>e</sub>n<sub>ces a</sub>r<sub>e su</sub>mm<sub>a</sub>ri<sub>se</sub>d in S<sub>ec</sub>ti<sub>o</sub>n $_ { \mathrm { A . 5 } }$

Theorem 3.1. (a) Assume $\Sigma _ { Z }$ is positive definite and that $\Sigma _ { Z Y }$ has full rank. Let $J ( T ) : = \mathrm { t r } \big ( \mathrm { C o v } ( \tilde { Z } , Y ) \big )$ denote the total latent dimension-covariate alignment, where $\tilde { Z } = T Z$ . Then the optimal transformations ofthe four regimes are ordered by their alignment strength,

$$
J ( T _ { \mathrm { e x } } ^ { * } ) ~ \le ~ J ( T _ { 0 } ^ { * } ) ~ \le ~ J ( T _ { \lambda } ^ { * } ) ~ \le ~ J ( T _ { 1 } ^ { * } ) , ~ \lambda \in [ 0 , 1 ]
$$

and $\lambda \mapsto J ( T _ { \lambda } ^ { * } )$ is non-decreasing, so the intermediate regime interpolates $J ( T _ { 0 } ^ { * } ) \dot { }$ and ${ \cal J } ( T _ { 1 } ^ { * } )$ . Moreover, all three inequalities hold with equality simultaneously if and only $i f \big ( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \big ) ^ { 1 / 2 }$ is diagonal, i.e. the whitened cross-covariance columns are mutually orthogonal. (b) Let $F ( T ) = \| \operatorname { C o v } ( \tilde { Z } - I \| _ { F } = \| T \Sigma _ { Z } T ^ { \top } - I \| _ { F }$ measure the deviation from mutually decorrelated unit-variance latent dimensions, then

$$
F ( T _ { 0 } ^ { * } ) \ \leq \ F ( T _ { 1 } ^ { * } ) a n d F ( T _ { 0 } ^ { * } ) \ \leq \ F ( T _ { \mathrm { e x } } ^ { * } ) .
$$

## 3.3 Statistical model

W<sub>e now em</sub>b<sub>e</sub>d th<sub>e</sub> f<sub>ramewor</sub>k i<sub>n a pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>ac</sub>t<sub>or ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>mo</sub>d<sub>e</sub>l th<sub>a</sub>t <sub>uses</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>es</sub> t<sub>o</sub> l<sub>earn</sub> i<sub>n</sub>f<sub>orme</sub>d f<sub>ac</sub>t<sub>ors an</sub>d th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y</sub> $T _ { \lambda }$ t<sub>o</sub> <sub>res</sub>h<sub>ape</sub> th<sub>em.</sub>

3.3.1 Factor analysis with pairwise dependencies with covariates. W<sub>e cons</sub>id<sub>er a ran</sub>d<sub>om vec</sub>t<sub>or</sub> $X \in \mathbb { R } ^ { D }$ <sub>o</sub>f � f<sub>ea</sub>t<sub>ures</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng a</sub> f<sub>ac</sub>t<sub>or mo</sub>d<sub>e</sub>l

$$
X | Z , W , \tau \sim { \cal N } ( W Z , D _ { 1 / \tau } ) ,
$$

where � is a latent re<sub>p</sub>resentation<sub>,</sub> with each com<sub>p</sub>onent corre-<sub>spon</sub>di<sub>ng</sub> t<sub>o</sub> <sub>a</sub> f<sub>ac</sub>t<sub>or,</sub> <sub>an</sub>d $W \in \mathbb { R } ^ { D \times P }$ is a matrix of loadin<sub>g</sub>s. The matrix $D _ { 1 / \tau } = \mathrm { d i a g } ( 1 / \tau _ { 1 } , . . . , 1 / \tau _ { D } )$ <sub>enco</sub>d<sub>es</sub> f<sub>ea</sub>t<sub>ure-spec</sub>ifi<sub>c no</sub>i<sub>se</sub> <sub>var</sub>i<sub>ances, w</sub>h<sub>ere ��</sub> d<sub>eno</sub>t<sub>es</sub> th<sub>e prec</sub>i<sub>s</sub>i<sub>on o</sub>f th<sub>e</sub> �<sub>-</sub>th <sub>o</sub>b<sub>serve</sub>d f<sub>ea-</sub> t<sub>ure.</sub> I<sub>n</sub> thi<sub>s mo</sub>d<sub>e</sub>l<sub>, w</sub>ith <sub>no</sub> f<sub>ur</sub>th<sub>er cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s on</sub> � <sub>or</sub> $Z ,$ th<sub>e</sub> r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> in<sub>va</sub>ri<sub>a</sub>nt t<sub>o</sub> in<sub>ve</sub>rtibl<sub>e</sub> lin<sub>ea</sub>r tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n<sub>s, as</sub>

$$
W Z = ( W T ^ { - 1 } ) ( T Z ) = \tilde { W } \tilde { Z } ,
$$

for an<sub>y</sub> invertible matrix $T \in \mathbb { R } ^ { P \times P }$

W<sub>e ex</sub>t<sub>en</sub>d thi<sub>s</sub> f<sub>ramewor</sub>k b<sub>y</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng a vec</sub>t<sub>or o</sub>f <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> $Y \in \mathbb { R } ^ { P }$ <sub>, w</sub>ith <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> m<sub>a</sub>trix $\Sigma _ { Y } : = \operatorname { C o v } ( Y )$ <sub>, assume</sub>d t<sub>o</sub> h<sub>ave</sub> <sub>un</sub>it di<sub>agona</sub>l <sub>en</sub>t<sub>r</sub>i<sub>es.</sub> E<sub>ac</sub>h l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>or</sub> i<sub>s</sub> th<sub>en mo</sub>d<sub>e</sub>l<sub>e</sub>d <sub>as</sub> b<sub>e</sub>i<sub>ng</sub> informed by t<sup>h</sup>e corresponding covariate t<sup>h</sup>roug<sup>h</sup> a pairwise <sup>l</sup>inear relationshi<sub>p</sub> (see Fi<sub>g</sub>. S6)

$$
Z | Y \sim N ( \beta ^ { ( 0 ) } + D _ { \beta } Y , D _ { 1 - \beta ^ { 2 } } ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\beta ^ { ( 0 ) } \in \mathbb { R } ^ { P }$ is a <sub>v</sub>ector of interce<sub>p</sub>ts<sub>,</sub> $D _ { \beta } = \mathrm { d i a g } ( \beta _ { 1 } , . . . , \beta _ { P } )$ <sub>an</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> $\beta _ { p } \in \left[ 0 , 1 \right]$ <sub>.</sub> Th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $D _ { 1 - \beta ^ { 2 } } = \mathrm { d i a g } \big ( 1 - \beta _ { 1 } ^ { 2 } , . . . , 1 -$ $\beta _ { P } ^ { 2 } )$ <sub>ensures un</sub>it <sub>var</sub>i<sub>ance o</sub>f <sub>eac</sub>h l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>or.</sub> Th<sub>e</sub> f<sub>u</sub>ll <sub>covar</sub>i<sub>ance</sub> of � is <sub>g</sub>iven b<sub>y</sub>

$$
\Sigma _ { Z } : = \operatorname { C o v } ( Z ) = D _ { \beta } \Sigma _ { Y } D _ { \beta } + D _ { 1 - \beta ^ { 2 } }
$$

<sub>w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors are genera</sub>ll<sub>y</sub> d<sub>epen</sub>d<sub>en</sub>t d<sub>ue</sub> t<sub>o</sub> th<sub>e</sub> d<sub>epen</sub>d<sub>ence s</sub>t<sub>ruc</sub>t<sub>ure o</sub>f�<sub>.</sub> H<sub>owever, con</sub>diti<sub>ona</sub>ll<sub>y on</sub> �<sub>,</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>ors</sub> b<sub>eco</sub>m<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>s</sub>in<sub>ce</sub> th<sub>e</sub>ir <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> m<sub>a</sub>trix i<sub>s</sub> di<sub>agona</sub>l<sub>:</sub>

$$
\Sigma _ { Z | Y } : = \operatorname { C o v } ( Z | Y ) = D _ { 1 - \beta ^ { 2 } } .
$$

U<sub>n</sub>d<sub>er</sub> thi<sub>s mo</sub>d<sub>e</sub>l<sub>,</sub> th<sub>e cross-covar</sub>i<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors an</sub>d co<sub>v</sub>ariates is <sub>g</sub>i<sub>v</sub>en b<sub>y</sub>

$$
\Sigma _ { Z Y } : = \operatorname { C o v } ( Z , Y ) = D _ { \beta } \Sigma _ { Y } .
$$

3.3.2 Informedfactor analysis (iFA). We introduce a probabilistic <sub>grap</sub>hi<sub>ca</sub>l <sub>mo</sub>d<sub>e</sub>l th<sub>a</sub>t f<sub>o</sub>ll<sub>ows</sub> th<sub>e</sub> <sub>genera</sub>ti<sub>ve</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f S<sub>ec-</sub> ti<sub>on</sub> 3<sub>.</sub>3<sub>.</sub>1 <sub>augmen</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>a</sub>dditi<sub>ona</sub>l <sub>un</sub>i<sub>n</sub>f<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors</sub> (Fi<sub>g</sub>. S6) and usin<sub>g</sub> transformations (i)-(iii) introduced in Section 3.2. Let $\mathbf { X } \in \mathbb { R } ^ { N \times D }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e o</sub>b<sub>serve</sub>d f<sub>ea</sub>t<sub>ures,</sub> $\mathbf { Y } \in \mathbb { R } ^ { N \times P }$ th<sub>e</sub> covariates (assumed standardized to unit variance <sub>p</sub>er column), and $\boldsymbol { Z } \in \mathbb { R } ^ { N \times K }$ th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors w</sub>ith $P \leq K .$ Th<sub>e</sub> fi<sub>rs</sub>t � f<sub>ac</sub>t<sub>ors are</sub> informed b<sub>y</sub> the covariates as in Section 3.3.1<sub>;</sub> the remainin<sub>g</sub> $K - P$ <sub>are un</sub>i<sub>n</sub>f<sub>orme</sub>d <sub>an</sub>d f<sub>o</sub>ll<sub>ow a s</sub>t<sub>an</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an pr</sub>i<sub>or.</sub> Th<sub>e</sub> i<sub>nver</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $T \in \mathbb { R } ^ { K \times K }$ is a<sub>pp</sub>lied to factors $\tilde { z } _ { n , \cdot } = z _ { n , \cdot } T ^ { \prime }$ <sub>,</sub> <sub>an</sub>d <sub>we</sub> t<sub>a</sub>k<sub>e</sub>� t<sub>o</sub> <sub>ac</sub>t t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>ll<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>un</sub>i<sub>n</sub>f<sub>orme</sub>d bl<sub>oc</sub>k<sub>,</sub> <sub>so</sub> $\tilde { z } _ { n , P + 1 : K } = z _ { n , P + 1 : K } .$ We <sub>p</sub>lace an automatic relevance determination (ARD) <sub>p</sub>rior on the l<sub>oa</sub>di<sub>ngs</sub> $W \in \mathbb { R } ^ { D \times K }$ to encoura<sub>g</sub>e factor-level s<sub>p</sub>arsit<sub>y,</sub> and Gamma <sub>p</sub>riors on the feature-s<sub>p</sub>ecific noise <sub>p</sub>recisions $\tau _ { d } .$ . T<sup>h</sup>e joint <sup>d</sup>istrib<sub>u</sub>ti<sub>on</sub> f<sub>ac</sub>t<sub>or</sub>i<sub>zes as</sub>

$$
\wp ( \mathbf { X } , \tilde { \mathbf { Z } } , W , \alpha , \tau \mid \mathbf { Y } , \beta ) = \prod _ { n = 1 } ^ { N } \prod _ { d = 1 } ^ { D } N \left( x _ { n , d } \mid \tilde { z } _ { n , \cdot } w _ { d , \cdot } ^ { \prime } , 1 / \tau _ { d } \right)\tag{9}
$$

$$
\begin{array} { l } { { \displaystyle \prod _ { n = 1 } ^ { N } \prod _ { p = 1 } ^ { P } N ( z _ { n , p } \mid \beta _ { p } ^ { ( 0 ) } + \beta _ { p } y _ { n , p } , 1 - \beta _ { p } ^ { 2 } ) \prod _ { n = 1 } ^ { N } \prod _ { k = P + 1 } ^ { K } N ( z _ { n , k } \mid 0 , 1 ) } } \\ { { \displaystyle \prod _ { d = 1 } ^ { D } \prod _ { k = 1 } ^ { K } N ( w _ { d , k } \mid 0 , 1 / \alpha _ { k } ) \prod _ { k = 1 } ^ { K } \mathcal { G } ( \alpha _ { k } \mid a _ { 0 } ^ { ( \alpha ) } , b _ { 0 } ^ { ( \alpha ) } ) } } \\ { { \displaystyle \prod _ { d = 1 } ^ { D } \mathcal { G } ( \tau _ { d } \mid a _ { 0 } ^ { ( \tau ) } , b _ { 0 } ^ { ( \tau ) } ) . } } \end{array}
$$

In the su<sub>pp</sub>lement E<sub>q</sub>. (10) we <sub>p</sub>rovide the joint distribution for <sub>mu</sub>lti<sub>-mo</sub>d<sub>a</sub>l d<sub>a</sub>t<sub>a.</sub>

3.3.3 Inference. The posterior $\hat { p } ( \tilde { \mathbf { Z } } , W , \alpha , \tau \mid \mathbf { X } , \mathbf { Y } , \beta , \beta ^ { ( 0 ) } )$ <sup>i</sup>s a<sub>p</sub>- <sub>p</sub>roximated b<sub>y</sub> variational inference [7, 53] with the mean-field f<sub>ac</sub>t<sub>o</sub>riz<sub>a</sub>ti<sub>o</sub>n

$$
\begin{array} { l } { \displaystyle q ( \tilde { Z } , W , \alpha , \tau \mid Y , \beta ) = \prod _ { n = 1 } ^ { N } q ( \tilde { z } _ { n , 1 : P } ) \prod _ { n = 1 } ^ { N } \prod _ { k = P + 1 } ^ { K } q ( \tilde { z } _ { n , k } ) } \\ { \displaystyle \prod _ { d = 1 } ^ { D } \prod _ { k = 1 } ^ { K } q ( w _ { d , k } ) \prod _ { k = 1 } ^ { K } q ( \alpha _ { k } ) \prod _ { d = 1 } ^ { D } q ( \tau _ { d } ) . } \end{array}
$$

Variational inference chooses <sub>�</sub> to maximize the e<sub>v</sub>idence lo<sub>w</sub>er bound (ELBO)

$$
\begin{array} { r } { \mathcal { L } ( q , \boldsymbol { \beta } , \boldsymbol { \beta } ^ { ( 0 ) } ) = \mathbb { E } _ { q } \big [ \log \mathnormal { p } ( \mathbf { X } , \mathbf { Z } , W , \alpha , \tau \mid \mathbf { Y } , \boldsymbol { \beta } , \boldsymbol { \beta } ^ { ( 0 ) } ) \big ] - \mathbb { E } _ { q } [ \log q ] , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> l<sub>og</sub> $p ( \mathbf { X } \mid \mathbf { Y } , { \boldsymbol { \beta } } , { \boldsymbol { \beta } } ^ { ( 0 ) } ) \ = \ { \mathcal { L } } + \operatorname { K L } ( q \| { \boldsymbol { p } } ) \ \geq \ { \mathcal { L } }$ . We m<sub>a</sub>ximiz<sub>e</sub> $\mathcal { L }$ b<sub>y coor</sub>di<sub>na</sub>t<sub>e ascen</sub>t<sub>:</sub> f<sub>or eac</sub>h l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e,</sub> th<sub>e</sub> o<sub>p</sub>timal variational <sub>p</sub>arameters are obtained from

$$
\log q ^ { * } ( \theta _ { j } ) = \mathbb { E } _ { q _ { - j } } \log p + \mathrm { c o n s t } ,
$$

<sub>w</sub>hil<sub>e</sub> $\beta$ <sub>an</sub>d $\beta ^ { ( 0 ) }$ are u<sub>p</sub>dated b<sub>y</sub> maximizin<sub>g</sub> $\mathcal { L }$ <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>em</sub> as <sub>p</sub>oint <sub>p</sub>arameters. The u<sub>p</sub>dates for $q ( \alpha _ { k } )$ <sub>an</sub>d $q ( \tau _ { d } )$ <sub>are s</sub>t<sub>an</sub>d<sub>ar</sub>d in Ba<sub>y</sub>esian factor anal<sub>y</sub>sis (e.<sub>g</sub>. [3]). Below we <sub>p</sub>rovide the u<sub>p</sub>dates s<sub>p</sub>ecific to the covariate-informed settin<sub>g</sub>.

Th<sub>e</sub> t<sub>wo na</sub>t<sub>ura</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ces o</sub>f th<sub>e parame</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on o</sub>f l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors</sub> l<sub>ea</sub>d t<sub>o</sub> dif<sub>eren</sub>t <sub>conven</sub>i<sub>ences.</sub> Th<sub>e</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> f<sub>or</sub> � <sub>an</sub>d <sub>�</sub> <sub>are</sub> <sub>s</sub>i<sub>mp</sub>l<sub>er</sub> <sub>w</sub>h<sub>en wr</sub>itt<sub>en</sub> i<sub>n</sub> t<sub>erms o</sub>f $\tilde { z } ,$ <sub>s</sub>i<sub>nce</sub> th<sub>e</sub> lik<sub>e</sub>lih<sub>oo</sub>d d<sub>epen</sub>d<sub>s on</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>ors on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>�</sub>˜<sub>.</sub> Th<sub>e</sub> $\beta _ { p } ^ { ( 0 ) } , \beta _ { p }$ u<sub>p</sub>dates<sub>,</sub> in contrast<sub>,</sub> are sim<sub>p</sub>ler i<sub>n</sub> t<sub>erms o</sub>f <sub>�, s</sub>i<sub>nce</sub> th<sub>e pr</sub>i<sub>or</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>on �.</sub> Th<sub>roug</sub>h<sub>ou</sub>t<sub>, we use</sub> <sub>�</sub>˜ <sub>as</sub> th<sub>e var</sub>i<sub>a</sub>ti<sub>ona</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>e an</sub>d <sub>recover</sub> th<sub>e momen</sub>t<sub>s o</sub>f <sub>� v</sub>i<sub>a</sub> th<sub>e</sub> in<sub>ve</sub>r<sub>se</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n $T ^ { - 1 }$

Updates for informed factors $\tilde { z } _ { n , 1 : P }$ . T<sup>h</sup>e conditiona<sup>l</sup> <sup>l</sup>og-density is a <sub>qu</sub>adratic form in $\tilde { z } _ { n , 1 : P ; }$ , so $q ^ { * } ( \tilde { z } _ { n , 1 : P } ) = N ( \tilde { \mu } _ { n } , \tilde { \Sigma } _ { n } )$ <sub>w</sub>ith

$$
\begin{array} { l } { { \tilde { \Sigma } _ { n } ^ { - 1 } = ( T ^ { \prime } ) ^ { - 1 } D _ { 1 / ( 1 - \beta ^ { 2 } ) } T ^ { - 1 } + \displaystyle \sum _ { d = 1 } ^ { D } \langle \tau _ { d } \rangle \langle w _ { d , 1 : P } w _ { d , 1 : P } ^ { \prime } \rangle , } } \\ { { \tilde { \mu } _ { n } = \tilde { \Sigma } _ { n } \bigg [ T ^ { - 1 } D _ { 1 / ( 1 - \beta ^ { 2 } ) } \left( \beta ^ { ( 0 ) } + D _ { \beta } y _ { n } \right) } } \\ { { \displaystyle \quad \quad + \displaystyle \sum _ { d = 1 } ^ { D } \langle \tau _ { d } \rangle \left( x _ { n , d } - \langle w _ { d , P + 1 : K } \rangle ^ { \prime } \langle \tilde { z } _ { n , P + 1 : K } \rangle \right) \langle w _ { d , 1 : P } \rangle \bigg ] , } } \end{array}
$$

where ⟨·⟩ denotes ex<sub>p</sub>ectation under <sub>�</sub> and $\begin{array} { r } { D _ { 1 / ( 1 - \beta ^ { 2 } ) } = \mathrm { d i a g } \big ( \frac { 1 } { 1 - \beta _ { p } ^ { 2 } } \big ) } \end{array}$

Updates for $\boldsymbol { \beta } _ { p } ^ { 0 }$ and $\beta _ { p }$ . T<sup>h</sup>e ELBO contribution t<sup>h</sup>at depends on $\beta _ { p } ^ { ( 0 ) }$ <sub>an</sub>d $\beta _ { p }$ i<sub>s</sub>

$$
\begin{array} { r } { \mathscr { L } _ { \boldsymbol { p } } ( \beta _ { \boldsymbol { p } } ^ { ( 0 ) } , \beta _ { \boldsymbol { p } } ) = - \frac { N } { 2 } \log ( 1 - \beta _ { \boldsymbol { p } } ^ { 2 } ) - \frac { \sum _ { n = 1 } ^ { N } \mathbb { E } ( z _ { n , \boldsymbol { p } } - \beta _ { \boldsymbol { p } } ^ { ( 0 ) } - \beta _ { \boldsymbol { p } } y _ { n , \boldsymbol { p } } ) ^ { 2 } } { 2 ( 1 - \beta _ { \boldsymbol { p } } ^ { 2 } ) } , } \end{array}
$$

M<sub>a</sub>ximizin<sub>g</sub> <sub>w</sub>ith r<sub>espec</sub>t t<sub>o</sub> $\beta _ { p } ^ { ( 0 ) }$ <sub>g</sub><sup>i</sup>ves $\beta _ { p } ^ { ( 0 ) } \ = \ \overline { { { \langle z _ { p } \rangle } } } - \beta _ { p } \bar { y } _ { p } ,$ <sub>, w</sub>ith $\begin{array} { r } { \overline { { \langle z _ { p } \rangle } } = N ^ { - 1 } \sum _ { n } \langle z _ { n p } \rangle } \end{array}$ <sub>an</sub>d $\begin{array} { r } { \bar { y } _ { \mathcal { P } } = N ^ { - 1 } \sum _ { n } y _ { n \mathcal { P } } } \end{array}$ . S<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> this ex-<sub>p</sub>ression back into $\mathcal { L } _ { p }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { c c c } { { \displaystyle { \mathcal { L } } _ { P } ( \beta _ { P } ) = - \frac { N } { 2 } \log ( 1 - \beta _ { P } ^ { 2 } ) - \frac { S _ { z } ^ { ( p ) } - 2 \beta _ { P } C _ { z y } ^ { ( p ) } + \beta _ { P } ^ { 2 } N } { 2 ( 1 - \beta _ { P } ^ { 2 } ) } , } } \end{array}
$$

<sub>w</sub>ith

$$
\begin{array} { l } { { S _ { z } ^ { ( p ) } = \displaystyle \sum _ { n } \langle ( z _ { n p } - \overline { { { \langle z _ { p } \rangle } } } ) ^ { 2 } \rangle , } } \\ { { { } } } \\ { { C _ { z y } ^ { ( p ) } = \displaystyle \sum _ { n } ( \langle z _ { n p } \rangle - \overline { { { \langle z _ { p } \rangle } } } ) ( y _ { n p } - \bar { y } _ { p } ) . } } \end{array}
$$

Th<sub>e</sub> <sub>s</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>rit<sub>y</sub> <sub>equa</sub>ti<sub>o</sub>n i<sub>s</sub> <sub>cu</sub>bi<sub>c</sub> in $\beta _ { p }$

$$
N \beta _ { p } ^ { 3 } - \bigl ( 1 + \beta _ { p } ^ { 2 } \bigr ) C _ { z y } ^ { ( p ) } + \beta _ { p } S _ { z } ^ { ( p ) } = 0 ,
$$

which we solve numericall<sub>y</sub> (one-dimensional root-findin<sub>g</sub> within (0, 1)). When the variational moments are close to the <sub>p</sub>rior $S _ { z } ^ { ( { p } ) }$ ≈ �<sub>,</sub> we <sub>g</sub>et the closed-form a<sub>pp</sub>roximation $\beta _ { p } \approx C _ { z y } ^ { ( p ) } / N$ <sub>,</sub> i<sub>.e.,</sub> th<sub>e</sub> em<sub>p</sub>irical correlation between $\langle z _ { p } \rangle$ <sub>an</sub>d $y _ { p } .$

Updates for loadings ���. T<sup>h</sup>e conditiona<sup>l l</sup>og-joint is quadratic i<sub>n</sub> <sub>eac</sub>h $w _ { d k }$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $q ^ { * } ( w _ { d k } ) = N \big ( \mu _ { d k } ^ { ( w ) } , ( \sigma _ { d k } ^ { ( w ) } ) ^ { 2 } \big )$ <sub>w</sub>ith <sub>p</sub>recision and mean

$$
\begin{array} { l } { ( \sigma _ { d k } ^ { ( w ) } ) ^ { - 2 } = \langle \alpha _ { k } \rangle + \langle \tau _ { d } \rangle \displaystyle \sum _ { n = 1 } ^ { N } \langle \tilde { z } _ { n k } ^ { 2 } \rangle , } \\ { \mu _ { d k } ^ { ( w ) } = ( \sigma _ { d k } ^ { ( w ) } ) ^ { 2 } \langle \tau _ { d } \rangle \displaystyle \left[ \displaystyle \sum _ { n = 1 } ^ { N } \langle \tilde { z } _ { n k } \rangle x _ { n d } - \displaystyle \sum _ { j \neq k } \langle w _ { d j } \rangle \displaystyle \sum _ { n = 1 } ^ { N } \langle \tilde { z } _ { n j } \tilde { z } _ { n k } \rangle \right] . } \end{array}
$$

Th<sub>e cross-momen</sub>t $\langle \tilde { z } _ { n j } \tilde { z } _ { n k } \rangle$ e<sub>q</sub>ua<sup>l</sup>s $\langle \tilde { z } _ { n j } \rangle \langle \tilde { z } _ { n k } \rangle$ w<sup>h</sup>enever � or <sup>�</sup> lies outside the informed block (mean-field inde<sub>p</sub>endence), and e<sub>q</sub>ua<sup>l</sup>s $\langle \tilde { z } _ { n j } \rangle \langle \tilde { z } _ { n k } \rangle + ( \tilde { \Sigma } _ { n } ) _ { j k }$ <sub>w</sub>h<sub>en</sub> b<sub>o</sub>th $j , k \le P ,$ <sub>,</sub> <sub>w</sub>ith ${ \tilde { \Sigma } } _ { n }$ <sub>as</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>a</sub>b<sub>ove.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> th<sub>ese</sub> <sub>express</sub>i<sub>ons</sub> <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>on</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orme</sub>d<sub>-</sub>bl<sub>oc</sub>k <sub>covar</sub>i<sub>ance exp</sub>li<sub>c</sub>it<sub>, an</sub>d <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d Ba<sub>y</sub>esian factor anal<sub>y</sub>sis u<sub>p</sub>date when $P = 0$

3.3.4 Transformation application. The transformations we want to <sup>a</sup>pp<sup>l</sup>y $( T _ { \lambda } ^ { * }$ of Section 5.0.1) de<sub>p</sub>end on $( \beta , \Sigma _ { Y } )$ <sub>,</sub> so <sub>w</sub>e do not o<sub>p</sub>timise � and � jointly. Instead, we use a two-stage procedure.

Stage 1. We fit the model with $T \ = \ I$ b<sub>y coor</sub>di<sub>na</sub>t<sub>e-ascen</sub>t VI <sub>as</sub> d<sub>escr</sub>ib<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub> Thi<sub>s y</sub>i<sub>e</sub>ld<sub>s</sub> $\hat { q } ^ { ( 1 ) } ( z _ { n , 1 : P } ) = N ( \mu _ { n } , \Sigma _ { n } )$ , <sub>p</sub>o<sup>i</sup>nt estimates $\hat { \beta } , \hat { \beta } ^ { ( 0 ) }$ <sub>, an</sub>d th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>ors</sub> f<sub>or</sub> $W , \alpha , \tau$

Computing $T _ { \lambda } ^ { * }$ . From $\hat { \beta }$ and the em<sub>p</sub>irical co<sub>v</sub>ariate co<sub>v</sub>ariance $\hat { \Sigma } _ { Y }$ <sub>we</sub> f<sub>orm</sub> $\hat { \Sigma } _ { Z } = D _ { \hat { \beta } } \hat { \Sigma } _ { Y } D _ { \hat { \beta } } + D _ { 1 - \hat { \beta } ^ { 2 } }$ <sub>an</sub>d $\hat { \Sigma } _ { Z Y } = D _ { \hat { \cal B } } \hat { \Sigma } _ { Y } $ , an<sup>d</sup> <sub>p</sub><sup>l</sup>u<sub>g</sub> th<sub>em</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm express</sub>i<sub>ons o</sub>f S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> W<sub>e</sub> bl<sub>oc</sub>k <sub>ex</sub>t<sub>en</sub>d $T _ { \lambda } ^ { * }$ t<sub>o ac</sub>t <sub>as</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y on</sub> th<sub>e un</sub>i<sub>n</sub>f<sub>orme</sub>d f<sub>ac</sub>t<sub>ors, so</sub> $T \in$ $\mathbb { R } ^ { K \times K }$ <sub>w</sub>ith $T _ { 1 : P , 1 : P } = T _ { \lambda } ^ { * }$ <sub>an</sub>d $T _ { P + 1 : K , P + 1 : K } = I .$

Stage 2. We replace the variational means of the informed block <sub>w</sub>ith th<sub>e</sub>i<sub>r</sub> �<sub>-</sub>t<sub>rans</sub>f<sub>orme</sub>d <sub>vers</sub>i<sub>ons,</sub> ${ \tilde { \mu } } _ { n } \gets T _ { 1 : P } \mu _ { n }$ <sub>.</sub> W<sub>e</sub> th<sub>en</sub> h<sub>o</sub>ld ${ \tilde { \mu } } _ { n , 1 : P } , \hat { \beta } , \hat { \beta } ^ { ( 0 ) }$ <sub>,</sub>� fi<sub>xe</sub>d <sub>an</sub>d <sub>con</sub>ti<sub>nue coor</sub>di<sub>na</sub>t<sub>e-ascen</sub>t VI f<sub>or</sub> th<sub>e res</sub>t of variational <sub>p</sub>arameters until conver<sub>g</sub>ence.

Sta<sub>g</sub>e 2 is a relativel<sub>y</sub> chea<sub>p</sub> fine-tunin<sub>g</sub> ste<sub>p</sub> (standard FA with some factor moments fixed) that can be re<sub>p</sub>eated, for the same St<sub>age</sub> 1 fit<sub>,</sub> <sub>across</sub> <sub>many</sub> <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f �<sub>.</sub> It l<sub>e</sub>t<sub>s</sub> th<sub>e</sub> l<sub>oa</sub>di<sub>ngs</sub> � <sub>a</sub>d<sub>ap</sub>t t<sub>o</sub> th<sub>e new</sub> b<sub>as</sub>i<sub>s.</sub>

Alth<sub>oug</sub>h th<sub>e</sub> <sub>genera</sub>ti<sub>ve</sub> <sub>mo</sub>d<sub>e</sub>l <sub>assumes</sub> <sub>con</sub>ti<sub>nuous</sub> �<sub>,</sub> th<sub>e</sub> f<sub>rame-</sub> <sub>wor</sub>k <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> bi<sub>nary</sub> $y _ { p }$ b<sub>y</sub> re<sub>p</sub>lacin<sub>g</sub> correlation coeficient with <sub>p</sub>oint-biserial correlations.

## 4 Experimental setup

Our ex<sub>p</sub>eriments have two <sub>g</sub>oals: to em<sub>p</sub>iricall<sub>y</sub> show the theoretical <sub>proper</sub>ti<sub>es o</sub>f th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>spaces on s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons, an</sub>d t<sub>o</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e</sub> f<sub>ramewor</sub>k’<sub>s</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>app</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>as</sub> <sub>a</sub> <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on an</sub>d i<sub>ns</sub>id<sub>e</sub> iFA <sub>on s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons an</sub>d <sub>rea</sub>l<sub>-wor</sub>ld d<sub>a</sub>t<sub>a.</sub>

4.0.1 Artificial data. We designed four data-generation scenarios, followin<sub>g</sub> the <sub>g</sub>enerative <sub>p</sub>rocess of the model defined in E<sub>q</sub>. (9) (see Fi<sub>g</sub>. S6 for <sub>g</sub>ra<sub>p</sub>hical re<sub>p</sub>resentation) with $T \ = \ I ,$ <sub>eac</sub>h t<sub>ar-</sub> <sub>g</sub>etin<sub>g</sub> a diferent realistic covariance structure (Tab. S2) amon<sub>g</sub> the normall<sub>y</sub> distributed covariates: 1. mixed <sub>p</sub>ositive and ne<sub>g</sub>ative correlations (PN) (e.<sub>g</sub>. blood-<sub>p</sub>anel markers, where some <sub>p</sub>airs move to<sub>g</sub>ether and others in o<sub>pp</sub>osite directions), 2. autore<sub>g</sub>ressive structure (AR) (e.<sub>g</sub>. measurements taken consecutivel<sub>y</sub> over time), 3. <sub>p</sub>urel<sub>y</sub> <sub>p</sub>ositive correlations (P) (e.<sub>g</sub>. bone measurements that all scale with overall bod<sub>y</sub> size), and 4. <sub>p</sub>urel<sub>y</sub> ne<sub>g</sub>ative correlations (N) (e.<sub>g</sub>. dumm<sub>y</sub>-coded cate<sub>g</sub>ories to which <sub>p</sub>atients are assi<sub>g</sub>ned). Across all scenarios we held fixed the number of features (� = 100), th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> $( K = 1 0 )$ , the number of covariates (� = 5), the f<sub>ac</sub>t<sub>or-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>s</sub>t<sub>reng</sub>th<sub>s</sub> $\beta = ( 0 . 9 , 0 . 7 5 , 0 . 6 , 0 . 4 5 , 0 . 3 )$ and the noise-to-si<sub>g</sub>nal ratio (� = 0.1) (Tab. S3). We var<sub>y</sub> two <sub>p</sub>arameters: �, controllin<sub>g</sub> covariate correlation stren<sub>g</sub>th (from 0 to stron<sub>g</sub> de<sub>p</sub>endence), and �, scalin<sub>g</sub> the � vector (Tab. S4). In sam<sub>p</sub>le-based ex<sub>p</sub>eriments we used $N = 5 0 0$ obser<sub>v</sub>ations and 20 re<sub>p</sub>etitions.

4.0.2 Large model’s representations. As a testbed for post-hoc ali<sub>g</sub>nment on realistic latent s<sub>p</sub>ace<sub>,</sub> we use ima<sub>g</sub>e re<sub>p</sub>resentations from three lar<sub>g</sub>e <sub>p</sub>retrained models s<sub>p</sub>annin<sub>g</sub> the main <sub>p</sub>retrainin<sub>g</sub> <sub>p</sub>aradi<sub>g</sub>ms: CLIP [45] (contrastive ima<sub>g</sub>e-text), DINOv2 [40] (selfsu<sub>p</sub>ervised self-distillation), and ViT [17] (su<sub>p</sub>ervised classification on Ima<sub>g</sub>eNet). As in<sub>p</sub>ut data, we use a validation subset of the

![](images/688cb6d0e1d32f5c7fd66b85a89cd1556c4401716495e9ec8bda0e3a02de435a.jpg)  
Figure 2: Post-hoc latent space transformations of pretrained representations (CLIP, $\mathbf { D I N O v 2 } ,$ and ViT): dimensions with max corr (latent dimensions most strongly correlated with the covariates) and Procrustes orthogonal rotation against the proposed transformation applied after Procrustes rotation.

Tin<sub>y</sub>-Ima<sub>g</sub>eNet dataset [37], which is a downsam<sub>p</sub>led version of Ima<sub>g</sub>eNet [15], coverin<sub>g</sub> 200 classes with 50 validation ima<sub>g</sub>es <sub>p</sub>er <sub>c</sub>l<sub>ass.</sub> F<sub>rom</sub> thi<sub>s su</sub>b<sub>se</sub>t<sub>, we re</sub>t<sub>a</sub>i<sub>n</sub> 2000 <sub>o</sub>b<sub>serva</sub>ti<sub>ons an</sub>d <sub>use</sub> th<sub>e</sub> � = 20 dumm<sub>y</sub>-coded class indicators as covariates corres<sub>p</sub>ondin<sub>g</sub> to animal cate<sub>g</sub>ories (see Fi<sub>g</sub>. S7 for the exact labels). For each <sub>o</sub>b<sub>serva</sub>ti<sub>on we ex</sub>t<sub>rac</sub>t th<sub>e represen</sub>t<sub>a</sub>ti<sub>ons</sub> f<sub>rom eac</sub>h <sub>o</sub>f th<sub>e</sub> th<sub>ree</sub> <sub>pre</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub>

4.0.3 Real-world data. We used the breast TCGA dataset [10], pre-<sub>p</sub>rocessed b<sub>y</sub> [46], consistin<sub>g</sub> of multi-modal data (mRNA, miRNA, and <sub>p</sub>roteomics) from 150 breast cancer <sub>p</sub>atients cate<sub>g</sub>orized into cancer subt<sub>yp</sub>es (Basal, Her2, and LumA), which we used as dumm<sub>y</sub>- <sub>co</sub>d<sub>e</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es.</sub> W<sub>e use</sub>d 5<sub>-</sub>f<sub>o</sub>ld <sub>cross-va</sub>lid<sub>a</sub>ti<sub>on, ensur</sub>i<sub>ng a s</sub>t<sub>ra</sub>ti<sub>-</sub> fi<sub>e</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>cancer su</sub>bt<sub>ypes across</sub> f<sub>o</sub>ld<sub>s, an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>ze</sub>d th<sub>e</sub> d<sub>a</sub>t<sub>a ma</sub>t<sub>r</sub>i<sub>ces w</sub>ithi<sub>n eac</sub>h f<sub>o</sub>ld <sub>pr</sub>i<sub>or</sub> t<sub>o ana</sub>l<sub>ys</sub>i<sub>s.</sub>

4.0.4 Comparison methods. We compare against methods divided into t<sup>h</sup>ree categories. Unsupervised: PCA, w<sup>h</sup>ic<sup>h</sup> assumes <sup>f</sup>u<sup>ll l</sup>atent inde<sub>p</sub>endence, and MOFA [3], which uses a mean-field a<sub>pp</sub>roximation that factorizes the variational distribution over factors<sub>,</sub> a<sub>pp</sub>roximating rat<sup>h</sup>er t<sup>h</sup>an strict<sup>l</sup>y en<sup>f</sup>orcing t<sup>h</sup>eir independence. Supervised but not paired: two supervised extensions o<sup>f</sup> PCA, SPCA (Bair) [4] and SPCA (Barshan) [5], PLS, which extracts com<sub>p</sub>onents maximizin<sub>g</sub> the data-covariate covariance, and iVAE [31], a VAE model with covariate-conditioned prior. Supervised and paired: Y as factors t<sub>a</sub>k<sub>es</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>es</sub> di<sub>rec</sub>tl<sub>y as</sub> f<sub>ac</sub>t<sub>ors,</sub> $Z _ { p } = Y _ { p }$ <sub>,</sub> achievin<sub>g</sub> <sub>p</sub>erfect <sub>a</sub>li<sub>gnmen</sub>t b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>u</sub>t <sub>no</sub>t <sub>con</sub>t<sub>ro</sub>lli<sub>ng</sub> f<sub>or</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence,</sub> and SOFA [11], an extension of MOFA that models factor-covariate de<sub>p</sub>endencies usin<sub>g</sub> a scalin<sub>g</sub> factor (sf) - a wei<sub>g</sub>ht on the factorcovariate term in t<sup>h</sup>e o<sup>b</sup>jective t<sup>h</sup>at can up- or <sup>d</sup>own-weig<sup>h</sup>t it relative to the data-factor terms (default sf = 0.1).

F<sub>or</sub> <sub>no</sub>t <sub>pa</sub>i<sub>re</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>we</sub> fi<sub>rs</sub>t <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub> <sub>a</sub>ll <sub>pa</sub>i<sub>rs</sub> <sub>o</sub>f <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> <sub>an</sub>d l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors.</sub> W<sub>e</sub> th<sub>en</sub> <sub>so</sub>l<sub>ve</sub> <sub>a</sub> li<sub>near</sub> <sub>sum</sub> assi<sub>g</sub>nment <sub>p</sub>roblem [35] to ali<sub>g</sub>n each latent with the covariate th<sub>a</sub>t b<sub>es</sub>t <sub>corre</sub>l<sub>a</sub>t<sub>es w</sub>ith it<sub>.</sub>

4.0.5 Evaluation: performance metrics. We evaluate performance usin<sub>g</sub> the avera<sub>g</sub>e factor-covariate correlations<sub>,</sub> the s<sub>q</sub>uared Frobe-<sub>n</sub>i<sub>us</sub> di<sub>s</sub>t<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e covar</sub>i<sub>ance ma</sub>t<sub>r</sub>i<sub>x o</sub>f l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors</sub> $Z$ and the identit<sub>y</sub> matrix<sub>,</sub> and the avera<sub>g</sub>e variance ex<sub>p</sub>lained <sub>p</sub>er factor. We also re<sub>p</sub>ort four disentan<sub>g</sub>lement metrics (Section C.2): disentan<sub>g</sub>lement (�), measurin<sub>g</sub> whether each latent factor encodes information about onl<sub>y</sub> one covariate, com<sub>p</sub>leteness (�), whether <sub>eac</sub>h <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> i<sub>s cap</sub>t<sub>ure</sub>d b<sub>y on</sub>l<sub>y one</sub> l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>or,</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve-</sub> ness $( I ) ,$ w<sup>h</sup>et<sup>h</sup>er t<sup>h</sup>e <sup>f</sup>actors joint<sup>l</sup>y contain t<sup>h</sup>e in<sup>f</sup>ormation nee<sup>d</sup>e<sup>d</sup> to <sub>p</sub>redict the covariate [18], and se<sub>p</sub>arated attribute <sub>p</sub>redictabilit<sub>y</sub> (SAP), the <sub>g</sub>a<sub>p</sub> between the to<sub>p</sub> and second-best sin<sub>g</sub>le-latent <sub>p</sub>redictor of each covariate [36]. All four take values in [0, 1], with hi<sub>g</sub>her values indicatin<sub>g</sub> better disentan<sub>g</sub>lement. We further re<sub>p</sub>ort <sub>norma</sub>li<sub>se</sub>d SAP<sub>,</sub> <sub>a</sub> <sub>var</sub>i<sub>an</sub>t th<sub>a</sub>t <sub>measures</sub> th<sub>e</sub> <sub>sca</sub>l<sub>e</sub> <sub>o</sub>f d<sub>om</sub>i<sub>nance,</sub> th<sub>a</sub>t di<sub>v</sub>id<sub>es</sub> SAP b<sub>y</sub> th<sub>e</sub> t<sub>op</sub> <sub>p</sub>r<sub>e</sub>di<sub>c</sub>t<sub>o</sub>r’<sub>s</sub> <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce.</sub>

## 5 Results

5.0.1 Numerical validation of the theoretical results. For simulation scenarios (1)-(4) defined above, the o<sub>p</sub>timal transformations $T _ { \lambda } ^ { * }$ <sub>an</sub>d $T _ { \mathrm { e x } } ^ { \ast }$ <sub>can</sub> b<sub>e compu</sub>t<sub>e</sub>d <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> f<sub>rom</sub> th<sub>e covar</sub>i<sub>ances</sub> $\Sigma _ { Z }$ <sub>an</sub>d $\dot { \Sigma _ { Z Y } }$ (Fi<sub>g</sub>. 1). Under $T _ { 1 } ^ { * }$ th<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween pa</sub>i<sub>rs</sub> $( Z _ { p } , Y _ { p } )$ i<sub>s</sub> <sub>s</sub>t<sub>ronges</sub>t<sub>, an</sub>d <sub>wea</sub>k<sub>er un</sub>d<sub>er</sub> $T _ { 1 / 2 } ^ { * } , T _ { 0 } ^ { * }$ <sub>, an</sub>d $T _ { \mathrm { e x } } ^ { \ast } .$ <sub>, as s</sub>t<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> Th<sub>eo-</sub> rem 3.1. For $T _ { 0 } ^ { * }$ <sub>,</sub> th<sub>e</sub> <sub>re</sub>d<sub>uce</sub>d <sub>a</sub>li<sub>gnmen</sub>t i<sub>s</sub> <sub>a</sub> <sub>consequence</sub> <sub>o</sub>f <sub>en</sub>f<sub>orc</sub>i<sub>ng</sub> <sub>uncorre</sub>l<sub>a</sub>t<sub>e</sub>d l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors, a cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>a</sub>b<sub>sen</sub>t <sub>un</sub>d<sub>er</sub> $T _ { 1 } ^ { * }$ <sub>, w</sub>hi<sub>c</sub>h <sub>consequen</sub>tl<sub>y</sub> <sub>pro</sub>d<sub>uces</sub> <sub>corre</sub>l<sub>a</sub>t<sub>e</sub>d f<sub>ac</sub>t<sub>ors.</sub> A <sub>para</sub>ll<sub>e</sub>l <sub>argumen</sub>t <sub>ap</sub> <sub>p</sub>lies to $T _ { \mathrm { e x } } ^ { \ast } .$ <sub>,</sub> which im<sub>p</sub>oses exclusive one-to-one factor-covariate d<sub>epen</sub>d<sub>ence a</sub>t <sub>a</sub> f<sub>ur</sub>th<sub>er cos</sub>t i<sub>n a</sub>li<sub>gnmen</sub>t<sub>.</sub> Th<sub>e resu</sub>lt ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> a <sub>g</sub>eneral <sub>p</sub>rinci<sub>p</sub>le: constrainin<sub>g</sub> the latent structure reduces the <sub>ac</sub>hi<sub>eva</sub>bl<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>ons an</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es.</sub> Moreover<sub>,</sub> in contrast to $T _ { 1 } ^ { * }$ <sub>, un</sub>d<sub>er</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> i<sub>mpose</sub>d b<sub>y</sub> $T _ { \mathrm { e x } } ^ { \ast }$ or $T _ { 0 } ^ { * }$ <sub>,</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>no</sub> l<sub>onger</sub> <sub>resem</sub>bl<sub>es</sub> th<sub>e</sub> d<sub>epen-</sub> d<sub>ence s</sub>t<sub>ruc</sub>t<sub>ure among</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>es.</sub> Si<sub>m</sub>il<sub>ar conc</sub>l<sub>us</sub>i<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> across all scenarios (Fi<sub>g</sub>s. S8-S11, <sub>p</sub>anels A, B, and C).

5.0.2 Post-hoc application ofthe transformations . When pretrained <sub>represen</sub>t<sub>a</sub>ti<sub>ons are ava</sub>il<sub>a</sub>bl<sub>e,</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons can</sub> b<sub>e compu</sub>t<sub>e</sub>d from em<sub>p</sub>irical estimates $\hat { \Sigma } _ { Z }$ <sub>an</sub>d $\hat { \Sigma } _ { Z Y }$ <sub>an</sub>d <sub>app</sub>li<sub>e</sub>d <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> t<sub>o</sub> th<sub>e</sub> latent re<sub>p</sub>resentation (Fi<sub>g</sub>. 2). The <sub>p</sub>ro<sub>p</sub>osed transformations achieve <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>s</sub>t<sub>ronger</sub> <sub>one-</sub>t<sub>o-one</sub> d<sub>epen</sub>d<sub>ence</sub> b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t di<sub>-</sub> <sub>mens</sub>i<sub>ons an</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> th<sub>an</sub> th<sub>e na</sub>i<sub>ve</sub> b<sub>ase</sub>li<sub>ne o</sub>f <sub>se</sub>l<sub>ec</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>mos</sub>t <sub>corre</sub>l<sub>a</sub>t<sub>e</sub>d di<sub>mens</sub>i<sub>on o</sub>f th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>on, an</sub>d th<sub>ey</sub> em<sub>p</sub>iricall<sub>y</sub> re<sub>p</sub>roduce the covariate de<sub>p</sub>endence-stren<sub>g</sub>th orderin<sub>g</sub> of Theorem 3.1. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> a Procrustes rotation [20, 29] directl<sub>y</sub> t<sub>o</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ng y</sub>i<sub>e</sub>ld<sub>s corre</sub>l<sub>a</sub>t<sub>e</sub>d l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>ons,</sub> b<sub>e-</sub> <sub>cause</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> i<sub>s</sub> it<sub>se</sub>lf <sub>corre</sub>l<sub>a</sub>t<sub>e</sub>d<sub>,</sub> $T _ { 0 } ^ { * }$ <sub>,</sub> i<sub>n con</sub>t<sub>ras</sub>t<sub>,</sub> fi<sub>rs</sub>t <sub>w</sub>hit<sub>ens</sub> th<sub>e represen</sub>t<sub>a</sub>ti<sub>on an</sub>d th<sub>us pro</sub>d<sub>uces uncorre</sub>l<sub>a</sub>t<sub>e</sub>d di<sub>men</sub> <sub>s</sub>i<sub>ons.</sub> Th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $T _ { 1 } ^ { * }$ t<sub>ra</sub>d<sub>es some o</sub>f thi<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>er covar</sub>i<sub>a</sub>t<sub>e a</sub>li<sub>gnmen</sub>t<sub>,</sub> th<sub>oug</sub>h f<sub>or a</sub>ll th<sub>ree mo</sub>d<sub>e</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> th<sub>e resu</sub>lti<sub>ng</sub> di<sub>mens</sub>i<sub>ons rema</sub>i<sub>n</sub> l<sub>ess corre</sub>l<sub>a</sub>t<sub>e</sub>d th<sub>an un</sub>d<sub>er</sub> th<sub>e</sub> <sub>p</sub>lain Procrustes rotation. The covariance matrices (Fi<sub>g</sub>. S7) confi<sub>rm</sub> b<sub>o</sub>th th<sub>e</sub> <sub>wea</sub>k <sub>a</sub>li<sub>gnmen</sub>t <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>an</sub>d th<sub>a</sub>t th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t i<sub>mpose</sub>d b<sub>y</sub> $T _ { \mathrm { e x } }$ <sub>an</sub>d $T _ { 0 }$ <sub>are sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>.</sub> Downstream a<sub>pp</sub>lications that re<sub>q</sub>uire decodin<sub>g</sub> from the transf<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t<sub>,</sub> <sub>e.g.</sub> <sub>new</sub> <sub>samp</sub>l<sub>e</sub> <sub>genera</sub>ti<sub>on</sub> th<sub>roug</sub>h th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l d<sub>eco</sub>d<sub>er,</sub> <sub>wou</sub>ld <sub>app</sub>l<sub>y</sub> th<sub>e</sub> i<sub>nverse</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> t<sub>o</sub> <sub>map</sub> b<sub>ac</sub>k t<sub>o</sub> th<sub>e</sub> d<sub>eco</sub>d<sub>er</sub>’<sub>s</sub> <sub>expec</sub>t<sub>e</sub>d i<sub>npu</sub>t <sub>space.</sub>

5.0.3 The iFA model controls the trade-of. Having established the th<sub>eore</sub>ti<sub>ca</sub>l t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-</sub>di<sub>mens</sub>i<sub>on-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen-</sub> d<sub>ence an</sub>d l<sub>a</sub>t<sub>en</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ure, we now s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e same</sub> t<sub>ra</sub>d<sub>e-o</sub>f <sub>emerges w</sub>h<sub>en</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons are</sub> f<sub>o</sub>ld<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> iFA <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> i<sub>n</sub>f<sub>erence</sub> <sub>proce</sub>d<sub>ure.</sub>

![](images/f71bd4988d6573a5e868c078e9316307c9ad51e740e739a7ec276a8c5d708a46.jpg)  
Figure 3: Trade-of between covariate dependence and factor independence for simulation scenarios 1 (PN) and 4 (N). Outlined markers denote means over 20 repetitions.

A<sub>s</sub> <sub>a</sub> <sub>san</sub>it<sub>y</sub> <sub>c</sub>h<sub>ec</sub>k <sub>on</sub> th<sub>e</sub> i<sub>n</sub>f<sub>erence</sub> <sub>proce</sub>d<sub>ure,</sub> <sub>we</sub> <sub>ver</sub>if<sub>y</sub> th<sub>a</sub>t the <sub>p</sub>re-transformation variant (iFA no � ) conver<sub>g</sub>es to the true <sub>p</sub>arameters as the number of observations <sub>g</sub>rows: in Scenario 1<sub>,</sub> b<sub>o</sub>th th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d $\beta$ <sub>an</sub>d th<sub>e</sub> i<sub>n</sub>f<sub>orme</sub>d l<sub>a</sub>t<sub>en</sub>t f<sub>ac</sub>t<sub>ors</sub> <sub>approac</sub>h th<sub>e</sub>i<sub>r</sub> true values (Fi<sub>g</sub>. S16). Moreover, iFA no � recovers the assumed avera<sub>g</sub>e factor-covariate de<sub>p</sub>endence in ever<sub>y</sub> scenario: dia<sub>g</sub>onal <sub>corre</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>concen</sub>t<sub>ra</sub>t<sub>e</sub> <sub>aroun</sub>d ${ \overline { { \beta } } } ,$ reac<sup>hi</sup>n<sub>g</sub> a<sub>pp</sub>rox<sup>i</sup>mate<sup>l</sup><sub>y</sub> 0.6 at � = 1 and 0.4 at � = 0.6 (Fi<sub>g</sub>. 3; Fi<sub>g</sub>s. S8-S11, <sub>p</sub>anel D).

In Fi<sub>g</sub>. 3<sub>,</sub> an ideal method would achieve avera<sub>g</sub>e covariate cor-<sub>re</sub>l<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f 1 <sub>an</sub>d F<sub>ro</sub>b<sub>en</sub>i<sub>us</sub> <sub>norm</sub> $^ { 0 , }$ joint<sup>l</sup>y maximising covariate d<sub>epen</sub>d<sub>ence an</sub>d f<sub>ac</sub>t<sub>or</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub> Th<sub>eorem</sub> 3<sub>.</sub>1 <sub>s</sub>h<sub>ows</sub> thi<sub>s</sub> i<sub>s</sub> unattaina<sup>bl</sup>e in enera<sup>l</sup>, as t<sup>h</sup>e two o<sup>b</sup>jectives tra<sup>d</sup>e o<sup>f</sup>. T<sup>h</sup>e <sup>f</sup>ami<sup>l</sup> of the iFA models with the transformation a<sub>pp</sub>lied (iFA<sub>�</sub>) traces this t<sub>ra</sub>d<sub>e-o</sub>f<sub>:</sub> <sub>as</sub> � i<sub>ncreases,</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>grows</sub> <sub>an</sub>d f<sub>ac</sub>t<sub>or</sub> inde<sub>p</sub>endence weakens (Fi<sub>g</sub>. 3; Fi<sub>g</sub>s. S8–S11, <sub>p</sub>anel D). The <sub>g</sub>ain in covariate de<sub>p</sub>endence from movin<sub>g</sub> toward $T _ { 1 } ^ { * }$ i<sub>s</sub> l<sub>arges</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>es are s</sub>t<sub>rong</sub>l<sub>y</sub> i<sub>n</sub>t<sub>er-corre</sub>l<sub>a</sub>t<sub>e</sub>d<sub>, a</sub>t th<sub>e cos</sub>t <sub>o</sub>f f<sub>ac</sub>t<sub>or</sub> inde<sub>p</sub>endence (Fi<sub>g</sub>s. S12–S15, <sub>p</sub>anel $\mathrm { A } ) .$ <sub>.</sub> P<sub>a</sub>n<sub>e</sub>l B <sub>o</sub>f Fi<sub>gs.</sub> S12-S15 <sub>co</sub>nfi<sub>rms</sub> th<sub>e</sub> <sub>same</sub> <sub>pa</sub>tt<sub>ern</sub> <sub>across</sub> th<sub>e</sub> $\beta$ <sub>sweep:</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>o</sub>f $\mathrm { i F A } _ { \lambda }$ ri<sub>se</sub> from around 0.2 to 0.6 (which are the true <sub>g</sub>enerative values), and $T _ { 1 } ^ { * }$ consistentl<sub>y</sub> <sub>y</sub>ields stron<sub>g</sub>er covariate de<sub>p</sub>endence and stron<sub>g</sub>er f<sub>ac</sub>t<sub>or</sub> i<sub>n</sub>t<sub>er-</sub>d<sub>epen</sub>d<sub>ence</sub> th<sub>an</sub> $T _ { 0 } ^ { * }$ <sub>across a</sub>ll $\beta$ <sub>va</sub>l<sub>ues.</sub>

Variance ex<sub>p</sub>lained <sub>p</sub>er factor remains hi<sub>g</sub>h across all settin<sub>g</sub>s <sub>an</sub>d <sub>a</sub>ll <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f �<sub>,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>res</sub>h<sub>apes</sub> th<sub>e</sub> latent <sub>g</sub>eometr<sub>y</sub> without sacrificin<sub>g</sub> reconstruction or <sub>p</sub>er-factor inter<sub>p</sub>retabilit<sub>y</sub> of �.

![](images/f5256e7b26ce4d39d4a11b9fdb4d91c9baf1a1ce7c5e225761a16ee170d1f3d4.jpg)  
Figure 4: Disentanglement metrics for the latent space transformed using $T _ { \lambda }$ family and the exclusive transform $T _ { \mathrm { e x } }$ as a function of covariate-correlation strength � in simulation scenario 1 (PN).

Per-factor inter<sub>p</sub>retabilit<sub>y</sub> is critical for real-data a<sub>pp</sub>lications<sub>,</sub> <sub>w</sub>h<sub>ere</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l f<sub>ac</sub>t<sub>ors s</sub>h<sub>ou</sub>ld <sub>map on</sub>t<sub>o mean</sub>i<sub>ng</sub>f<sub>u</sub>l di<sub>rec</sub>ti<sub>ons</sub> in the observed features. We confirm this on real data (Fi<sub>g</sub>. 5): the <sub>var</sub>i<sub>ance exp</sub>l<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l f<sub>ac</sub>t<sub>ors rema</sub>i<sub>ns</sub> hi<sub>g</sub>h <sub>across</sub> th<sub>e</sub> entire $\mathrm { i F A } _ { \lambda }$ f<sub>am</sub>il<sub>y, w</sub>ith $T _ { 1 } ^ { * }$ ex<sub>p</sub>lainin<sub>g</sub> the most and $T _ { 0 } ^ { * }$ th<sub>e</sub> l<sub>eas</sub>t<sub>.</sub> Th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence an</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> i<sub>s preserve</sub>d th<sub>roug</sub>h<sub>ou</sub>t<sub>.</sub>

5.0.4 Disentanglement-metric view of the trade-of. For populationlevel transformations (Fi<sub>g</sub>. 4) and for the iFA<sub>�</sub> (Fi<sub>g</sub>s. S12-S15), disentan<sub>g</sub>lement (�) and com<sub>p</sub>leteness (�) are lower at small � and <sup>i</sup>m<sub>p</sub>rove as $\lambda$ <sub>grows, w</sub>ith <sub>mos</sub>t <sub>prom</sub>i<sub>nen</sub>t dif<sub>erence w</sub>h<sub>en</sub> th<sub>e</sub> <sub>cova</sub>ri<sub>a</sub>t<sub>e co</sub>rr<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s a</sub>r<sub>e s</sub>tr<sub>o</sub>n <sub>.</sub> Enf<sub>o</sub>r<sub>c</sub>in l<sub>a</sub>t<sub>e</sub>nt-dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n ind<sub>e</sub> <sub>p</sub>endence carries a $D / C$ <sub>cos</sub>t<sub>: eac</sub>h f<sub>ac</sub>t<sub>or</sub> h<sub>o</sub>ld<sub>s</sub> l<sub>ess covar</sub>i<sub>a</sub>t<sub>e-spec</sub>ifi<sub>c</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on, an</sub>d <sub>eac</sub>h <sub>covar</sub>i<sub>a</sub>t<sub>e</sub>’<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s sprea</sub>d <sub>across mu</sub>l ti<sub>p</sub>le factors. Informativeness (�) tracks si<sub>g</sub>nal stren<sub>g</sub>th instead - more correlated covariates (lar<sub>g</sub>er �) and stron<sub>g</sub>er factor-covariate de<sub>p</sub>endencies (lar<sub>g</sub>er �) both increase � re<sub>g</sub>ardless of �. SAP and its <sub>norma</sub>li<sub>se</sub>d <sub>var</sub>i<sub>an</sub>t<sub>,</sub> i<sub>n con</sub>t<sub>ras</sub>t<sub>,</sub> f<sub>avour</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons w</sub>ith <sub>sma</sub>ll �<sub>: w</sub>h<sub>en</sub> f<sub>ac</sub>t<sub>ors are cons</sub>t<sub>ra</sub>i<sub>ne</sub>d t<sub>o</sub> b<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>, eac</sub>h <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> i<sub>s</sub> <sub>p</sub>redicted most stron<sub>g</sub>l<sub>y</sub> b<sub>y</sub> a sin<sub>g</sub>le latent. The two variants disa<sub>g</sub>ree on $T _ { \mathrm { e x } } ^ { \ast } \mathrm { . }$ normalised SAP is hi<sub>g</sub>h as each factor <sub>g</sub>en<sub>u</sub>inel<sub>y</sub> carries in f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t <sub>on</sub>l<sub>y one covar</sub>i<sub>a</sub>t<sub>e,</sub> b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on, w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>unnorma</sub>li<sub>se</sub>d <sub>vers</sub>i<sub>on</sub> i<sub>s</sub> l<sub>ow as</sub> th<sub>e overa</sub>ll d<sub>epen</sub>d<sub>ence magn</sub>it<sub>u</sub>d<sub>e</sub> is small (Fi<sub>g</sub>. 4).

![](images/a374417a5a0208ff965ee0607ff691ed23892556eae3d33a4f62410b81e6ab79.jpg)  
Figure 5: Results on the breast TCGA multi-omics data. Outlined markers denote means over 20 repetitions.

5.0.5 iFA’s strengths over competing methods. Most competing methods fall behind the front traced b<sub>y</sub> the iFA<sub>�</sub> famil<sub>y</sub> (Fi<sub>g</sub>. 3, Fi<sub>g</sub>. 5, Fi<sub>g</sub>s. S8-S11). Some unsu<sub>p</sub>ervised and su<sub>p</sub>ervised baselines achieve <sub>s</sub>t<sub>rong</sub> f<sub>ac</sub>t<sub>or</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence a</sub>t th<sub>e cos</sub>t <sub>o</sub>f l<sub>ow covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence:</sub> MOFA, PCA, PLS, and SPCA (Bair), the last three of which <sub>p</sub>roduce inde<sub>p</sub>endent factors b<sub>y</sub> construction. The re<sub>g</sub>ression baseline (Y as factors) sits at the o<sub>pp</sub>osite extreme, attainin<sub>g</sub> maximal covariate d<sub>epen</sub>d<sub>ence</sub> b<sub>u</sub>t i<sub>n</sub>h<sub>er</sub>iti<sub>ng</sub> th<sub>e same</sub> f<sub>ac</sub>t<sub>or</sub> i<sub>n</sub>t<sub>er-</sub>d<sub>epen</sub>d<sub>ence as</sub> th<sub>e</sub> <sub>cova</sub>ri<sub>a</sub>t<sub>es</sub> th<sub>e</sub>m<sub>se</sub>l<sub>ves.</sub> SOFA i<sub>s</sub> th<sub>e c</sub>l<sub>oses</sub>t <sub>co</sub>m<sub>pe</sub>tit<sub>o</sub>r in <sub>s</sub>tr<sub>uc</sub>t<sub>u</sub>r<sub>e</sub>: it also <sub>p</sub>arameterises the stren<sub>g</sub>th of covariate de<sub>p</sub>endence<sub>,</sub> via a sca<sup>l</sup>ing <sup>f</sup>actor on t<sup>h</sup>e o<sup>b</sup>jective, <sup>h</sup>owever, t<sup>h</sup>is parameterisation gives <sub>on</sub>l<sub>y</sub> i<sub>n</sub>di<sub>rec</sub>t <sub>con</sub>t<sub>ro</sub>l<sub>, s</sub>i<sub>nce</sub> th<sub>e sca</sub>li<sub>ng</sub> f<sub>ac</sub>t<sub>or rewe</sub>i<sub>g</sub>ht<sub>s</sub> l<sub>oss</sub> t<sub>erms</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> Thi<sub>s</sub> l<sub>ac</sub>k <sub>o</sub>f <sub>s</sub>t<sub>eera</sub>bilit<sub>y</sub> causes SOFA to colla<sub>p</sub>se toward Y as factors at hi<sub>g</sub>h scalin<sub>g</sub> fact<sub>ors.</sub> Wh<sub>en</sub> th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> <sub>are</sub> th<sub>emse</sub>l<sub>ves</sub> <sub>s</sub>t<sub>rong</sub> f<sub>ac</sub>t<sub>ors</sub> <sub>o</sub>f <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> �<sub>,</sub> thi<sub>s</sub> <sub>co</sub>ll<sub>apse</sub> i<sub>s</sub> b<sub>en</sub>i<sub>gn,</sub> b<sub>u</sub>t <sub>w</sub>h<sub>en</sub> th<sub>ey</sub> <sub>are</sub> <sub>on</sub>l<sub>y</sub> <sub>wea</sub>kl<sub>y</sub> i<sub>n</sub>f<sub>or-</sub> mative<sub>,</sub> SOFA loses an<sub>y</sub> advanta<sub>g</sub>e over the re<sub>g</sub>ression baseline and its factors lose inter<sub>p</sub>retabilit<sub>y</sub> in terms of�. The lack of steerabilit<sub>y</sub> i<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> <sub>o</sub>n r<sub>ea</sub>l d<sub>a</sub>t<sub>a</sub> <sub>w</sub>h<sub>e</sub>r<sub>e</sub> SOFA <sub>eve</sub>n <sub>a</sub>t th<sub>e</sub> l<sub>a</sub>r<sub>ges</sub>t <sub>sca</sub>lin<sub>g</sub> f<sub>ac</sub>t<sub>or ac</sub>hi<sub>eves wea</sub>k<sub>er covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> th<sub>an</sub> iFA <sub>a</sub>t l<sub>arge</sub> � (Fi<sub>g</sub>. 5). Across disentan<sub>g</sub>lement metrics, SOFA in <sub>g</sub>eneral s<sub>p</sub>ans a b<sub>roa</sub>d<sub>er range o</sub>f <sub>scores</sub> th<sub>an</sub> $\mathrm { i F A } _ { \lambda }$ (Fi<sub>g</sub>s. S12-S15). When covariate <sub>co</sub>rr<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s a</sub>r<sub>e va</sub>ri<sub>e</sub>d<sub>,</sub> it i<sub>s</sub> $\mathrm { i F A } _ { \lambda }$ <sub>w</sub>h<sub>ose scores s</sub>hift <sub>sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y,</sub> while SOFA’s remain lar<sub>g</sub>el<sub>y</sub> unchan<sub>g</sub>ed.

$\mathrm { i F A } _ { \lambda }$ is slower than the classical baselines but faster than SOFA. O<sub>nce</sub> th<sub>e</sub> b<sub>ase mo</sub>d<sub>e</sub>l i<sub>s</sub> fitt<sub>e</sub>d<sub>, mu</sub>lti<sub>p</sub>l<sub>e</sub> $T _ { \lambda }$ t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons can</sub> b<sub>e</sub> a<sub>pp</sub>lied at ne<sub>g</sub>li<sub>g</sub>ible cost<sub>,</sub> makin<sub>g</sub> re<sub>g</sub>ime ex<sub>p</sub>loration chea<sub>p</sub>er than refittin<sub>g</sub> (Fi<sub>g</sub>. S17).

## 6 Conclusions

W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d <sub>a superv</sub>i<sub>se</sub>d f<sub>ramewor</sub>k f<sub>or</sub> di<sub>sen</sub>t<sub>ang</sub>l<sub>e</sub>d <sub>represen</sub>t<sub>a-</sub> ti<sub>on</sub> l<sub>earn</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>nav</sub>i<sub>ga</sub>t<sub>es</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>space.</sub> F<sub>our</sub> di<sub>s-</sub> entan<sub>g</sub>lement re<sub>g</sub>imes<sub>,</sub> inde<sub>p</sub>endent<sub>,</sub> intermediate<sub>,</sub> unconstrained<sub>,</sub> and exclusive<sub>,</sub> arise as solutions of o<sub>p</sub>timizations <sub>p</sub>roblems<sub>,</sub> and we <sub>prove</sub>d <sub>an</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> <sub>reg</sub>i<sub>mes</sub> b<sub>y</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> (Theorem 3.1). The resultin<sub>g</sub> transformations admit closed-form <sub>express</sub>i<sub>ons</sub> <sub>an</sub>d <sub>app</sub>l<sub>y</sub> b<sub>o</sub>th <sub>pos</sub>t<sub>-</sub>h<sub>oc</sub> t<sub>o</sub> <sub>pre</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> <sub>an</sub>d inside a new <sub>p</sub>robabilistic factor-anal<sub>y</sub>sis model (iFA). Ex<sub>p</sub>eriments <sub>con</sub>fi<sub>rm</sub> th<sub>a</sub>t iFA <sub>ena</sub>bl<sub>e</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>a</sub>bilit<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen-</sub> d<sub>ence</sub> <sub>an</sub>d l<sub>a</sub>t<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>recons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

## References

[1] Antonio Almudévar and Alfonso Orte<sub>g</sub>a. 2026. Rethinkin<sub>g</sub> Disentan<sub>g</sub>lement under Dependent Factors o<sup>f</sup>Variation. Transactions on Machine Learning Research (2026).

[2] Niccolo Anceschi, Federico Ferrari, David B. Dunson, and Himel Mallick. 2024. Ba<sub>y</sub>esian Joint Additive Factor Models for Multiview Learnin<sub>g</sub>. doi:10.48550/ arxiv.2406.00778

[3] Ricard Ar<sub>g</sub>ela<sub>g</sub>uet, Britta Velten, Damien Arnol, Sascha Dietrich, Thorsten Zenz, John C Marioni, Florian Buettner, Wolf<sub>g</sub>an<sub>g</sub> Huber, and Oliver Ste<sub>g</sub>le. 2018. M<sub>u</sub>lti-Omics Factor Anal<sub>y</sub>sis - a frame<sub>w</sub>ork for <sub>u</sub>ns<sub>up</sub>er<sub>v</sub>ised inte<sub>g</sub>ration of mu<sup>l</sup>ti-omics data sets. Molecular Systems Biology 14, 6 (2018), e8124. doi:10. 15252/msb.20178124

[4] Eric Bair, Trevor Hastie, Debashis Paul, and Robert Tibshirani. 2006. Prediction by Supervised Principa<sup>l</sup> Components. J. Amer. Statist. Assoc. 101, 473 (2006), 119–137. doi:10.1198/016214505000000628

[5] Elnaz Barshan, Ali Ghodsi, Zohreh Azimifar, and Mansoor Zol<sub>g</sub>hadri Jahromi. 2011. Su<sub>p</sub>ervised <sub>p</sub>rinci<sub>p</sub>al com<sub>p</sub>onent anal<sub>y</sub>sis: Visualization<sub>,</sub> classification and regression on subspaces and submani<sup>f</sup>o<sup>l</sup>ds. Pattern Recognition 44, 7 (2011), 1357–1371. <sup>d</sup>oi:10.1016/j. atco .2010.12.015

[6] Yoshua Ben<sub>g</sub>io, Aaron Courville, and Pascal Vincent. 2013. Re<sub>p</sub>resentation Learning: A Review and New Perspectives. IEEE Trans. Pattern Anal. Mach. Intell. 35, 8 (Au<sub>g</sub>. 2013), 1798–1828. doi:10.1109/TPAMI.2013.50

[7] David Blei, Al<sub>p</sub> Kucukelbir, and Jon McAulife. 2016. Variational Inference: A Review <sup>f</sup>or Statisticians. J. Amer. Statist. Assoc. 112 (01 2016). doi:10.1080 01621459.2017.1285773

[8] Diane Bouchacourt, R<sub>y</sub>ota Tomioka, and Sebastian Nowozin. 2018. Multi-Level V<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l A<sub>u</sub>t<sub>oe</sub>n<sub>co</sub>d<sub>e</sub>r: L<sub>ea</sub>rnin<sub>g</sub> Di<sub>se</sub>nt<sub>a</sub>n<sub>g</sub>l<sub>e</sub>d R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> Fr<sub>o</sub>m Gr<sub>oupe</sub>d Observations. Proceedings ofthe AAAI Conference on Artificial Intelligence 32, 1 (A<sub>p</sub>ril 2018). doi:10.1609/aaai.v32i1.11867

[9] Philemon Brakel and Yoshua Ben<sub>g</sub>io. 2017. Learnin<sub>g</sub> Inde<sub>p</sub>endent Features with Ad<sub>ve</sub>r<sub>sa</sub>ri<sub>a</sub>l N<sub>e</sub>t<sub>s</sub> f<sub>o</sub>r N<sub>o</sub>n-lin<sub>ea</sub>r ICA<sub>. a</sub>rXi<sub>v</sub>:<sub>a</sub>rXi<sub>v</sub>:1710<sub>.</sub>05050

[10] Cancer Genome Atlas Network. 2012. Com<sub>p</sub>rehensive molecular <sub>p</sub>ortraits of <sup>h</sup>uman breast tumours. Nature 490, 7418 (Oct. 2012), 61–70.

[11] Tüma<sub>y</sub> Ca<sub>p</sub>raz, Harald Vöhrin<sub>g</sub>er, Klaus Sebastian Au<sub>g</sub>usto Kru<sub>g</sub>er Serrano, Ricardo Omar Ramirez Flores, Julio Saez-Rodri<sub>g</sub>uez, and Wolf<sub>g</sub>an<sub>g</sub> Huber. 2025. Semi-su<sub>p</sub>ervised Omics Factor Anal<sub>y</sub>sis (SOFA) disentan<sub>g</sub>les known and latent sources o<sup>f</sup> variation in mu<sup>l</sup>ti-omic data. bioRxiv (2025). doi:10.1101/2024.10.10. 617527

[12] Rick<sub>y</sub> T. Q. Chen, Xuechen Li, Ro<sub>g</sub>er Grosse, and David Duvenaud. 2018. Isolatin<sub>g</sub> Sources o<sup>f</sup> Disentang<sup>l</sup>ement in Variationa<sup>l</sup> Autoencoders. In Advances in Neural Information Processing Systems.

[13] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Il<sub>y</sub>a Sutskever, and Pieter Abbeel. 2016. InfoGAN: inter<sub>p</sub>retable re<sub>p</sub>resentation learnin<sub>g</sub> b<sub>y</sub> information maximizing generative adversaria<sup>l</sup> nets. In Proceedings ofthe 30th International Conference on Neural Information Processing Systems (Barce<sup>l</sup>ona, Spain) (NIPS’16). C<sub>u</sub>rr<sub>a</sub>n A<sub>ssoc</sub>i<sub>a</sub>t<sub>es</sub> In<sub>c.,</sub> R<sub>e</sub>d H<sub>oo</sub>k<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 2180–2188<sub>.</sub>

[14] Zhi Chen, Yijie Bei, and C<sub>y</sub>nthia Rudin. 2020. Conce<sub>p</sub>t whitenin<sub>g</sub> for inter<sub>p</sub>retable image recognition. Nat. Mach. Intell. 2, 12 (Dec. 2020), 772–782.

[15] Jia Den , Wei Don , Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Ima geNet: A <sup>l</sup>arge-sca<sup>l</sup>e <sup>h</sup>ierarc<sup>h</sup>ica<sup>l</sup> image database. In 2009IEEE Conference on Computer Vision and Pattern Recognition. 248–255. doi:10.1109/CVPR.2009.5206848

[16] Andrea Dittadi, Frederik Träuble, Francesco Locatello, Manuel Wuthrich, Vaibhav A<sub>grawa</sub>l<sub>,</sub> Ol<sub>e</sub> Wi<sub>n</sub>th<sub>er,</sub> St<sub>e</sub>f<sub>an</sub> B<sub>auer, an</sub>d B<sub>ern</sub>h<sub>ar</sub>d S<sub>c</sub>hölk<sub>op</sub>f<sub>.</sub> 2021<sub>.</sub> O<sub>n</sub> th<sub>e</sub> Trans<sup>f</sup>er o<sup>f</sup> Disentang<sup>l</sup>ed Representations in Rea<sup>l</sup>istic Settings. In International Conference on Learning Representations.

[17] Alexe<sub>y</sub> Dosovitski<sub>y</sub>, Lucas Be<sub>y</sub>er, Alexander Kolesnikov, Dirk Weissenborn, Xi-<sub>ao</sub>h<sub>ua</sub> Zh<sub>a</sub>i<sub>,</sub> Th<sub>o</sub>m<sub>as</sub> Unt<sub>e</sub>rthin<sub>e</sub>r<sub>,</sub> M<sub>os</sub>t<sub>a</sub>f<sub>a</sub> D<sub>e</sub>h<sub>g</sub>h<sub>a</sub>ni<sub>,</sub> M<sub>a</sub>tthi<sub>as</sub> Mind<sub>e</sub>r<sub>e</sub>r<sub>,</sub> G<sub>eo</sub>r<sub>g</sub> Hei<sub>g</sub>old, S<sub>y</sub>lvain Gell<sub>y</sub>, Jakob Uszkoreit, and Neil Houlsb<sub>y</sub>. 2021. An Ima<sub>g</sub>e is Wort<sup>h</sup> 16x16 Words: Trans<sup>f</sup>ormers <sup>f</sup>or Image Recognition at Sca<sup>l</sup>e. In International Conference on Learning Representations.

[18] Cian Eastwood and Christo<sub>p</sub>her K. I. Williams. 2018. A framework for the <sub>q</sub>uantitative eva<sup>l</sup>uation o<sup>f</sup> disentang<sup>l</sup>ed representations. In International Conference on Learning Representations.

[19] Christina M Funke, Paul Vicol, Kuan-Chieh Wan<sub>g</sub>, Matthias Kuemmerer, Richard Zemel<sub>,</sub> and Matthias Beth<sub>g</sub>e. 2022. Disentan<sub>g</sub>lement and Generalization Under Corre<sup>l</sup>ation S<sup>h</sup>i<sup>f</sup>ts. In ICLR2022 Workshop on the Elements ofReasoning: Objects, Structure and Causality.

[20] Gene H Go<sup>l</sup>ub and C<sup>h</sup>ar<sup>l</sup>es F Van Loan. 2013. Matrix Computations (4 ed.). Jo<sup>h</sup>ns H<sub>op</sub>kin<sub>s</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> Pr<sub>ess,</sub> B<sub>a</sub>ltim<sub>o</sub>r<sub>e,</sub> MD<sub>.</sub>

[21] Jerem<sub>y</sub> P G<sub>yg</sub>i, Anna Konstorum, Shrikant Pawar, Edel Aron, Steven H Kleinstein, and Le<sub>y</sub>in<sub>g</sub> Guan. 2024. A su<sub>p</sub>ervised Ba<sub>y</sub>esian factor model for the identification o<sup>f</sup> mu<sup>l</sup>ti-omics signatures. Bioinformatics 40, 5 (May 2024). doi:10.1101/2023.01. 25.525545

[22] Naama Hadad, Lior Wolf, and Moni Shahar. 2018. A Two-Ste<sub>p</sub> Disentan<sub>g</sub>lement Met<sup>h</sup>od. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[23] Irina Hi<sub>gg</sub>ins, Loic Matthe<sub>y</sub>, Arka Pal, Christo<sub>p</sub>her Bur<sub>g</sub>ess, Xavier Glorot, M<sub>a</sub>tth<sub>ew</sub> B<sub>o</sub>t<sub>v</sub>i<sub>n</sub>i<sub>c</sub>k<sub>,</sub> Sh<sub>a</sub>ki<sub>r</sub> M<sub>o</sub>h<sub>ame</sub>d<sub>, an</sub>d Al<sub>exan</sub>d<sub>er</sub> L<sub>erc</sub>h<sub>ner.</sub> 2017<sub>.</sub> b<sub>e</sub>t<sub>a-</sub> VAE: L<sub>ea</sub>rnin<sub>g</sub> B<sub>as</sub>i<sub>c</sub> Vi<sub>sua</sub>l C<sub>o</sub>n<sub>cep</sub>t<sub>s w</sub>ith <sub>a</sub> C<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d V<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Fr<sub>a</sub>m<sub>ewo</sub>rk<sub>.</sub> In International Conference on Learning Representations.

[24] Nicholas J. Hi<sub>g</sub>ham. 1986. Com<sub>p</sub>utin<sub>g</sub> the Polar Decom<sub>p</sub>osition—with A<sub>pp</sub>lications. SIAM J. Sci. Stat. Comput. 7, 4 (Oct. 1986), 1160–1174.

[25] H Hotellin<sub>g</sub>. 1933. Anal<sub>y</sub>sis of a com<sub>p</sub>lex of statistical variables into <sub>p</sub>rinci<sub>p</sub>al components. J. Educ. Psychol. 24, 6 (Sept. 1933), 417–441. doi:10.1037/<sup>h</sup>0071325

[26] Aa<sub>p</sub>o H<sub>y</sub>värinen, Hiroaki Sasaki, and Richard E. Turner. 2018. Nonlinear ICA Using Auxi<sup>l</sup>iary Variab<sup>l</sup>es and Genera<sup>l</sup>ized Contrastive Learning. ArXiv abs/1805.08651 (2018).

[27] A. H<sub>y</sub>värinen and E. Oja. 2000. Inde<sub>p</sub>endent com<sub>p</sub>onent anal<sub>y</sub>sis: al<sub>g</sub>orithms and app<sup>l</sup>ications. Neural Networks 13, 4 (2000), 411–430. doi:10.1016/S0893- 6080(00)00026-5

[28] A<sub>y</sub>a Abdelsalam Ismail, Julius Adeba<sub>y</sub>o, Hector Corrada Bravo, Ste<sub>p</sub>hen Ra, and Kyung<sup>h</sup>yun C<sup>h</sup>o. 2024. Concept Bott<sup>l</sup>enec<sup>k</sup> Generative Mode<sup>l</sup>s. In The Twelfth International Conference on Learning Representations.

[29] Wolf<sub>g</sub>an<sub>g</sub> Kabsch. 1976. A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A 32 (1976), 922–923.

[30] A<sub>g</sub>nan Kess<sub>y</sub>, Alex Lewin, and Korbinian Strimmer. 2018. O<sub>p</sub>timal Whitenin<sub>g</sub> and Decorre<sup>l</sup>ation. The American Statistician 72, 4 (Jan. 2018), 309–314. doi:10. 1080/00031305.2016.1277159

[31] Il<sub>y</sub>es Khemakhem, Diederik P. Kin<sub>g</sub>ma, and Aa<sub>p</sub>o H<sub>y</sub>värinen. 2019. Variational Autoencoders and Non<sup>l</sup>inear ICA: A Uni<sup>f</sup> in Framewor<sup>k</sup>. In International Conference on Artificial Intelligence and Statistics.

[32] Hyunji<sup>k</sup> Kim and Andriy Mni<sup>h</sup>. 2018. Disentang<sup>l</sup>ing by Factorising. In Proceedings ofthe 35th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 80), Jenni<sup>f</sup>er D and Andreas Krause (Eds.). PMLR, 2649– 2658.

[33] Arto Klami, Se<sub>pp</sub>o Virtanen, Eemeli Le<sub>pp</sub>äaho, and Samuel Kaski. 2014. Grou<sub>p</sub> Factor Ana<sup>l</sup>ysis. IEEE transactions on neural networks and learning systems 26 (11 2014). doi:10.1109/TNNLS.2014.2376974

[34] Jack Kl<sub>y</sub>s, Jake Snell, and Richard Zemel. 2018. Learnin<sub>g</sub> latent subs<sub>p</sub>aces in variationa<sup>l</sup> autoencoders. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (Montréa<sup>l</sup>, Canada) (NIPS’18). Curran A<sub>ssoc</sub>i<sub>a</sub>t<sub>es</sub> In<sub>c.,</sub> R<sub>e</sub>d H<sub>oo</sub>k<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 6445–6455<sub>.</sub>

[35] Harold W. Kuhn. 1955. The Hun<sub>g</sub>arian method for the assi<sub>g</sub>nment <sub>p</sub>roblem. Naval Research Logistics (NRL) 52 (1955).

[36] Abhishek Kumar, Prasanna Satti<sub>g</sub>eri, and Avinash Balakrishnan. 2018. Variational Inference of Disentan<sub>g</sub>led Latent Conce<sub>p</sub>ts from Unlabeled Observations. In International Conference on Learning Representations.

[37] Ya Le and Xuan S. Yan<sub>g</sub>. 2015. Tin<sub>y</sub> Ima<sub>g</sub>eNet Visual Reco<sub>g</sub>nition Challen<sub>g</sub>e.

[38] Francesco Locatello, Stefan Bauer, Mario Lučić, Gunnar Rätsch, S<sub>y</sub>lvain Gell<sub>y</sub>, Bernhard Schölko<sub>p</sub>f<sub>,</sub> and Olivier Frederic Bachem. 2019. Challen<sub>g</sub>in<sub>g</sub> Common A<sub>ssu</sub>m<sub>p</sub>ti<sub>o</sub>n<sub>s</sub> in th<sub>e</sub> Un<sub>supe</sub>r<sub>v</sub>i<sub>se</sub>d L<sub>ea</sub>rnin<sub>g o</sub>f Di<sub>se</sub>nt<sub>a</sub>n<sub>g</sub>l<sub>e</sub>d R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> In International Conference on Machine Learning. Best Paper Award.

[39] Francesco Locatello, Ben Poole, Gunnar Rätsch, Bernhard Schölko<sub>p</sub>f, Olivier B<sub>ac</sub>h<sub>em, an</sub>d Mi<sub>c</sub>h<sub>ae</sub>l T<sub>sc</sub>h<sub>annen.</sub> 2020<sub>.</sub> W<sub>ea</sub>kl<sub>y-superv</sub>i<sub>se</sub>d di<sub>sen</sub>t<sub>ang</sub>l<sub>emen</sub>t wit<sup>h</sup>out compromises. In Proceedings of the 37th International Conference on Machine Learning (ICML’20). JMLR.org, Artic<sup>l</sup>e 589, 12 pages.

[40] Maxime O<sub>q</sub>uab, Timothée Darcet, Théo Moutakanni, Hu<sub>y</sub> V. Vo, Marc Szafraniec, V<sub>as</sub>il Kh<sub>a</sub>lid<sub>ov,</sub> Pi<sub>e</sub>rr<sub>e</sub> F<sub>e</sub>rn<sub>a</sub>nd<sub>e</sub>z<sub>,</sub> D<sub>a</sub>ni<sub>e</sub>l H<sub>a</sub>ziz<sub>a,</sub> Fr<sub>a</sub>n<sub>c</sub>i<sub>sco</sub> M<sub>assa,</sub> Al<sub>aae</sub>ldin E<sup>l</sup>-Nou<sup>b</sup>y, Mi<sup>d</sup>o Assran, Nico<sup>l</sup>as Ba<sup>ll</sup>as, Wojciec<sup>h</sup> Ga<sup>l</sup>u<sup>b</sup>a, Russe<sup>ll</sup> Howes, Po-Yao H<sub>u</sub>an<sub>g,</sub> Shan<sub>g</sub>-Wen Li<sub>,</sub> Ishan Misra<sub>,</sub> Michael Rabbat<sub>,</sub> Vas<sub>u</sub> Sharma<sub>,</sub> Gabriel S<sub>y</sub>nnaeve, Hu Xu, Herve Je<sub>g</sub>ou, Julien Mairal, Patrick Labatut, Armand Joulin, an<sup>d</sup> Piotr Bojanows<sup>k</sup>i. 2024. DINOv2: Learning Ro<sup>b</sup>ust Visua<sup>l</sup> Features wit<sup>h</sup>out Supervision. Transactions on Machine Learning Research (2024). Featured C<sub>e</sub>rtifi<sub>ca</sub>ti<sub>o</sub>n<sub>.</sub>

[41] Elise F Palzer, Christine H Wendt, Russell P Bowler, Crai<sub>g</sub> P Hersh, Sandra E Safo, and Eric F Lock. 2022. sJIVE: Su ervised joint and individual variation exp<sup>l</sup>ained. Comput. Stat. Data Anal. 175, 107547 (Nov. 2022), 107547. doi:10.1016/ j.cs<sup>d</sup>a.2022.107547

[42] Karl Pearson. 1901. LIII. On lines and <sub>p</sub>lanes of closest fit to s<sub>y</sub>stems of <sub>p</sub>oints in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2, 11 (1901), 559–572. doi:10.1080/14786440109462720

[43] F. Pedre<sub>g</sub>osa, G. Varo<sub>q</sub>uaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubour<sub>g</sub>, J. Vander<sub>p</sub>las, A. Passos, D. Courn<sub>apeau,</sub> M<sub>.</sub> Br<sub>uc</sub>h<sub>e</sub>r<sub>,</sub> M<sub>.</sub> P<sub>e</sub>rr<sub>o</sub>t<sub>, a</sub>nd E<sub>.</sub> D<sub>uc</sub>h<sub>es</sub>n<sub>ay.</sub> 2011<sub>.</sub> S<sub>c</sub>ikit-l<sub>ea</sub>rn: M<sub>ac</sub>hin<sub>e</sub> Learning in Pyt<sup>h</sup>on. Journal ofMachine Learning Research 12 (2011), 2825–2830

[44] Arber Qoku and Florian Buettner. 2023. Encodin<sub>g</sub> Domain Knowled<sub>g</sub>e in Multi-<sub>v</sub>ie<sub>w</sub> Latent Variable Models: A Ba<sub>y</sub>esian A<sub>pp</sub>roach <sub>w</sub>ith Str<sub>u</sub>ct<sub>u</sub>red S<sub>p</sub>arsit<sub>y</sub>. In Proceedings ofThe 26th International Conference on Artificial Intelligence and Statistics (Proceedings ofMachine Learning Research, Vol. 206), Francisco Ruiz, Jennifer D<sub>y</sub>, and Jan-Willem van de Meent (Eds.). PMLR, 11545–11562.

[45] Alec Radford, Jon<sub>g</sub> Wook Kim, Chris Hallac<sub>y</sub>, Adit<sub>y</sub>a Ramesh, Gabriel Goh, Sandhini A<sub>g</sub>arwal, Girish Sastr<sub>y</sub>, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krue<sub>g</sub>er<sub>,</sub> and Il<sub>y</sub>a Sutskever. 2021. Learnin<sub>g</sub> Transferable Visual Models From Natura<sup>l</sup> Language Supervision. ArXiv abs/2103.00020 (2021).

[46] Florian Rohart, Benoît Gautier, Amrit Sin<sub>g</sub>h, and Kim-Anh Lê Cao. 2017. <sub>m</sub>i<sub>x</sub>O<sub>m</sub>i<sub>cs:</sub> A<sub>n</sub> R <sub>pac</sub>k<sub>age</sub> f<sub>or</sub> ‘<sub>om</sub>i<sub>cs</sub> f<sub>ea</sub>t<sub>ure se</sub>l<sub>ec</sub>ti<sub>on an</sub>d <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> d<sub>a</sub>t<sub>a</sub> i<sub>n</sub>t<sub>egra-</sub> tion. PLOS Computational Biology 13, 11 (11 2017), 1–19. doi:10.1371/journa<sup>l</sup>. <sub>p</sub>cbi.1005752

[47] Karsten Roth, Mark Ibrahim, Ze<sub>y</sub>ne<sub>p</sub> Akata, Pascal Vincent, and Diane Bouchaco<sub>u</sub>rt. 2023. Disentan<sub>g</sub>lement of Correlated Factors <sub>v</sub>ia Ha<sub>u</sub>sdorf Factorized Support. In The Eleventh International Conference on Learning Representations.

[48] Sarah Samorodnitsk<sub>y</sub>, Chris H. Wendt, and Eric F. Lock. 2024. Ba<sub>y</sub>esian simultaneous <sup>f</sup>actorization and prediction using mu<sup>l</sup>ti-omic data. Computational Statistics & Data Analysis 197 (Sept. 2024), 107974. doi:10.1016/j.csda.2024.107974

[49] Peter H. Schönemann. 1966. A Generalized Solution ofthe Ortho<sub>g</sub>onal Procrustes Prob<sup>l</sup>em. Psychometrika 31, 1 (Marc<sup>h</sup> 1966), 1–10. doi:10.1007/b<sup>f</sup>02289451

[50] Kih<sub>y</sub>uk Sohn, Hon<sub>g</sub>lak Lee, and Xinchen Yan. 2015. Learnin<sub>g</sub> Structured Out<sub>p</sub>ut Representation using Deep Conditiona<sup>l</sup> Generative Mode<sup>l</sup>s. In Neural Information Processing Systems.

[51] L. L. T<sup>h</sup>urstone. 1931. Mu<sup>l</sup>tip<sup>l</sup>e Factor Ana<sup>l</sup>ysis. Psychological Review 38, 5 (1931), 406<sub>–</sub>427<sub>.</sub> d<sub>o</sub>i<sub>:</sub>10<sub>.</sub>1037/h0069792

[52] Frederik Träuble, Elliot Crea<sub>g</sub>er, Niki Kilbertus, Francesco Locatello, Andrea Dittadi<sub>,</sub> Anir<sub>u</sub>dh Go<sub>y</sub>al<sub>,</sub> Bernhard Schölko<sub>p</sub>f<sub>,</sub> and Stefan Ba<sub>u</sub>er. 2021. On Disentang<sup>l</sup>ed Representations Learned <sup>f</sup>rom Corre<sup>l</sup>ated Data. In Proceedings ofthe 38th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 139), Marina Mei<sup>l</sup>a and Tong Z<sup>h</sup>ang (Eds.). PMLR, 10401–10412.

[53] Martin J. Wainwri<sub>g</sub>ht and Michael I. Jordan. 2008. Gra<sub>p</sub>hical Models, Ex<sub>p</sub>onential Fami<sup>l</sup>ies, and Variationa<sup>l</sup> In<sup>f</sup>erence. Foundations and Trends® in Machine Learning 1, 1–2 (2008), 1–305. doi:10.1561/2200000001

[54] Xin Wan<sub>g</sub>, Hon<sub>g</sub> Chen, Si’ao Tan<sub>g</sub>, Zihao Wu, and Wenwu Zhu. 2024. Disentang<sup>l</sup>ed Representation Learning. IEEE Trans. Pattern Anal. Mach. Intell. 46, 12 (Dec. 2024), 9677–9696. doi:10.1109/TPAMI.2024.3420937

[55] Herman Wold. 1975. Soft Modellin<sub>g</sub> b<sub>y</sub> Latent Variables: The Non-Linear Iterative Partia<sup>l</sup> Least Squares (NIPALS) Approac<sup>h</sup>. Journal of Applied Probability 12, S1 (1975), 117–142. doi:10.1017/S0021900200047604

[56] Rui<sub>y</sub>u Zhan<sub>g</sub>, Ce Zhao, Xin Zhao, Lin Nie, and Wai-Fun<sub>g</sub> Lam. 2025. Structural E<sub>q</sub>uation-VAE: Disentan<sub>g</sub>led Latent Re<sub>p</sub>resentations for Tabular Data. (2025). doi:10.2139/ssrn.5384208

[57] Shiwen Zhao, Chuan Gao, Sa<sub>y</sub>an Mukherjee, and Barbara E En<sub>g</sub>elhardt. 2016. Bayesian group <sup>f</sup>actor ana<sup>l</sup>ysis wit<sup>h</sup> structured sparsity. Journal of Machine Learning Research 17, 196 (2016), 1–47.

[58] Yifan Zhou, Kaixuan Luo, Lifan Lian<sub>g</sub>, Men<sub>g</sub>jie Chen, and Xin He. 2023. A new Ba<sub>y</sub>esian factor anal<sub>y</sub>sis method im<sub>p</sub>roves detection of <sub>g</sub>enes and biolo<sub>g</sub>ical processes a<sup>f</sup>ected by perturbations in sing<sup>l</sup>e-ce<sup>ll</sup> CRISPR screening. Nature Methods 20, 11 (Sept. 2023), 1693–1703. doi:10.1038/s41592-023-02017-4

## A Proofs and derivations

## A.1 Reformulation of an orthogonal Procrustes problem [20]

Let $Z \in \mathbb { R } ^ { P }$ <sub>an</sub>d $Y \in \mathbb { R } ^ { P }$ b<sub>e</sub> b<sub>o</sub>th <sub>cen</sub>t<sub>ere</sub>d <sub>so</sub> th<sub>a</sub>t $\mathbb { E } [ Z ] = \mathbb { E } [ Y ] =$ 0. Let ${ \overline { { Z } } } : = \Sigma _ { Z } ^ { - 1 / 2 } Z$ d<sub>eno</sub>t<sub>e</sub> th<sub>e w</sub>hit<sub>ene</sub>d <sub>vec</sub>t<sub>or, w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\operatorname { C o v } ( { \overline { { Z } } } ) = \Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z } \Sigma _ { Z } ^ { - 1 / 2 } = I .$ Th<sub>e or</sub>th<sub>ogona</sub>l P<sub>rocrus</sub>t<sub>es pro</sub>bl<sub>em</sub> in <sub>p</sub>o<sub>pu</sub>lation terms is

$$
\operatorname* { m i n } _ { Q ^ { \prime } Q = I } \mathbb { E } { \left\| { Y } - Q \Sigma _ { Z } ^ { - 1 / 2 } Z \right\| } ^ { 2 } .
$$

Ex<sub>p</sub>andin<sub>g</sub> the s<sub>q</sub>uared norm

$$
\mathbb { E } { \left\| { Y - Q \overline { { Z } } } \right\| } ^ { 2 } = \mathbb { E } { \left\| { Y } \right\| } ^ { 2 } ~ - ~ 2 \mathbb { E } { \left[ { Y ^ { \prime } Q \overline { { Z } } } \right] } ~ + ~ \mathbb { E } { \left[ { \overline { { Z } } ^ { \prime } Q ^ { \prime } Q \overline { { Z } } } \right] } .
$$

Th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>s cons</sub>t<sub>an</sub>t i<sub>n</sub> $Q ,$ <sub>, an</sub>d th<sub>e</sub> l<sub>as</sub>t t<sub>erm equa</sub>l<sub>s</sub> $\mathbb { E } [ \overline { { Z } } ^ { \prime } \overline { { Z } } ] =$ $\operatorname { t r } ( I ) = P$ <sub>s</sub>in<sub>ce</sub> $Q ^ { \prime } Q = I ,$ h<sub>e</sub>n<sub>ce</sub> i<sub>s a</sub>l<sub>so co</sub>n<sub>s</sub>t<sub>a</sub>nt<sub>.</sub> Minimizin<sub>g</sub> th<sub>e</sub> o<sup>b</sup>jective is t<sup>h</sup>ere<sup>f</sup>ore equiva<sup>l</sup>ent to maximizing t<sup>h</sup>e cross term. Usin<sub>g</sub> $\mathbb { E } [ Z Y ^ { \prime } ] = \Sigma _ { Z Y }$ <sub>an</sub>d th<sub>e cyc</sub>li<sub>c</sub> i<sub>nvar</sub>i<sub>ance o</sub>f th<sub>e</sub> t<sub>race,</sub>

$$
\mathbb { E } \bigl [ Y ^ { \prime } Q \Sigma _ { Z } ^ { - 1 / 2 } Z \bigr ] = \mathrm { t r } \bigl ( Q \Sigma _ { Z } ^ { - 1 / 2 } \mathbb { E } [ Z Y ^ { \prime } ] \bigr ) = \mathrm { t r } \bigl ( Q \Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z Y } \bigr ) .
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
\operatorname* { m i n } _ { Q ^ { \prime } Q = I } \mathbb { E } \bigl \| Y - Q \Sigma _ { Z } ^ { - 1 / 2 } Z \bigr \| ^ { 2 } \quad \Leftrightarrow \quad \operatorname* { m a x } _ { Q ^ { \prime } Q = I } \mathrm { t r } \bigl ( Q \Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z Y } \bigr ) ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em we</sub> d<sub>e</sub>fi<sub>ne</sub>d<sub>.</sub>

## A.2 Closed-form solution of $T _ { 0 } ^ { * }$ [24]

Let $A = U D V ^ { \prime }$ b<sub>e</sub> th<sub>e s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ue</sub> d<sub>ecompos</sub>iti<sub>on o</sub>f �<sub>.</sub> Th<sub>en</sub>

$$
A ^ { \prime } A = ( U D V ^ { \prime } ) ^ { \prime } ( U D V ^ { \prime } ) = V D U ^ { \prime } U D V ^ { \prime } = V D ^ { 2 } V ^ { \prime } ,
$$

<sub>w</sub>hich is an ei<sub>g</sub>endecom<sub>p</sub>osition<sub>,</sub> so

$$
( A ^ { \prime } A ) ^ { - 1 / 2 } = V D ^ { - 1 } V ^ { \prime } .
$$

Multi<sub>p</sub>l<sub>y</sub>in<sub>g</sub> b<sub>y</sub> $A ^ { \prime } = V D U ^ { \prime }$ <sub>g</sub><sup>i</sup>ves

$$
( A ^ { \prime } A ) ^ { - 1 / 2 } A ^ { \prime } = V D ^ { - 1 } V ^ { \prime } V D U ^ { \prime } = V D ^ { - 1 } D U ^ { \prime } = V U ^ { \prime } .
$$

H<sub>ence</sub> th<sub>e max</sub>i<sub>m</sub>i<sub>zer can</sub> b<sub>e wr</sub>itt<sub>en</sub> i<sub>n c</sub>l<sub>ose</sub>d f<sub>orm as</sub>

$$
Q ^ { * } = V U ^ { \prime } = ( A ^ { \prime } A ) ^ { - 1 / 2 } A ^ { \prime } .
$$

With $A = \Sigma _ { Z } ^ { - 1 / 2 } \Sigma _ { Z Y }$ <sub>, we</sub> h<sub>ave</sub> $A ^ { \prime } A = \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y }$ <sub>an</sub>d $A ^ { \prime } = \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 / 2 }$ so

$$
\boldsymbol { Q } ^ { * } = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { - 1 / 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 / 2 } ,
$$

<sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
T _ { 0 } ^ { * } = Q ^ { * } \Sigma _ { Z } ^ { - 1 / 2 } = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { - 1 / 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } .
$$

## A.3 Notes on $T _ { 1 } ^ { * }$ : connection to regression.

The stron<sub>g</sub>-de<sub>p</sub>endence transformation coincides with multivariate l<sub>eas</sub>t<sub>-squares</sub> <sub>regress</sub>i<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>es</sub> <sub>on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> The <sub>p</sub>o<sub>pu</sub>lation OLS coeficient of $Y _ { p }$ <sub>o</sub>n � i<sub>s</sub> $b _ { p } ^ { * } = \Sigma _ { Z } ^ { - 1 } ( \Sigma _ { Z Y } ) _ { \cdot p } ,$ so th<sub>e op</sub>ti<sub>ma</sub>l

$$
t _ { \beta } ^ { * } = \Sigma _ { Z } ^ { - 1 } ( \Sigma _ { Z Y } ) . _ { \cdot p } / \big ( ( \Sigma _ { Z Y } ) _ { \cdot p } ^ { \prime } \Sigma _ { Z } ^ { - 1 } ( \Sigma _ { Z Y } ) . _ { \cdot p } \big ) ^ { 1 / 2 } = b _ { p } ^ { * } / \| b _ { p } ^ { * } \| _ { \Sigma _ { Z } } ,
$$

<sub>w</sub>h<sub>ere</sub> $\| \ b { x } \| _ { \Sigma _ { Z } } : = ( \ b { x } ^ { \prime } \Sigma _ { Z } \ b { x } ) ^ { 1 / 2 }$ <sub>,</sub> i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or,</sub> <sub>resca</sub>l<sub>e</sub>d <sub>so</sub> th<sub>a</sub>t $\mathrm { V a r } ( t _ { p } ^ { * \prime } Z ) = 1$ . E<sub>q</sub>uivalentl<sub>y</sub> $T _ { 1 } ^ { * } = D _ { 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$ <sub>,</sub> <sub>w</sub>ith $\Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$ the matrix of re<sub>g</sub>ression coeficients. The attained ali<sub>g</sub>nment is the <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>,</sub>

$$
J ( T _ { 1 } ^ { * } , p ) = \sqrt { ( \Sigma _ { Z Y } ) _ { \cdot p } ^ { \prime } \Sigma _ { Z } ^ { - 1 } ( \Sigma _ { Z Y } ) _ { \cdot p } } = \sqrt { ( M ^ { 2 } ) _ { p p } } = R _ { p } ,
$$

th<sub>e max</sub>i<sub>ma</sub>l <sub>corre</sub>l<sub>a</sub>ti<sub>on o</sub>f $Y _ { p }$ with an<sub>y</sub> linear combination of� (for d<sub>er</sub>i<sub>va</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s o</sub>f $t _ { p } ^ { * }$ f<sub>or genera</sub>l $Z$ <sub>an</sub>d $J ( T _ { 1 } ^ { * } , p )$ , see Section A.5).

## A.4 Identifiability

Proposition A.1. Fix a regime $\bullet \in \{ 0 , 1 , \lambda , \mathrm { e x } \}$ and let $T _ { \bullet } ^ { * } ( Z )$ denote its optimal transformation. For any invertible $A ~ \in ~ \bar { \mathbb { R } } ^ { \bar { P } \times \bar { P } }$ replacing the latent by the equivalent representation $Z ^ { \prime } = A Z$ yields the same structured representation:

$$
T _ { \bullet } ^ { * } ( Z ^ { \prime } ) Z ^ { \prime } = T _ { \bullet } ^ { * } ( Z ) Z .
$$

Hence $\tilde { Z } _ { \bullet } : = T _ { \bullet } ^ { \ast } ( Z ) Z$ identifies a canonical representation of the latent.

Proof. T<sup>h</sup>e o<sup>b</sup>jective an<sup>d</sup> constraints o<sup>f</sup> eac<sup>h</sup> regime <sup>d</sup>epen<sup>d</sup> <sub>on</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>on</sub>l<sub>y</sub> th<sub>roug</sub>h th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>vec</sub>t<sub>or</sub> $\tilde { Z } = T Z ,$ , via $\mathrm { C o v } ( \tilde { Z } , Y ) = T \Sigma _ { Z Y }$ <sub>an</sub>d $\mathrm { C o v } ( \tilde { Z } ) = T \Sigma _ { Z } T ^ { \prime }$ <sub>.</sub> U<sub>n</sub>d<sub>er</sub> $Z ^ { \prime } = A Z$ <sub>one</sub> h<sub>as</sub> $\Sigma _ { Z ^ { \prime } } = A \Sigma _ { Z } A ^ { \prime }$ <sub>an</sub>d $\Sigma _ { Z ^ { \prime } Y } = A \Sigma _ { Z Y }$ <sub>, so</sub> th<sub>e su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> $T \mapsto T ^ { \prime } =$ $T A ^ { - 1 }$ is a <sup>b</sup>ijection <sup>b</sup>etween trans<sup>f</sup>ormations o<sup>f</sup> $Z$ <sub>an</sub>d <sub>o</sub>f $Z ^ { \prime }$ <sub>w</sub>ith $T ^ { \prime } Z ^ { \prime } = T Z ,$ h<sub>ence</sub> C<sub>ov</sub> $( T ^ { \prime } Z ^ { \prime } , Y ) = \mathrm { C o v } ( T Z , Y )$ <sub>an</sub>d $\mathrm { C o v } ( T ^ { \prime } Z ^ { \prime } ) =$ Cov(��). The objective and feasible set for $Z ^ { \prime }$ th<sub>ere</sub>f<sub>ore co</sub>i<sub>nc</sub>id<sub>e</sub> wit<sup>h</sup> t<sup>h</sup>ose <sup>f</sup>or � un<sup>d</sup>er t<sup>h</sup>is <sup>b</sup>ijection, so <sup>b</sup>ot<sup>h</sup> pro<sup>bl</sup>ems <sup>h</sup>ave t<sup>h</sup>e same o<sub>p</sub>timal re<sub>p</sub>resentation ${ \tilde { Z } } ,$ <sub>an</sub>d $T _ { \bullet } ^ { * } ( Z ^ { \prime } ) Z ^ { \prime } = T _ { \bullet } ^ { * } ( Z ) Z .$ □

Corollary A.2. Taking $A = T _ { 0 } ^ { * }$ in Proposition A.1, solving the regime in the whitened frame and composing with the whitening yields the general � solution: $T _ { \bullet } ^ { * } = T _ { \bullet } ^ { * } ( \overline { { Z } } ) \ : T _ { 0 } ^ { * }$

## A.5 Summary of the defined transformations

W<sub>e now un</sub>if<sub>y an</sub>d <sub>summar</sub>i<sub>se</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>a</sub>b<sub>ove,</sub> usin<sub>g</sub> the followin<sub>g</sub> notation. Let $M : = ( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } ) ^ { 1 / 2 }$ <sub>, an</sub>d l<sub>e</sub>t

$$
J _ { T } = J ( T ) : = \operatorname { t r } \big ( \operatorname { C o v } ( T Z , Y ) \big )
$$

d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on-covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>a</sub>li<sub>gnmen</sub>t<sub>,</sub> <sub>w</sub>ith <sub>per-</sub> dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n <sub>co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n

$$
J _ { T , p } = J ( T , p ) : = { \bigl ( } \operatorname { C o v } ( T Z , Y ) { \bigr ) } _ { p p } .
$$

Th<sub>roug</sub>h<sub>ou</sub>t<sub>, we assume</sub> $\Sigma _ { Z } \succ 0$ <sub>an</sub>d th<sub>a</sub>t $\Sigma _ { Z Y }$ h<sub>as</sub> f<sub>u</sub>ll <sub>ran</sub>k<sub>.</sub> U<sub>n</sub>d<sub>er</sub> these assum<sub>p</sub>tions $M : = ( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } ) ^ { 1 / 2 }$ is s<sub>y</sub>mmetric<sub>,</sub> and <sub>p</sub>ositi<sub>ve</sub> d<sub>e</sub>finit<sub>e.</sub> S<sub>ec</sub>ti<sub>o</sub>n<sub>s</sub> $\mathrm { A } . 5 . 1 \mathrm { - } \mathrm { A } . 5 . 4$ d<sub>er</sub>i<sub>ve</sub> th<sub>e genera</sub>l f<sub>orm o</sub>f <sub>eac</sub>h transformation to<sub>g</sub>ether with its <sub>p</sub>er-dimension ali<sub>g</sub>nment $J ( T , p )$ Th<sub>e resu</sub>lt<sub>s are summar</sub>i<sub>se</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> S1<sub>.</sub>

## A.5.1 Transformation $T _ { 0 } ^ { * }$

(a) Usin<sub>g</sub> (2), we <sub>g</sub>et

$$
T _ { 0 } ^ { * } = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { - 1 / 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } = D M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } .
$$

(b) There is no scalin<sub>g</sub> in $T _ { 0 } ^ { * }$ <sub>,</sub> th<sub>us</sub> $D = I .$

(c) The cross-covariance induced b<sub>y</sub> $T _ { 0 } ^ { * }$ i<sub>s</sub>

$$
\begin{array} { r l } & { \mathrm { C o v } ( T _ { 0 } ^ { * } Z , Y ) = T _ { 0 } ^ { * } \Sigma _ { Z Y } } \\ & { \qquad = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { - 1 / 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } } \\ & { \qquad = \left( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } \right) ^ { 1 / 2 } = M . } \end{array}
$$

th<sub>us</sub>

$$
J ( T _ { 0 } ^ { * } , p ) = M _ { p p } .
$$

<table><tr><td>Transformation T</td><td>(a) T formula (general Z)</td><td> $( \mathrm { b } ) d _ { p } ( D = \mathrm { d i a g } ( d _ { p } ) )$ </td><td> $\mathrm { ( c ) } J _ { T , p }$ </td></tr><tr><td> $T _ { 0 } ^ { * }$ </td><td> $D M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$ </td><td>1</td><td> $M _ { p p }$ </td></tr><tr><td> $T _ { 1 } ^ { * }$ </td><td> $D \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$ </td><td> $\left( ( M ^ { 2 } ) _ { p p } \right) ^ { - 1 / 2 }$ </td><td> $\sqrt { ( M ^ { 2 } ) _ { \ d p p } } = \| M _ { \cdot p } \|$ </td></tr><tr><td> $T _ { \lambda } ^ { * }$ </td><td> $D \big ( ( 1 - \lambda ) M ^ { - 1 } + \lambda I \big ) \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$ </td><td> $\big ( ( ( 1 - \lambda ) I + \lambda M ) _ { p p } ^ { 2 } \big ) ^ { - 1 / 2 }$ </td><td> $( 1 - \lambda ) M _ { p p } + \lambda ( M ^ { 2 } ) _ { p p }$   $\sqrt { ( 1 - \lambda ) ^ { 2 } + 2 \lambda ( 1 - \lambda ) M _ { p p } + \lambda ^ { 2 } ( M ^ { 2 } ) _ { p p } }$ </td></tr><tr><td> $T _ { \mathrm { e x } } ^ { \ast }$ </td><td> $D \Sigma _ { Z Y } ^ { - 1 }$ </td><td> $\left( ( M ^ { - 2 } ) _ { p p } \right) ^ { - 1 / 2 }$ </td><td> $\left( ( M ^ { - 2 } ) _ { p p } \right) ^ { - 1 / 2 }$ </td></tr></table>

Table S1: Summary of the transformations, (a) their formulas, (b) diagonal normalizers, and (c) per-dimension correlations with covariates.

A.5.2 Transformation $T _ { 1 } ^ { * }$

(a) From (4), the o<sub>p</sub>timal rows in the whitened form are

$$
t _ { \mathscr { p } } ^ { \ast } = \frac { \left( \Sigma _ { \overline { { Z } } Y } \right) \cdot _ { \mathscr { p } } } { \left. \left( \Sigma _ { \overline { { Z } } Y } \right) \cdot _ { \mathscr { p } } \right. } .
$$

Collectin<sub>g</sub> them and usin<sub>g</sub> $\Sigma _ { \overline { { Z } } Y } = T _ { 0 } ^ { * } \Sigma _ { Z Y } = M$ <sub>g</sub><sup>i</sup>ves

$$
T _ { 1 } ^ { * } ( { \overline { { Z } } } ) = D _ { 1 } M ,
$$

th<sub>us</sub>

$$
T _ { 1 } ^ { * } = T _ { 1 } ^ { * } ( \overline { { { Z } } } ) T _ { 0 } ^ { * } = D _ { 1 } M M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } = D _ { 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } .
$$

(b) The normalizer $D _ { 1 }$ <sub>sca</sub>l<sub>es eac</sub>h r<sub>ow</sub> t<sub>o u</sub>nit n<sub>o</sub>rm<sub>.</sub> Sin<sub>ce</sub> $\| ( \Sigma _ { \overline { { Z } } Y } ) . _ { \mathcal { P } } \| = \| ( T _ { 0 } ^ { * } \Sigma _ { Z Y } ) . _ { \mathcal { P } } \| = \| M . _ { \mathcal { P } } \|$ <sub>,</sub> <sub>an</sub>d $\| M _ { \cdot p } \| ^ { 2 } = M _ { \cdot p } ^ { \prime } M _ { \cdot p } =$ $( M ^ { 2 } ) _ { p p }$ (usin<sub>g</sub> $M = M ^ { \prime } )$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
D _ { 1 } = \mathrm { d i a g } \big ( \| M _ { \cdot p } \| ^ { - 1 } \big ) = \mathrm { d i a g } \big ( ( M ^ { 2 } ) _ { p p } ^ { - 1 / 2 } \big ) .
$$

(c) The cross-covariance induced b<sub>y</sub> $T _ { 1 } ^ { * }$ i<sub>s</sub>

$$
\begin{array} { r } { \mathrm { C o v } ( T _ { 1 } ^ { * } Z , Y ) = T _ { 1 } ^ { * } \Sigma _ { Z Y } = D _ { 1 } M M , } \end{array}
$$

so its <sub>�</sub>-t<sup>h</sup> <sup>d</sup>ia<sub>g</sub>ona<sup>l</sup> entr<sub>y</sub> <sub>g</sub>ives t<sup>h</sup>e <sub>p</sub>er-<sup>d</sup>imension a<sup>l</sup>i<sub>g</sub>nment

$$
J ( T _ { 1 } ^ { * } , p ) = \sqrt { \left( M ^ { 2 } \right) _ { \ L ^ { p } \ L ^ { p } } } .
$$

A.5.3 Transformation $T _ { \lambda } ^ { * }$

us<sup>i</sup>n<sub>g</sub> $\langle e _ { \boldsymbol { p } } , M _ { \cdot \boldsymbol { p } } \rangle = M _ { \boldsymbol { p p } }$ <sub>an</sub>d $\| \boldsymbol { M } _ { \cdot p } \| ^ { 2 } = ( M ^ { 2 } ) _ { p p }$ <sub>.</sub> H<sub>e</sub>n<sub>ce</sub>

(a) From (6), the o<sub>p</sub>timal rows in the whitened form are

$$
t _ { \pmb { \jmath } } ^ { * } = \frac { \lambda ( \Sigma _ { \overline { { { Z } } } Y } ) . _ { \pmb { \jmath } } + ( 1 - \lambda ) e _ { p } } { \| \lambda ( \Sigma _ { \overline { { { Z } } } Y } ) . _ { \pmb { \jmath } } + ( 1 - \lambda ) e _ { p } \| } ,
$$

th<sub>us</sub>

$$
T _ { \lambda } ^ { * } ( \overline { { Z } } ) = D _ { \lambda } \big ( ( 1 - \lambda ) I + \lambda M \big ) ,
$$

$$
d _ { \lambda , p } = \Big ( ( 1 - \lambda ) ^ { 2 } + 2 \lambda ( 1 - \lambda ) M _ { p p } + \lambda ^ { 2 } ( M ^ { 2 } ) _ { p p } \Big ) ^ { - 1 / 2 } .
$$

and com<sub>p</sub>osin<sub>g</sub> with the whitenin<sub>g</sub> $T _ { 0 } ^ { * } = M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$

$$
T _ { \lambda } ^ { \ast } = T _ { \lambda } ^ { ( \overline { { { Z } } } ) \ast } T _ { 0 } ^ { \ast } = D _ { \lambda } \left( ( 1 - \lambda ) M ^ { - 1 } + \lambda I \right) \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } .
$$

(b) The dia<sub>g</sub>onal entries of the normalizer $D _ { \lambda }$ are $d _ { \lambda , p } = \| ( ( 1 -$ $\lambda ) I + \lambda M ) . _ { \mathscr { p } } \Vert ^ { - 1 }$ . Ex<sub>p</sub>andin<sub>g</sub> the s<sub>q</sub>uared norm<sub>,</sub>

$$
\begin{array} { r l } & { \left\| \big ( ( 1 - \lambda ) I + \lambda M ) _ { \cdot p } \right\| ^ { 2 } } \\ & { \quad = \left\| \big ( 1 - \lambda ) e _ { p } + \lambda M _ { \cdot p } \right\| ^ { 2 } } \\ & { \quad = \left( 1 - \lambda \right) ^ { 2 } + 2 \lambda \big ( 1 - \lambda \big ) \left. e _ { p } , M _ { \cdot p } \right. + \lambda ^ { 2 } \| M _ { \cdot p } \| ^ { 2 } } \\ & { \quad = \left( 1 - \lambda \right) ^ { 2 } + 2 \lambda \big ( 1 - \lambda \big ) M _ { p p } + \lambda ^ { 2 } ( M ^ { 2 } ) _ { p p } , } \end{array}
$$

(c) The cross-covariance induced b<sub>y</sub> $T _ { \lambda } ^ { * }$ i<sub>s</sub>

$$
\mathrm { C o v } \bigl ( T _ { \lambda } ^ { * } Z , Y \bigr ) = T _ { \lambda } ^ { * } \Sigma _ { Z Y } = D _ { \lambda } \left( \bigl ( 1 - \lambda \bigr ) I + \lambda M \right) M ,
$$

so its $\mathcal { P } ^ { - }$ -th dia<sub>g</sub>onal entr<sub>y</sub> <sub>g</sub>ives the <sub>p</sub>er-dimension ali<sub>g</sub>nment

$$
J ( T _ { \lambda } ^ { * } , p ) = \frac { ( 1 - \lambda ) M _ { p p } + \lambda ( M ^ { 2 } ) _ { p p } } { \sqrt { ( 1 - \lambda ) ^ { 2 } + 2 \lambda ( 1 - \lambda ) M _ { p p } + \lambda ^ { 2 } ( M ^ { 2 } ) _ { p p } } } .
$$

A.5.4 Transformation $T _ { \mathrm { e x } } ^ { \ast }$

(a) From (7), the o<sub>p</sub>timal rows are the unit-normalized rows of $\Sigma _ { \overline { { Z } } Y } ^ { - 1 } ,$

$$
t _ { \mathcal { P } } ^ { * } = \frac { ( \Sigma _ { \overline { { Z } } Y } ^ { - 1 } ) _ { \mathcal { P } } ^ { \prime } . } { \Vert ( \Sigma _ { \overline { { Z } } Y } ^ { - 1 } ) _ { \mathcal { P } } . \Vert } ,
$$

which <sub>g</sub>ives

$$
T _ { \mathrm { e x } } ^ { ( \overline { { { Z } } } ) * } = D _ { \mathrm { e x } } M ^ { - 1 } ,
$$

and com<sub>p</sub>osin<sub>g</sub> with the whitenin<sub>g</sub> $T _ { 0 } ^ { * } = M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 }$

$$
\begin{array} { l } { { T _ { \mathrm { e x } } ^ { * } = T _ { \mathrm { e x } } ^ { * } ( \overline { { Z } } ) T _ { 0 } ^ { * } = D _ { \mathrm { e x } } M ^ { - 1 } M ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } } } \\ { { \phantom { T _ { \mathrm { e x } } ^ { * } = } = D _ { \mathrm { e x } } M ^ { - 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } = D _ { \mathrm { e x } } \Sigma _ { Z Y } ^ { - 1 } , } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>s</sub>t<sub>ep uses</sub>

$$
M ^ { - 2 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } = ( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } ) ^ { - 1 } \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } = \Sigma _ { Z Y } ^ { - 1 } .
$$

(b) The normalizer $D _ { \mathrm { e x } } = \mathrm { d i a g } ( d _ { \mathrm { e x } , p } )$ i<sub>s</sub> $d _ { \mathrm { e x } , p } = \lVert ( M ^ { - 1 } ) _ { p } . \rVert ^ { - 1 } \mathrm { . }$ Since � (and $M ^ { - 1 } )$ i<sub>s</sub> <sub>sy</sub>mm<sub>e</sub>tri<sub>c,</sub> $\| ( M ^ { - 1 } ) _ { \hat { p } \cdot } \| ^ { 2 } = ( M ^ { - 1 } \bar { M } ^ { - 1 } ) _ { \hat { p } \hat { p } } =$ $( M ^ { - 2 } ) _ { p p } ,$ <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
D _ { \mathrm { e x } } = \mathrm { d i a g } \bigl ( \bigl \| ( M ^ { - 1 } ) _ { p } . \bigr \| ^ { - 1 } \bigr ) = \mathrm { d i a g } \bigl ( \bigl ( M ^ { - 2 } \bigr ) _ { p p } ^ { - 1 / 2 } \bigr ) .
$$

(c) The cross-covariance induced b<sub>y</sub> $T _ { \mathrm { e x } } ^ { \ast }$ i<sub>s</sub>

$$
\operatorname { C o v } ( T _ { \mathrm { e x } } ^ { \ast } Z , Y ) = T _ { \mathrm { e x } } ^ { \ast } \Sigma _ { Z Y } = D _ { \mathrm { e x } } \Sigma _ { Z Y } ^ { - 1 } \Sigma _ { Z Y } = D _ { \mathrm { e x } } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> di<sub>agona</sub>l<sub>, so</sub> th<sub>e cross-covar</sub>i<sub>ance cons</sub>t<sub>ra</sub>i<sub>n</sub>t h<sub>o</sub>ld<sub>s</sub> and the <sub>p</sub>er-dimension ali<sub>g</sub>nment is

$$
J ( T _ { \mathrm { e x } } ^ { \ast } , / p ) = \bigl ( D _ { \mathrm { e x } } \bigr ) _ { \mathscr { p p } } = \bigl ( ( M ^ { - 2 } ) _ { \mathscr { p p } } \bigr ) ^ { - 1 / 2 } .
$$

## A.6 Proof of Theorem 3.1

(a) The theorem follows b<sub>y</sub> summin<sub>g</sub> the <sub>p</sub>er-dimension ine<sub>q</sub>ualities <sub>over</sub> <sub>�.</sub> Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith it<sub>s</sub> <sub>equa</sub>lit<sub>y</sub> <sub>con</sub>diti<sub>on,</sub> i<sub>s</sub> Lemma A.3. The remainin<sub>g</sub> t<sub>w</sub>o<sub>,</sub> to<sub>g</sub>ether <sub>w</sub>ith the monotonicit<sub>y</sub> of $\lambda \mapsto J ( T _ { \lambda } ^ { * } )$ <sub>,</sub> <sub>a</sub>r<sub>e</sub> L<sub>e</sub>mm<sub>a</sub> A<sub>.</sub>4<sub>.</sub> In l<sub>e</sub>mm<sub>as</sub> <sub>p</sub>r<sub>oo</sub>f<sub>s,</sub> <sub>we</sub> <sub>use</sub> th<sub>e</sub> n<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n and <sub>p</sub>er-dimension re<sub>p</sub>resentations of Section ${ \mathrm { A . 5 } } ,$ in <sub>p</sub>articular that $M = ( \Sigma _ { Z Y } ^ { \prime } \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } ) ^ { 1 / 2 }$

(b) The theorem follows directl<sub>y</sub> from the <sub>p</sub>roblem formulations.

Lemma A.3. For each ${ \boldsymbol { \mathscr { P } } } ,$

$$
J _ { T _ { e x } ^ { * } , p } ~ \leq ~ J _ { T _ { 0 } ^ { * } , p }
$$

with equality if and only $i f M _ { p p } ^ { 2 } = \| M _ { \cdot p } \| ^ { 2 }$ , i.e. the column $M _ { \cdot p }$ has no of-diagonal entries.

Proof<sub>.</sub> Sin<sub>ce</sub> � i<sub>s</sub> <sub>sy</sub>mm<sub>e</sub>tri<sub>c</sub> <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>finit<sub>e,</sub> th<sub>e</sub> t<sub>wo</sub> <sub>pe</sub>rdi<sub>mens</sub>i<sub>on va</sub>l<sub>ues are</sub> $J _ { T _ { 0 } ^ { * } , p } = M _ { p p }$ <sub>an</sub>d $J _ { T _ { e x } ^ { * } , \rho } = \left( ( M ^ { - 2 } ) _ { p p } \right) ^ { - 1 / 2 }$ , so th<sub>e c</sub>l<sub>a</sub>i<sub>m</sub> i<sub>s</sub>

$$
\left( ( M ^ { - 2 } ) _ { p p } \right) ^ { - 1 / 2 } \ \leq \ M _ { p p } .
$$

We <sub>p</sub>rove it throu<sub>g</sub>h the intermediate <sub>q</sub>uantit<sub>y</sub> $\left( ( M ^ { - 1 } ) _ { p p } \right) ^ { - 1 }$ , us<sup>i</sup>n<sub>g</sub> the Cauch<sub>y</sub>-Schwarz ine<sub>q</sub>ualit<sub>y</sub> twice.

Step $\ L \ L _ { l : } \left( ( M ^ { - 2 } ) _ { \ L { \hat { p } \ L } } \right) ^ { - 1 / 2 } \ L \ L \leq \left( ( M ^ { - 1 } ) _ { \ L \hat { p } \ L { \hat { p } } } \right) ^ { - 1 }$ . App<sup>l</sup>y Cauc<sup>h</sup>y-Sc<sup>h</sup>warz to $e _ { p }$ <sub>an</sub>d $M ^ { - 1 } e _ { p } { \mathrm { : } }$

$$
( \boldsymbol { M } ^ { - 1 } ) _ { \boldsymbol { p } \boldsymbol { p } } = \langle e _ { \boldsymbol { p } } , \boldsymbol { M } ^ { - 1 } \boldsymbol { e } _ { \boldsymbol { p } } \rangle \leq \| \boldsymbol { e } _ { \boldsymbol { p } } \| \| \boldsymbol { M } ^ { - 1 } \boldsymbol { e } _ { \boldsymbol { p } } \| = \big ( ( \boldsymbol { M } ^ { - 2 } ) _ { \boldsymbol { p } \boldsymbol { p } } \big ) ^ { 1 / 2 } ,
$$

us<sup>i</sup>n<sub>g</sub> $\| e _ { p } \| = 1$ <sub>an</sub>d $\| M ^ { - 1 } e _ { p } \| ^ { 2 } = e _ { p } ^ { \prime } M ^ { - 2 } e _ { p } = ( M ^ { - 2 } ) _ { p p }$ . <sup>S</sup><sub>q</sub>uar<sup>i</sup>n<sub>g</sub> and invertin<sub>g g</sub>ives the claim.

Step $2 \colon \left( ( M ^ { - 1 } ) _ { \ L { \mathscr { p p } } } \right) ^ { - 1 } \leq M _ { \ L { \mathscr { p p } } }$ . App<sup>l</sup>y Cauc<sup>h</sup>y-Sc<sup>h</sup>warz to $M ^ { 1 / 2 } e _ { p }$ <sub>an</sub>d $M ^ { - 1 / 2 } e _ { p } ;$

$$
\begin{array} { r l } & { 1 = \langle e _ { \dot { p } } , e _ { \dot { p } } \rangle = \langle M ^ { 1 / 2 } e _ { \dot { p } } , M ^ { - 1 / 2 } e _ { \dot { p } } \rangle } \\ & { \qquad \leq \| M ^ { 1 / 2 } e _ { \dot { p } } \| \| M ^ { - 1 / 2 } e _ { \dot { p } } \| = M _ { \dot { p } \dot { p } } ^ { 1 / 2 } ( M ^ { - 1 } ) _ { \dot { p } \dot { p } } ^ { 1 / 2 } , } \end{array}
$$

so $M _ { p p } \left( M ^ { - 1 } \right) _ { \mathscr { p p } } \geq 1 _ { \cdot }$ , i.e.  (�<sup>−1</sup>)<sub>��</sub>  <sup>−1</sup> ≤ �<sub>��</sub> .

Chainin<sub>g</sub> the two ste<sub>p</sub>s<sub>,</sub>

$$
J _ { T _ { e x } ^ { * } , \boldsymbol { p } } = \bigl ( ( M ^ { - 2 } ) _ { \boldsymbol { p } \boldsymbol { p } } \bigr ) ^ { - 1 / 2 } \le \bigl ( ( M ^ { - 1 } ) _ { \boldsymbol { p } \boldsymbol { p } } \bigr ) ^ { - 1 } \le M _ { \boldsymbol { p } \boldsymbol { p } } = J _ { T _ { 0 } ^ { * } , \boldsymbol { p } } .
$$

E<sub>q</sub>ualit<sub>y</sub> in Ste<sub>p</sub> 1 holds if $M ^ { - 1 } e _ { p } \parallel e _ { p } ,$ i<sub>.e.</sub> $e _ { p }$ is an ei<sub>g</sub>envector of �<sub>;</sub> the same condition makes Ste<sub>p</sub> 2 an e<sub>qu</sub>alit<sub>y</sub>. This is e<sub>qu</sub>ival<sub>en</sub>t t<sub>o</sub> $M _ { \cdot p }$ havin<sub>g</sub> no of-dia<sub>g</sub>onal entries<sub>,</sub> i.e. $\dot { M } _ { p p } ^ { 2 } = \lVert M _ { \cdot p } \rVert ^ { 2 }$ <sub>,</sub> <sub>an</sub>d other<sub>w</sub>ise both ine<sub>qu</sub>alities are strict. □

Lemma A.4. The map $\lambda \mapsto J _ { T _ { \lambda } ^ { * } , p }$ is non-decreasing on [0, 1], thus

$$
J _ { T _ { 0 } ^ { * } , { p } } \ \leq \ J _ { T _ { \lambda } ^ { * } , { p } } \ \leq \ J _ { T _ { 1 } ^ { * } , { p } } , \qquad \lambda \in [ 0 , 1 ] .
$$

The inequalities are strict unless $\lambda \in \{ 0 , 1 \}$ or $M _ { p p } ^ { 2 } = \| M _ { \cdot p } \| ^ { 2 }$ , i.e.   
unless the column $M _ { \cdot p }$ has no of-diagonal entries.

Proof<sub>.</sub> Fr<sub>o</sub>m S<sub>ec</sub>ti<sub>o</sub>n A<sub>.</sub>5<sub>,</sub> th<sub>e</sub> <sub>pe</sub>r-dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n <sub>a</sub>li<sub>g</sub>nm<sub>e</sub>nt i<sub>s</sub>

$$
J _ { T _ { \lambda } ^ { * } , p } = \frac { \lambda \| M _ { \cdot p } \| ^ { 2 } + \left( 1 - \lambda \right) M _ { p p } } { \sqrt { \lambda ^ { 2 } \| M _ { \cdot p } \| ^ { 2 } + 2 \lambda ( 1 - \lambda ) M _ { p p } + ( 1 - \lambda ) ^ { 2 } } } .
$$

Writin<sub>g</sub> thi<sub>s</sub> <sub>as</sub> $u ( \lambda ) w ( \lambda ) ^ { - 1 / 2 }$ <sub>w</sub>ith <sub>numera</sub>t<sub>or� an</sub>d $w = \lambda ^ { 2 } \lVert M . _ { p } \rVert ^ { 2 } +$ $2 \lambda ( 1 - \lambda ) M _ { p p } + ( 1 - \lambda ) ^ { 2 }$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { \frac { d } { d \lambda } J _ { T _ { \lambda } ^ { * } , p } = w ^ { - 3 / 2 } \big ( u ^ { \prime } w - \frac { 1 } { 2 } u w ^ { \prime } \big ) } \end{array}$ <sub>.</sub> Ex-<sub>p</sub>an<sup>di</sup>n<sub>g</sub> $u ^ { \prime } w - { \textstyle \frac { 1 } { 2 } } u w ^ { \prime }$ <sub>cance</sub>l<sub>s</sub> <sub>a</sub>ll t<sub>erms</sub> <sub>excep</sub>t $\big ( 1 - \lambda \big ) \big ( \| M _ { \cdot p } \| ^ { 2 } - M _ { \dot { p } \dot { p } } ^ { 2 } \big )$

![](images/8339655a8495212b74c7e387c075222408ec2b69e67fe6ca1d66a0a48c8cef19.jpg)  
Figure S6: Graphical model for iFA (uni-modal case).

<sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
\frac { d } { d \lambda } J _ { T _ { \lambda } ^ { * } , \mathcal { P } } = \frac { \left( 1 - \lambda \right) \left( \Vert M _ { \cdot \mathcal { P } } \Vert ^ { 2 } - M _ { \mathcal { P } \mathcal { P } } ^ { 2 } \right) } { \left( \lambda ^ { 2 } \Vert M _ { \cdot \mathcal { P } } \Vert ^ { 2 } + 2 \lambda ( 1 - \lambda ) M _ { \mathcal { P } \mathcal { P } } + ( 1 - \lambda ) ^ { 2 } \right) ^ { 3 / 2 } } .
$$

Both factors in the numerator are non-ne<sub>g</sub>ative on $\left[ 0 , 1 \right] : 1 - \lambda \geq 0 .$ <sub>an</sub>d $\begin{array} { r } { M _ {  { p p } } ^ { 2 } \le \sum _ { k } M _ { k  { p } } ^ { 2 } = \| M _ {  { p } } \| ^ { 2 } } \end{array}$ <sub>s</sub>in<sub>ce</sub> $M _ { p p }$ i<sub>s one en</sub>t<sub>ry o</sub>f th<sub>e co</sub>l<sub>umn</sub> $M _ { \cdot p }$ <sub>.</sub> H<sub>e</sub>n<sub>ce</sub> $\begin{array} { r } { \frac { d } { d \lambda } J _ { T _ { \lambda } ^ { * } , p } \geq 0 , } \end{array}$ <sub>, an</sub>d th<sub>e en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>va</sub>l<sub>ues</sub> $J _ { T _ { 0 } ^ { * } , p } = M _ { p p }$ <sub>an</sub>d $J _ { T _ { 1 } ^ { * } , p } = \| M _ { \cdot p } \|$ <sub>g</sub>ive the lower and u<sub>pp</sub>er bound<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. The derivative vanishes onl<sub>y</sub> at $\lambda = 1$ <sub>or</sub> <sub>w</sub>h<sub>en</sub> $\| M _ { \cdot p } \| ^ { 2 } = M _ { p p } ^ { 2 } .$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>y</sub>ields the strictness condition. □

## B Additional model details

Let � $\iota = 1 , \ldots , M$ i<sub>n</sub>d<sub>ex</sub> th<sub>e mo</sub>d<sub>a</sub>liti<sub>es</sub> i<sub>n</sub> th<sub>e mu</sub>lti<sub>-mo</sub>d<sub>a</sub>l d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> F<sub>or</sub> <sub>eac</sub>h <sub>mo</sub>d<sub>a</sub>lit<sub>y</sub> $m ,$ <sub>we</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d d<sub>a</sub>t<sub>a</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> b<sub>y</sub> $\mathbf { X } ^ { ( m ) }$ the corres<sub>p</sub>ondin<sub>g</sub> loadin<sub>g</sub> matrix b<sub>y</sub> $\mathbf { W } ^ { ( m ) }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>mo</sub>d<sub>a</sub>lit<sub>y-</sub> s<sub>p</sub>ecific <sub>p</sub>arameters b<sub>y</sub> ${ \pmb { \alpha } } ^ { ( m ) }$ <sub>an</sub>d $\pmb { \tau } ^ { ( m ) }$ . The latent re<sub>p</sub>resentation $\tilde { \mathbf { Z } }$ is s<sup>h</sup>are<sup>d</sup> across a<sup>ll</sup> mo<sup>d</sup>a<sup>l</sup>ities. T<sup>h</sup>e joint <sup>d</sup>istri<sup>b</sup>ution <sup>f</sup>actorizes as

$$
\begin{array} { l } { \displaystyle \rho ( \{ \mathbf x ^ { ( m ) } \} _ { m = 1 } ^ { M } , \tilde { \mathcal { Z } } _ { s } \{ \mathbf w ^ { ( m ) } \} _ { m = 1 } ^ { M } , \{ \boldsymbol { \varepsilon } ^ { ( m ) } \} _ { m = 1 } ^ { M } , \{ \boldsymbol { \tau } ^ { ( m ) } \} _ { m = 1 } ^ { M } , \{ \boldsymbol { \tau } ^ { ( m ) } \} _ { m = 1 } ^ { M } \mid \nabla _ { \mathcal { N } , \beta } ) = } \\ { \displaystyle \ \prod _ { n = 1 } ^ { M } \prod _ { m = 1 } ^ { M } \prod _ { d = 1 } ^ { D } \big ( \boldsymbol { x } _ { n , d } ^ { ( m ) } \mid \boldsymbol { \tilde { z } } _ { n } , ( \boldsymbol { w } _ { d , \cdot } ^ { ( m ) } ) ^ { \prime } , \boldsymbol { 1 } / \boldsymbol { \tau } _ { d } ^ { ( m ) } \big ) } \\ { \displaystyle \prod _ { n = 1 } ^ { N } \prod _ { f = 1 } ^ { P } \big ( \boldsymbol { z } _ { n , p } \mid \beta _ { \bar { \phi } } ^ { ( 0 ) } + \beta _ { p } y _ { n , p } , 1 - \beta _ { p } ^ { 2 } \big ) \prod _ { n = 1 } ^ { N } \prod _ { k = P + 1 } ^ { K } N ( z _ { n , k } \mid 0 , 1 ) } \\ { \displaystyle \prod _ { m = 1 } ^ { M } \prod _ { d = 1 } ^ { D } \big ( \prod _ { k = 1 } ^ { K } \big ( \boldsymbol { w } _ { d , k } ^ { ( m ) } \mid 0 , 1 / \alpha _ { k } ^ { ( m ) } \big ) \prod _ { m = 1 } ^ { M } K \big ) \mathcal { G } ( \alpha _ { k } ^ { ( m ) } \mid a _ { 0 } ^ { ( \alpha ) } , b _ { 0 } ^ { ( \alpha ) } ) } \\ { \displaystyle \prod _ { m = 1 } ^ { M } \prod _ { f = 1 } ^ { D } \mathcal { G } ( \tau _ { d } ^ { ( m ) } \mid a _ { 0 } ^ { ( \alpha ) } , b _ { 0 } ^ { ( \alpha ) } ) , } \end{array}
$$

The <sub>g</sub>ra<sub>p</sub>hical model for the uni-modal case is shown in Fi<sub>g</sub>. S6.

![](images/d0f723cdb13c69d1f230754a70f30887a62b133a4aeff7f380a76b37f37ad79d.jpg)  
Table S2: Simulation scenarios with diferent covariate dependency structures

## C Additional details on the numerical experiments

## C.1 Simulation scenarios

We <sub>g</sub>enerate s<sub>y</sub>nthetic data from the <sub>g</sub>enerative model of E<sub>q</sub>. (9) (Fi<sub>g</sub>. S6) with � = �, var<sub>y</sub>in<sub>g</sub> the covariance structure of the co-<sub>var</sub>i<sub>a</sub>t<sub>es</sub> t<sub>o re</sub>fl<sub>ec</sub>t <sub>qua</sub>lit<sub>a</sub>ti<sub>ve</sub>l<sub>y</sub> dif<sub>eren</sub>t b<sub>u</sub>t <sub>rea</sub>li<sub>s</sub>ti<sub>c</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>pa</sub>tt<sub>e</sub>rn<sub>s.</sub> Th<sub>e</sub> f<sub>ou</sub>r <sub>sce</sub>n<sub>a</sub>ri<sub>os</sub> <sub>w</sub>ith dif<sub>e</sub>r<sub>e</sub>nt <sub>cova</sub>ri<sub>a</sub>t<sub>e</sub>-<sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>are</sub> <sub>summar</sub>i<sub>ze</sub>d i<sub>n</sub> T<sub>a</sub>b<sub>.</sub> S2<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>parame</sub>t<sub>ers</sub> h<sub>e</sub>ld fi<sub>xe</sub>d across all of them in Tab. S3. In sim<sub>u</sub>lations<sub>,</sub> <sub>w</sub>e <sub>v</sub>ar<sub>y</sub> t<sub>w</sub>o <sub>p</sub>arameters (Tab. S4): � ∈ [0, 1] controls the stren<sub>g</sub>th of the covariate correlations b<sub>y</sub> inter<sub>p</sub>olatin<sub>g</sub> the covariance matrix toward the identit<sub>y,</sub> <sub>an</sub>d $b \in \left[ 0 , 1 \right]$ <sub>sca</sub>l<sub>es</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>or-covar</sub>i<sub>a</sub>t<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or con</sub>t<sub>ro</sub>l<sub>-</sub> li<sub>ng</sub> th<sub>e s</sub>t<sub>reng</sub>th <sub>o</sub>f th<sub>e pa</sub>i<sub>rw</sub>i<sub>se</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e</sub> d<sub>epen</sub>d<sub>ence.</sub>

<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td> $\overline { { N } }$ </td><td>number of observations</td><td>500</td></tr><tr><td>D</td><td>number of features</td><td>100</td></tr><tr><td>K</td><td>number of factors</td><td>10</td></tr><tr><td>β</td><td>a vector of coefficients</td><td>(0.9, 0.75, 0.6, 0.45, 0.3)</td></tr><tr><td>θ</td><td>noise to signal ratio</td><td>0.1</td></tr></table>

Table S3: Simulation parameter settings

## C.2 Metrics

Let $Z ~ = ~ ( z _ { 1 } , \ldots , z _ { K } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d $Y \ =$ $( y _ { 1 } , \dotsc , y _ { P } )$ th<sub>e o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es.</sub> F<sub>or eac</sub>h <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> $y _ { p } ,$ <sub>we</sub> fit a re<sub>g</sub>ressor <sub>p</sub>re<sup>di</sup>ct<sup>i</sup>n<sub>g</sub> $y _ { p }$ f<sub>rom</sub> $Z$ and obtain a <sub>p</sub>er-latent im<sub>p</sub>ort<sub>ance vec</sub>t<sub>or; co</sub>ll<sub>ec</sub>ti<sub>ng</sub> th<sub>ese across � y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> i<sub>mpor</sub>t<sub>ance ma</sub>t<sub>r</sub>i<sub>x</sub> $R \in \mathbb { R } ^ { K \times P }$ <sub>, w</sub>h<sub>ere</sub> $R _ { k p }$ <sub>measures</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f $z _ { k }$ to <sub>p</sub>redict-<sup>i</sup>n<sub>g</sub> $y _ { p } .$ . On sam<sub>p</sub>le data, followin<sub>g</sub> [18], we set $R _ { k p }$ t<sub>o</sub> th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> Lasso coe<sup>fi</sup>cient. For popu<sup>l</sup>ation-<sup>l</sup>eve<sup>l</sup> metrics, we exp<sup>l</sup>oit t<sup>h</sup>e joint <sub>covar</sub>i<sub>ance o</sub>f � <sub>an</sub>d � <sub>an</sub>d <sub>use</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> OLS <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s:</sub> $R _ { k p } = | ( \Sigma _ { Z } ^ { - 1 } \Sigma _ { Z Y } ) _ { k p } |$ . We com<sub>p</sub>ute these metrics onl<sub>y</sub> for the informed (ali ned with covariates) latent dimensions.

• Disentanglement (D) measures the degree to which each l<sub>a</sub>t<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> $z _ { k }$ <sub>enco</sub>d<sub>es</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> covariate. A latent that contributes to <sub>p</sub>redictin<sub>g</sub> onl<sub>y</sub> one $y _ { p }$ i<sub>s</sub> f<sub>u</sub>ll<sub>y</sub> di<sub>sen</sub>t<sub>ang</sub>l<sub>e</sub>d<sub>;</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t th<sub>a</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> t<sub>o</sub> <sub>many</sub> i<sub>s</sub> <sub>en</sub>t<sub>ang</sub>l<sub>e</sub>d<sub>.</sub> F<sub>orma</sub>ll<sub>y,</sub>

$$
D = \sum _ { k = 1 } ^ { K } \rho _ { k } D _ { k } , \quad D _ { k } = 1 - \frac { H ( \tilde { R } _ { k , \cdot } ) } { \log P } ,
$$

$$
\tilde { R } _ { k p } = \frac { R _ { k p } } { \sum _ { q = 1 } ^ { P } R _ { k q } } , \quad \rho _ { k } = \frac { \sum _ { p } R _ { k p } } { \sum _ { k ^ { \prime } , p ^ { \prime } } R _ { k ^ { \prime } p ^ { \prime } } } ,
$$

where �(·) is the Shannon entro<sub>py</sub> of the row-normalised im<sub>p</sub>ortance vector. A hi<sub>g</sub>h � means each latent is s<sub>p</sub>e-<sub>c</sub>i<sub>a</sub>li<sub>se</sub>d t<sub>o one</sub> f<sub>ac</sub>t<sub>or an</sub>d <sub>a</sub> l<sub>ow</sub> � <sub>means</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t<sub>s are</sub> entan<sub>g</sub>led across factors. [18]

## • Completeness (C)

M<sub>easures</sub> th<sub>e</sub> d<sub>egree</sub> t<sub>o w</sub>hi<sub>c</sub>h <sub>eac</sub>h <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th f<sub>ac</sub>t<sub>or</sub> i<sub>s</sub> <sub>cap</sub>t<sub>ure</sub>d b<sub>y a s</sub>i<sub>ng</sub>l<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t<sub>.</sub> F<sub>orma</sub>ll<sub>y,</sub>

$$
C = \sum _ { p = 1 } ^ { P } \omega _ { \hat { p } } C _ { \hat { p } } , \quad C _ { \hat { p } } = 1 - \frac { H ( \tilde { R } _ { \cdot , \hat { p } } ) } { \log K } ,
$$

$$
\tilde { R } _ { k p } = \frac { R _ { k p } } { \sum _ { k ^ { \prime } = 1 } ^ { K } R _ { k ^ { \prime } p } } , \quad \omega _ { p } = \frac { \sum _ { k } R _ { k p } } { \sum _ { k ^ { \prime } , p ^ { \prime } } R _ { k ^ { \prime } p ^ { \prime } } } .
$$

A hi<sub>g</sub>h � <sub>means eac</sub>h f<sub>ac</sub>t<sub>or</sub> i<sub>s concen</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n one</sub> l<sub>a</sub>t<sub>en</sub>t<sub>,</sub> <sub>w</sub>h<sub>ereas a</sub> l<sub>ow</sub> � <sub>means a</sub> f<sub>ac</sub>t<sub>or</sub>’<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s sprea</sub>d across man<sub>y</sub> latents. [18]

• Informativeness (I) measures whether the latent represent<sub>a</sub>ti<sub>on ac</sub>t<sub>ua</sub>ll<sub>y con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on nee</sub>d<sub>e</sub>d t<sub>o recover</sub> th<sub>e groun</sub>d<sub>-</sub>t<sub>ru</sub>th f<sub>ac</sub>t<sub>ors -</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f h<sub>ow</sub> th<sub>a</sub>t i<sub>n</sub>f<sub>or-</sub> mation is or<sub>g</sub>anised. Com<sub>p</sub>uted as the mean out-of-sam<sub>p</sub>le <sub>p</sub>redictive <sub>p</sub>erformance of the re<sub>g</sub>ressors:

$$
I = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } R ^ { 2 } ( \hat { y } _ { p } , y _ { p } ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\hat { y } _ { p }$ is the <sub>p</sub>rediction of $y _ { p }$ fr<sub>o</sub>m �<sub>.</sub> A hi<sub>g</sub>h � indi<sub>ca</sub>t<sub>es</sub> t<sup>h</sup>e <sup>l</sup>atents joint<sup>l</sup>y preserve covariate in<sup>f</sup>ormation an<sup>d</sup> a <sup>l</sup>ow � indicates information loss. [18]

## • Separated Attribute Predictability (SAP)

F<sub>o</sub>r <sub>eac</sub>h <sub>pa</sub>ir $( z _ { k } , y _ { p } )$ <sub>,</sub> fit a <sub>u</sub>nivariate re<sub>g</sub>ression $\hat { y } _ { p } = a _ { k p } z _ { k } +$ $b _ { k p }$ and com<sub>p</sub>ute the <sub>p</sub>redictive score $S _ { k p } = R ^ { 2 } ( \hat { y } _ { p } , y _ { p } )$ <sub>.</sub> F<sub>o</sub>r <sub>eac</sub>h <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th f<sub>ac</sub>t<sub>or,</sub> <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> <sub>gap</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub>

The Trade-of Between Covariate Dependence and Latent Structure in Representation Learning
<table><tr><td>Parameter</td><td>Description</td><td>Values</td></tr><tr><td> $\overline { { \alpha \in [ 0 , 1 ] } }$ </td><td>Controls the strength of covariate dependences. The covariance matrix is transformed as 0, 0.25, 0.5, 0.75, 1  $\Sigma _ { Y } ^ { ( \alpha ) } = \alpha \Sigma _ { Y } + ( 1 - \alpha ) I ,$  with smaller values yielding weaker correlations (0 - no correlations, 1 - strongest correlations).</td><td></td></tr><tr><td> $b \in [ 0 , 1 ]$ </td><td>Controls the strength of latent-covariate dependence by scaling the coefficient vector:  $\beta ^ { ( b ) } = b \beta ;$  the smallest b is selected so that the lowest  $\beta _ { p } = 0 . 1$  , then b are spread equilly  $( 1 / 3$  - smallest factor-covariate dependence 0.2 on average, 1 strongest factor covariate dependence 0.6 on average)</td><td>1/3, 1/2, 2/3, 3/4, 1</td></tr></table>

Table S4: Simulation parameters varied across experiments.

hi<sub>g</sub>h<sub>es</sub>t <sub>an</sub>d <sub>secon</sub>d<sub>-</sub>hi<sub>g</sub>h<sub>es</sub>t <sub>s</sub>i<sub>ng</sub>l<sub>e-</sub>f<sub>ea</sub>t<sub>ure score, an</sub>d <sub>aver-</sub> a<sub>g</sub>e over <sup>f</sup>actors:

$$
\mathrm { S A P } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \left( S _ { k _ { p } ^ { ( 1 ) } , p } - S _ { k _ { p } ^ { ( 2 ) } , p } \right) ,
$$

<sub>w</sub>h<sub>ere</sub> $k _ { p } ^ { ( 1 ) }$ <sub>an</sub>d $k _ { p } ^ { ( 2 ) }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t<sub>s w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>an</sub>d <sub>secon</sub>d<sub>-</sub>hi<sub>g</sub>h<sub>es</sub>t <sub>scores</sub> f<sub>or</sub> f<sub>ac</sub>t<sub>or</sub> <sub>�.</sub> A hi<sub>g</sub>h SAP <sub>means</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>s</sub>t<sub>an</sub>d<sub>s</sub> <sub>ou</sub>t <sub>as</sub> th<sub>e</sub> d<sub>om</sub>i<sub>nan</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>o</sub>f <sub>eac</sub>h <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th f<sub>ac</sub>t<sub>or, an</sub>d <sub>a</sub> l<sub>ow</sub> SAP <sub>means severa</sub>l l<sub>a</sub>t<sub>en</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>power</sub> f<sub>or</sub> th<sub>e</sub> <sub>same</sub> f<sub>ac</sub>t<sub>or.</sub> U<sub>n</sub>lik<sub>e</sub> � <sub>an</sub>d $C ,$ w<sup>h</sup>ic<sup>h</sup> use t<sup>h</sup>e joint-regression importance matrix, SAP measures sin<sub>g</sub>le-feature <sub>p</sub>redictabilit<sub>y</sub> and is conse<sub>q</sub>uentl<sub>y</sub> more <sub>sens</sub>iti<sub>ve</sub> t<sub>o</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>among</sub> th<sub>e</sub> <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th f<sub>ac</sub>t<sub>ors.</sub> [36]

• Normalised Separated Attribute Predictability (nSAP) A <sub>sca</sub>l<sub>e</sub>-in<sub>va</sub>ri<sub>a</sub>nt <sub>va</sub>ri<sub>a</sub>nt <sub>o</sub>f SAP<sub>,</sub> in <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>gap</sub> i<sub>s</sub> di<sub>v</sub>id<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> t<sub>op</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub>’<sub>s</sub> <sub>score:</sub>

$$
\mathrm { n S A P } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \left( S _ { k _ { \ / p } ^ { ( 1 ) } , p } - S _ { k _ { \ / p } ^ { ( 2 ) } , p } \right) / S _ { k _ { \ / p } ^ { ( 1 ) } , p } .
$$

U<sub>n</sub>lik<sub>e</sub> SAP<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>gap</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> <sub>m</sub>i<sub>xes</sub> t<sub>oge</sub>th<sub>er</sub> th<sub>e s</sub>t<sub>reng</sub>th <sub>o</sub>f th<sub>e</sub> d<sub>om</sub>i<sub>nan</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or w</sub>ith its <sup>l</sup>ead over t<sup>h</sup>e runner-up, nSAP iso<sup>l</sup>ates t<sup>h</sup>e relative domi-<sub>nance o</sub>f th<sub>e</sub> b<sub>es</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or.</sub> Thi<sub>s</sub> i<sub>s par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y use</sub>f<sub>u</sub>l <sub>w</sub>h<sub>en</sub> o<sub>v</sub>erall <sub>p</sub>redicti<sub>v</sub>e scores are lo<sub>w</sub>: a small SAP <sub>g</sub>a<sub>p</sub> ma<sub>y</sub> <sub>re</sub>fl<sub>ec</sub>t <sub>e</sub>ith<sub>er a wea</sub>k d<sub>om</sub>i<sub>nan</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or or genu</sub>i<sub>ne</sub>l<sub>y com</sub> <sub>p</sub>etin<sub>g p</sub>redictors<sub>,</sub> whereas nSAP disentan<sub>g</sub>les these two cases.

## C.3 Methods details

T<sub>a</sub>b<sub>.</sub> S5 li<sub>s</sub>t<sub>s</sub> th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>we</sub> <sub>compare</sub> <sub>aga</sub>i<sub>ns</sub>t<sub>,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> <sub>p</sub>acka<sub>g</sub>es and h<sub>yp</sub>er<sub>p</sub>arameters used. To kee<sub>p</sub> the com<sub>p</sub>arison fair<sub>,</sub> <sub>every me</sub>th<sub>o</sub>d i<sub>s g</sub>i<sub>ven</sub> th<sub>e same num</sub>b<sub>er o</sub>f f<sub>ac</sub>t<sub>ors an</sub>d th<sub>e same op-</sub> timisation bud<sub>g</sub>et: � = 10 and 2000 iterations (or e<sub>p</sub>ochs/SVI ste<sub>p</sub>s) for the simulations, and � = 20 and 5000 iterations for the real d<sub>a</sub>t<sub>a.</sub> M<sub>e</sub>th<sub>o</sub>d<sub>s w</sub>h<sub>ose</sub> f<sub>ac</sub>t<sub>ors are no</sub>t <sub>pa</sub>i<sub>re</sub>d <sub>w</sub>ith th<sub>e covar</sub>i<sub>a</sub>t<sub>es</sub> b<sub>y</sub> construction (PCA, MOFA, PLS, both su<sub>p</sub>ervised PCA variants, and iVAE) have their informed factors selected b<sub>y</sub> Hun<sub>g</sub>arian match in<sub>g</sub> on |corr(�, �)|, with si<sub>g</sub>ns ali<sub>g</sub>ned. SOFA and the re<sub>g</sub>ression b<sub>ase</sub>li<sub>ne</sub> <sub>are</sub> <sub>pa</sub>i<sub>re</sub>d b<sub>y</sub> d<sub>es</sub>i<sub>gn</sub> <sub>an</sub>d <sub>nee</sub>d <sub>no</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng.</sub>

Our method (iFA) is fitted with the same � and iteration bud<sub>g</sub>et: variational inference for 2000 (5000 for the real data) iterations, <sub>p</sub>receded b<sub>y</sub> 250 (1000) <sub>p</sub>retrainin<sub>g</sub> iterations of the unsu<sub>p</sub>ervised <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>an</sub>d <sub>s</sub>t<sub>oppe</sub>d <sub>ear</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> ELBO <sub>c</sub>h<sub>ange</sub> f<sub>a</sub>ll<sub>s</sub> b<sub>e</sub>l<sub>ow</sub>

$5 \times 1 0 ^ { - 7 }$ . The fine-tunin<sub>g</sub> ste<sub>p</sub> is run at � ∈ {0, 0.25, 0.5, 0.75, 1} for th<sub>e s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons, an</sub>d $\lambda \in \left\{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 0 . 9 \right\}$ f<sub>or rea</sub>l d<sub>a</sub>t<sub>a.</sub>

## C.4 Image representations

R<sub>epresena</sub>ti<sub>ons are ex</sub>t<sub>rac</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>ree pre</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d throu<sub>g</sub>h the Hu<sub>gg</sub>in<sub>g</sub>Face transformers:

• ViT [17],

htt<sub>p</sub>s://hu<sub>gg</sub>in<sub>g</sub>face.co/<sub>g</sub>oo<sub>g</sub>le/vit-base-<sub>p</sub>atch16-224<sub>,</sub>

• CLIP [45], htt<sub>p</sub>s://hu<sub>gg</sub>in<sub>g</sub>face.co/o<sub>p</sub>enai/cli<sub>p</sub>-vit-lar<sub>g</sub>e-<sub>p</sub>atch14<sub>,</sub>

• DINOv2 [40], htt<sub>p</sub>s://hu<sub>gg</sub>in<sub>g</sub>face.co/facebook/dinov2-base.

F<sub>o</sub>r ViT <sub>a</sub>nd DINO<sub>v</sub>2 <sub>we</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l’<sub>s</sub> <sub>poo</sub>l<sub>e</sub>d <sub>ou</sub>t<sub>pu</sub>t<sub>,</sub> f<sub>o</sub>r CLIP we ta<sup>k</sup>e t<sup>h</sup>e projecte<sup>d</sup> image <sup>f</sup>eatures, eac<sup>h</sup> o<sup>f d</sup>imension 768.

## D Supplementary figures

Thi<sub>s sec</sub>ti<sub>on co</sub>ll<sub>ec</sub>t<sub>s</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e se</sub>t <sub>o</sub>f <sub>resu</sub>lt<sub>s across a</sub>ll <sub>s</sub>i<sub>mu-</sub> lation scenarios<sub>,</sub> the <sub>p</sub>retrained re<sub>p</sub>resentations<sub>,</sub> and the runtime com<sub>p</sub>arison. The main text re<sub>p</sub>orts Scenarios 1 (PN) and 4 (N). For <sub>comp</sub>l<sub>e</sub>t<sub>eness,</sub> <sub>some</sub> <sub>pane</sub>l<sub>s</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> t<sub>ex</sub>t <sub>are</sub> <sub>repea</sub>t<sub>e</sub>d h<sub>ere:</sub>

• Fi<sub>g</sub>. S8, <sub>p</sub>anels A and B also shown in Fi<sub>g</sub>. 1;

• Fi<sub>g</sub>. S8 and Fi<sub>g</sub>. S11, ri<sub>g</sub>ht <sub>p</sub>anel of D also shown in Fi<sub>g</sub>. 3.

Fi<sub>gure</sub> S7 <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-covar</sub>i<sub>a</sub>t<sub>e an</sub>d l<sub>a</sub>t<sub>en</sub>t<sub>-</sub>l<sub>a</sub>t<sub>en</sub>t <sub>covar</sub>i<sub>ance</sub> m<sub>a</sub>tri<sub>ces</sub> f<sub>o</sub>r CLIP<sub>,</sub> ViT<sub>, a</sub>nd DINO<sub>v</sub>2 <sub>u</sub>nd<sub>e</sub>r th<sub>e</sub> t<sub>wo</sub> b<sub>ase</sub>lin<sub>es a</sub>nd th<sub>e propose</sub>d t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub>

Fi<sub>gures</sub> S8<sub>-</sub>S11 <sub>repor</sub>t<sub>,</sub> f<sub>or eac</sub>h <sub>o</sub>f th<sub>e</sub> f<sub>our scenar</sub>i<sub>os,</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a-</sub> <sub>g</sub>eneratin<sub>g</sub> covariance matrices (<sub>p</sub>anel A), the covariance matrices after each transformation (<sub>p</sub>anel B), the trade-of traced b<sub>y</sub> the famil<sub>y</sub> of transformations (<sub>p</sub>anel C), and the em<sub>p</sub>irical com<sub>p</sub>arison of all methods under two <sub>p</sub>arameter settin<sub>g</sub>s (<sub>p</sub>anel D).

Fi<sub>g</sub>ures S12-S15 re<sub>p</sub>ort<sub>,</sub> for each scenario<sub>,</sub> the three <sub>p</sub>ro<sub>p</sub>osed metrics (avera<sub>g</sub>e factor-covariate correlation, factor inde<sub>p</sub>endence, and avera<sub>g</sub>e variance ex<sub>p</sub>lained <sub>p</sub>er factor) and the five disentan<sub>g</sub>lement metrics (�, �, �, SAP, nSAP), var<sub>y</sub>in<sub>g</sub> the stren<sub>g</sub>th of the covariate correlations (�, <sub>p</sub>anel A) and of the factor-covariate associations (�, <sub>p</sub>anel B).

Fi<sub>g</sub>ure S16 re<sub>p</sub>orts <sub>p</sub>arameter recover<sub>y</sub> as the number of obser-<sub>va</sub>ti<sub>ons</sub> <sub>grows:</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s,</sub> th<sub>e</sub>i<sub>r</sub> <sub>mean</sub> <sub>square</sub>d <sub>error,</sub> <sub>an</sub>d th<sub>e cos</sub>i<sub>ne s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> b<sub>e</sub>t<sub>ween recovere</sub>d <sub>an</sub>d t<sub>rue</sub> i<sub>n</sub>f<sub>orme</sub>d f<sub>ac</sub>t<sub>ors.</sub>

Fi<sub>gure</sub> S17 <sub>compares</sub> th<sub>e run</sub>ti<sub>me o</sub>f <sub>a</sub>ll <sub>me</sub>th<sub>o</sub>d<sub>s.</sub>

<table><tr><td>Name</td><td>Description</td><td>Implementation details</td></tr><tr><td>PCA</td><td>ear method extracting orthogonal directions that max- imise variance in X. [25, 42]</td><td>PrincipalComponent Analysis. An unsupervised lin- PCA(n_components=10/20) - sklearn 1.8.0 [43]</td></tr><tr><td>MOFA</td><td>Multi-Omicss Factor Analysis A probabilistic multi- mofapy20.7.4 view factor analysis model with a mean-field variational model options: factors=10/20, [3]</td><td>approximation that approximates factor independence. spikeslab_weights=True, ard_weights=True, ard_factors=True, train options: convergence_mode=&quot;medium&quot;,</td></tr><tr><td>PLS</td><td>maximise the covariance between X and Y, subject to 1.8.0 [43] orthogonality of successive components. [55]</td><td> $\mathrm { i t e r } { = } 2 0 0 0 / 5 0 0 0$  Partial Least Squares. Extracts latent directions that PLSRegression(n_components=10/20) - sklearn</td></tr><tr><td>SPCA (Bair 2006)</td><td>Screening-based supervised PCA. A two-step coefficient above a threshold), then applies standard [43] PCA to the surviving submatrix. The factors are PCA directions of a covariate-relevant subset of features.</td><td>Our implementation: features filtered by absolute method that first screens features of X by their uni- correlation above the 0.5 empirical quantile treshold, variate association with Y (e.g., absolute correlation then PCA(n_components=10/20) - sklearn 1.8.0</td></tr><tr><td>SPCA (Barshan 2011)</td><td>maximising the Hilbert-Schmidt Independence Crite- bandwidth, linear kernel on X, rion with Y. The objective reduces to a generalised scipy 1.17.1 and numpy 2.4.6 eigenproblem on X&#x27;HLHX, where L is the Y-kernel matrix and  $\begin{array} { r } { H = I - \frac { 1 } { n } \mathbf { 1 } \mathbf { 1 } ^ { \prime } } \end{array}$  is the centring matrix.</td><td>HSIC-based supervised PCA. Finds directions in X Our implementation: RBF kernel on Y with median</td></tr><tr><td>iVAE</td><td>prior over Z is conditioned on the auxiliary variables Y ian variant via a factorised exponential family,  $\begin{array} { r } { p ( z \mid y ) = \prod _ { p } p ( z _ { p } } \end{array}$  y). Trained by maximising the conditional ELBO with units, LeakyReLU(0.1); Adam, both an encoder  $q ( \boldsymbol { z } \mid \boldsymbol { x } , \boldsymbol { y } )$  and a decoder p(x | z). [31] epochs=2000/5000.</td><td>Identifiable VAE. A variational autoencoder where the Our implementation: PyTorch 1.13.1+cu117, Gauss- encoder, prior and decoder MLPs with 2 × 64 hidden  $\mathrm { l r } \mathrm { = } ~ 1 0 ^ { - 3 }$  , batch= 256,</td></tr><tr><td>SOFA</td><td>Semi-supervised Factor Analysis An extension of biosofa 0.7.5, K = 10/20, llh=&#x27;gaussian&#x27;, MOFA that models pairwise factor-covariate dependen-&#x27;bernoul1i&#x27;, n_steps=2000/5000, lr=0.01 cies. A scaling factor (sf) reweights the factor-covariate loss term relative to the data-factor terms, providing indirect control over the factor-covariate dependence strength. [11]</td><td>e sf  $\mathsf { \bar { \Pi } } \in \{ 0 . 0 1 , \dots , 0 . 0 5 , 0 . 1 \}$ </td></tr><tr><td>Y as factors</td><td>Regression. A supervised baseline that takes the co- LinearRegression, PCA - sklearn 1.8.0 variates themselves as the first P informed factors and fills the remaining K – P slots with the leading principal components of the residual  $X - Y \hat { \beta } _ { y } .$  where  $\hat { \beta } _ { y }$  is the OLS fit of X on Y.</td><td></td></tr></table>

Table S5: Comparison methods, with a brief description and implementation details.

![](images/40f6394fc7f5e860703950f865ebe86ee85a4c4a11aef924a66ca4e84d738855.jpg)

![](images/516139453b9aa62fbe248543742b2cb67381fe03f858b5860c31e83cf379dd02.jpg)

![](images/81d3bb01bce372b0c4d0f4857c75919c6c53003504d4e9074266236471a9fe22.jpg)  
Figure S7: Covariance structure of the pretrained image representations under each transformation, for A. CLIP, B. ViT, and C. DINOv2. Each panel shows latent-covariate (top) and latent-latent (bottom) covariances for the two baselines (dimensions with max correlation, Procrustes rotation) and the proposed $T _ { \mathrm { e x } } , T _ { 0 } , T _ { 1 }$

![](images/de6f1a5a7ffd17c1b0b9940fe67b1678367093f5f8ccfc42603adfae323dd8d4.jpg)

![](images/5904ae899c5d853cc286938f3165e5dd6487eac99cbb73054725b2858957d6d4.jpg)

B  
![](images/1b24a06542d05dfe6f4b0d2fb60be27f2c406ddf4709fd94b3f28200ee1cf25a.jpg)

![](images/e5e70b67947908d5aa271ce3846489e82e00ad9fb523bca362667698ad23ac22.jpg)

![](images/d4c76dfdd7a349b8c8311835ff3187529fac1a69103bc21071563bcc58c1cebc.jpg)

![](images/ae0b2467e9e809badb8a9655d6408c4d1834e27342a572cfc3652636e769a885.jpg)

![](images/6c666f17f49d0d6a27643f84e044bb26392db3def1a9de4dbd6ee4191f3db2b5.jpg)

![](images/3a3d8646224ad41a0c4806265ddd81ffb90271d6535f4a641b3c7911c1e18e61.jpg)

Figure S8: Scenario 1 (PN) results. A. Covariance matrices $\Sigma _ { Y Z }$ and $\Sigma _ { Z }$ in the data-generating setting. B. Covariance matrices $\Sigma _ { Y \tilde { Z } }$ and $\Sigma _ { \tilde { Z } }$ after transformations $T _ { \mathbf { e x } } , T _ { 0 } , T _ { 1 / 2 } ,$ and � . C. Frontier after transformations $T _ { \lambda }$ and $T _ { \mathbf { e x } } .$ . D. Empirical results for method under two parameter settings: baseline and reduced �.  
![](images/e6d1539deacfbc1fe62b2a4c9caed28c491ecc04ec3daabd120708fe23a64a85.jpg)

![](images/a052b3dfe61d8a3ac8f6a6ae54c32cfdb73b66c77deab99f686f7c573fc0372c.jpg)

![](images/bd9fa55870a984490d5e34703d44385f8b662e191786c5ac602c9ca2bbd0abdf.jpg)

![](images/7a80292dde58eaa4c28d0ceb8a709ac510b0cca0ca7e327cc57dc5f489744d1d.jpg)

B  
![](images/a2009d6a6df56ac094bce035554e1b5bc5374738671bf4bb67e522eb17d35472.jpg)

![](images/ef625b0e1d874564255d8a8b75d59776ff34ef97dc29b0084ec3f50f6b99b854.jpg)

![](images/d601cf0a174f7ab375f776e5029ff579caa4089baeddd840f94009b73c185cd5.jpg)

![](images/839612d6e9d566e941ee93b81d59d816d4e89e2999de83e091099a90a1ce178a.jpg)  
Figure S9: Scenario 2 (AR) results.A. Covariance matrices $\Sigma _ { Y Z }$ and $\Sigma _ { Z }$ in the data-generating setting. B. Covariance matrices $\Sigma _ { Y \tilde { Z } }$ and $\Sigma _ { \tilde { Z } }$ after transformations $T _ { \mathbf { e x } } , T _ { 0 } , T _ { 1 / 2 } ,$ and $T _ { 1 } .$ . C. Frontier after transformations $T _ { \lambda }$ and $T _ { \mathbf { e x } }$ . D. Empirical results for method under two parameter settings: baseline and reduced �.

![](images/34ad8324f01a5c4cf09e485d9c67a8a93cccbe3ecf1d9eb1c951743d1f9f7814.jpg)

C  
![](images/04ce269c64bdb86d24d7e409b35e577012c1ebac545997ffb6e139ec98ec1412.jpg)

![](images/1e74a1cc48a02d1cedef274cfa605845455e7aa46aaacb6020f2bf594afe0caa.jpg)  
B

![](images/a4aab9c9b7429fdad808f3c0f62d97f66b9315341aa1c86eff47175e34e1eebb.jpg)

![](images/f1d4badf1e9875e3ca88a097570404bfb2a761047aae6cc7a409f90fc18e65ab.jpg)

![](images/42642af21c2374c210e47e092f9f3ed3e7cca1fd1bdb215c4efafdba7600ecab.jpg)

![](images/1ac8a8dec71fa8788a1437a652b34926d68ce70803192419a594adc42f458dfe.jpg)

![](images/3145b912949cc268369aea81e7ae0735811c0129c6f2dddaf8d903906231a20d.jpg)

Figure S10: Scenario 3 (P) results. A. Covariance matrices $\Sigma _ { Y Z }$ and $\Sigma _ { Z }$ in the data-generating setting. B. Covariance matrices $\Sigma _ { Y \tilde { Z } }$ and $\Sigma _ { \tilde { Z } }$ after transformations $T _ { \mathbf { e x } } , T _ { 0 } , T _ { 1 / 2 }$ , and � . C. Frontier after transformations $T _ { \lambda }$ and $T _ { \mathbf { e x } }$ . D. Empirical results for methods under two parameter settings: baseline and reduced �.  
![](images/7b737d1eee0875b80f14e303fae5392ef5556db4ef727c9b7732283ecd586954.jpg)

C  
![](images/da9ef94bfea8a5514b3f266b906954cdd227ff64d7c9ba05100bcb7f99aefcac.jpg)

D  
![](images/fdaa0b8bb65aa0c5416a8dfed7e01673e16ea09727b5de53abdd2d997e96b8f6.jpg)

![](images/8759a56bb91f9776cc499a70c0f051925d1c27ab4472306285cb32c04f15d192.jpg)

B  
![](images/2a5c71cc19337976c25e47c6d5a6df88feb40d2a7fc4333457c022ac9e3dd0f4.jpg)

![](images/75c3274fd936c2085e279430f52fedd8e2f16b8d93f148bacf902746659a8108.jpg)

![](images/97ef5dee083784e27fff05fc158b7ea7ad3e5d1648a7c3d916e246cf46b49914.jpg)  
Figure S11: Scenario 4 (N) results. A. Covariance matrices $\Sigma _ { Y Z }$ and $\Sigma _ { Z }$ in the data-generating setting. B. Covariance matrices $\Sigma _ { Y \tilde { Z } }$ and $\Sigma _ { \tilde { Z } }$ after transformations $T _ { 0 } , T _ { 1 / 2 } ,$ and $T _ { 1 }$ (no $T _ { \mathbf { e x } }$ is reported, as in Scenario $\mathbf { \boldsymbol { \mathbf { \rho } } } _ { \mathbf { \boldsymbol { \mathbf { \rho } } } } ( \mathbf { \boldsymbol { N } } ) \Sigma _ { Z Y }$ is not invertible). C. Frontier of transformations � . D. Empirical results for methods under two parameter settings: baseline and reduced �.

![](images/eb4dec6e0d30e3ac21feb65c2b7d552a16d8b3a62866b1959a6cbe5efbda6b43.jpg)  
Figure S12: Performance in Scenario 1 (PN), measured by three proposed metrics (average factor–covariate correlations, factor independence measured by the squared Frobenius distance of the correlation matrix of� from the identity, and average variance explained per factor) and five disentanglement metrics (disentanglement, completeness, informativeness, SAP, and normalised SAP). A. Varying the strength of correlations among covariates. B. Varying the strength of factor–covariate associations.

![](images/d7798c1078b0230ea529944388ffd79f77d629cb4b3310fe94459da75902f76b.jpg)  
Figure S13: Performance in Scenario 2 (AR), measured by three proposed metrics (average factor–covariate correlations, factor independence measured by the squared Frobenius distance of the correlation matrix of� from the identity, and average variance explained per factor) and five disentanglement metrics (disentanglement, completeness, informativeness, SAP, and normalised SAP). A. Varying the strength of correlations among covariates. B. Varying the strength of factor–covariate associations

![](images/576561141c67ae9aebc6d1e8fc17059e51d28316c38b411a1dfe6e533f0ca902.jpg)  
Figure S14: Performance in Scenario 3 (P), measured by three proposed metrics (average factor–covariate correlations, factor independence measured by the squared Frobenius distance of the correlation matrix of� from the identity, and average variance explained per factor) and five disentanglement metrics (disentanglement, completeness, informativeness, SAP, and normalised SAP). A. Varying the strength of correlations among covariates. B. Varying the strength of factor–covariate associations

![](images/df6ac83447bdeff298cb52d73b16b631c3bd09038058efba348183bc2ac5ccb3.jpg)  
Figure S15: Performance in Scenario 4 (N), measured by three proposed metrics (average factor–covariate correlations, factor independence measured by the squared Frobenius distance of the correlation matrix of� from the identity, and average variance explained per factor) and five disentanglement metrics (disentanglement, completeness, informativeness, SAP, and normalised SAP). A. Varying the strength of correlations among covariates. B. Varying the strength of factor–covariate associations

![](images/692d2a52a47f83a10c7beeb09d1d674d59806d420d2bb08573f9e767faa14043.jpg)

![](images/087c858061c90ef75050ab05b8b3ab832de1e8383d3ae7439175a85666bf97b5.jpg)

![](images/e42e85adeba3b980cf321a53d37a1077cfc973bad2d1b21fe71b89c0238675fe.jpg)

Figure S16: Consistency of iFA (no �) as the number of observations � grows, Scenario 1 (PN). A. Recovery of factor-covariate coeficients. B. Mean squared error of the estimated $\hat { \beta } _ { p }$ averaged over �. C. cosine similarity between recovered and true informed factors. Shaded bands show ±1 s.d. over repetitions.  
![](images/e96a31102b7963651b6205f5ae3bef35ee4d0688e866fd6f3fae1892dc9bc400.jpg)  
Figure S17: Runtime of all methods on the one simulation setup.