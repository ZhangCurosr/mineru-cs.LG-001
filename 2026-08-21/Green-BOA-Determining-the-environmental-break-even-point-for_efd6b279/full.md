# Green BOA: Determining the environmental break-even point for ML-based data compression

C<sub>a</sub>t<sub>e</sub>rin<sub>a</sub> D<sub>og</sub>li<sub>o</sub>ni <sub>ca</sub>t<sub>er</sub>i<sub>na.</sub>d<sub>og</sub>li<sub>on</sub>i<sub>@manc</sub>h<sub>es</sub>t<sub>er.ac.u</sub>k Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r Manchester<sub>,</sub> UK

Ak<sub>s</sub>h<sub>a</sub>t G<sub>up</sub>t<sub>a</sub>   
a<sup>k</sup>s<sup>h</sup>at.<sub>g</sub>u<sub>p</sub>ta-  
4<sub>@pos</sub>t<sub>gra</sub>d<sub>.manc</sub>h<sub>es</sub>t<sub>er.ac.u</sub>k   
Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r   
Manchester<sub>,</sub> UK

Th<sub>omas</sub> Elli<sub>o</sub>tt th<sub>omas.e</sub>lli<sub>o</sub>tt<sub>@manc</sub>h<sub>es</sub>t<sub>er.ac.u</sub>k Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r Manchester<sub>,</sub> UK

H<sub>a</sub>nzil<sub>a</sub> H<sub>ussa</sub>in h<sub>anz</sub>il<sub>a</sub>h<sub>ussa</sub>i<sub>n</sub>234<sub>@gma</sub>il<sub>.com</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r Manchester<sub>,</sub> UK

Sanji<sup>b</sup>an Sengupta sanji<sup>b</sup>an.sengupta@cern.c<sup>h</sup> CERN/Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r Manchester<sub>,</sub> UK

## Abstract

We summarise t<sup>h</sup>e outcome o<sup>f</sup> two summer interns<sup>h</sup>ip projects b<sub>ase</sub>d <sub>a</sub>t th<sub>e</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y o</sub>f M<sub>anc</sub>h<sub>es</sub>t<sub>er,</sub> f<sub>ocuse</sub>d <sub>on</sub> th<sub>e</sub> b<sub>rea</sub>k<sub>-even</sub> <sub>p</sub>oint in terms of environmental sustainabilit<sub>y</sub> for ML-based data com<sub>p</sub>ression al<sub>g</sub>orithms. Usin<sub>g</sub> the exam<sub>p</sub>le of a ML-based lossl<sub>ess compress</sub>i<sub>on a</sub>l<sub>gor</sub>ith<sub>m, we compare es</sub>ti<sub>ma</sub>t<sub>es</sub> f<sub>or</sub> th<sub>e car</sub>b<sub>on-</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>f<sub>ras</sub>t<sub>ruc</sub>t<sub>ure nee</sub>d<sub>e</sub>d f<sub>or</sub> ML t<sub>ra</sub>i<sub>n</sub>i<sub>ng an</sub>d i<sub>n</sub>f<sub>er-</sub> <sub>ence w</sub>ith th<sub>e car</sub>b<sub>on-equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>sav</sub>i<sub>ngs</sub> f<sub>rom re</sub>d<sub>uce</sub>d di<sub>s</sub>k <sub>s</sub>t<sub>orage</sub> re<sub>q</sub>uirements<sub>,</sub> and discuss their break-even <sub>p</sub>oint.

Context. Big Data experiments, such as those at the Large Hadron Collider [2] and in other astro<sub>p</sub>article <sub>p</sub>h<sub>y</sub>sics ex<sub>p</sub>eriments (e.<sub>g</sub>. [13]), will be recordin<sub>g</sub> several Exab<sub>y</sub>tes of data. This has a si<sub>g</sub>nificant cost in terms of both bud<sub>g</sub>et [1] and environmental resources for data stora<sub>g</sub>e (see e.<sub>g</sub>. [11, 15]). R&D on data com<sub>p</sub>ression is on<sub>g</sub>oin<sub>g</sub>, and includes machine learnin<sub>g</sub> (ML-) based data com<sub>p</sub>ression. Since this kind of com<sub>p</sub>ression techni<sub>q</sub>ues are <sub>g</sub>enerall<sub>y</sub> more com<sub>p</sub>utationall<sub>y</sub> intensive than standard com<sub>p</sub>ression al<sub>g</sub>orithms such as ZSTD [4],and LZMA [12], we investi<sub>g</sub>ate the break-even <sub>p</sub>oint between the ��<sub>2</sub>-e<sub>q</sub>uivalent cost of trainin<sub>g</sub> and executin<sub>g</sub> th<sub>e</sub> ML<sub>-</sub>b<sub>ase</sub>d <sub>compress</sub>i<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> <sub>an</sub>d th<sub>e</sub> <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>p</sub>ll<sub>us</sub> <sub>op-</sub> <sub>era</sub>ti<sub>ona</sub>l �� <sub>-equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>o</sub>f di<sub>s</sub>k <sub>s</sub>t<sub>orage</sub> th<sub>a</sub>t <sub>wou</sub>ld b<sub>e</sub> di<sub>sp</sub>l<sub>ace</sub>d b<sub>y</sub> storin<sub>g</sub> data in its com<sub>p</sub>ressed form.

Methods. We consider the energy use of a ML-based data compression al<sub>g</sub>orithms develo<sub>p</sub>ed at the Universit<sub>y</sub> of Manchester [8], and estimate its ener<sub>gy</sub> usa<sub>g</sub>e for ML trainin<sub>g</sub> and inference (com<sub>p</sub>ression/decom<sub>p</sub>ression round tri<sub>p</sub>) on an Nvidia T4 GPU. We consider <sub>severa</sub>l <sub>coun</sub>t<sub>ry-spec</sub>ifi<sub>c scenar</sub>i<sub>os</sub> f<sub>or w</sub>h<sub>ere</sub> th<sub>ese s</sub>t<sub>eps w</sub>ill b<sub>e</sub> execute<sup>d</sup>, to convert ener<sub>gy</sub> consum<sub>p</sub>t<sup>i</sup>on <sup>i</sup>nto a $C O _ { 2 }$ -e<sub>q</sub>uivalent <sup>b</sup>ase<sup>d</sup> on avera<sub>g</sub>e ener<sub>gy</sub> m<sup>i</sup>x.

W<sub>e</sub> th<sub>en cons</sub>id<sub>er</sub> th<sub>e car</sub>b<sub>on-equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>cos</sub>t <sub>o</sub>f <sub>manu</sub>f<sub>ac</sub>t<sub>ur</sub>i<sub>ng</sub> and o<sub>p</sub>eratin<sub>g</sub> disk stora<sub>g</sub>e devices<sub>,</sub> in the sha<sub>p</sub>e of Hard Disk Drives (HDD) and ta<sub>p</sub>es, for a 5 <sub>y</sub>ears lifetime. As exam<sub>p</sub>les, we consider Sea<sub>g</sub>ate Exos X18 HDDs [14] and ta<sub>p</sub>e cartrid<sub>g</sub>es [10] (followin<sub>g</sub> the methodolo<sub>g</sub>ies in [11, 16]), h<sub>yp</sub>otheticall<sub>y</sub> located in the UK.

Ins<sub>p</sub>ired b<sub>y</sub> [3], we define the break-even <sub>p</sub>oint for this <sub>p</sub>roject as the size ofthe uncompressed dataset at which the carbon cost of training and running the compression algorithm equals the embodied carbon of the extra disk storage needed for the uncompressed data. F<sub>ur</sub>th<sub>er</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on on</sub> th<sub>e me</sub>th<sub>o</sub>d<sub>s can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

R<sub>a</sub>th<sub>er</sub> th<sub>an a</sub> f<sub>u</sub>ll <sub>ana</sub>l<sub>ys</sub>i<sub>s,</sub> thi<sub>s</sub> i<sub>s a proo</sub>f<sub>-o</sub>f<sub>-pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e ca</sub>l<sub>cu</sub>l<sub>a-</sub> tion with several assum<sub>p</sub>tions usin<sub>g</sub> a limited amount of data. Onl<sub>y</sub> <sub>ope</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>cos</sub>t<sub>s</sub> fr<sub>o</sub>m CPU<sub>,</sub> GPU <sub>a</sub>nd RAM <sub>a</sub>r<sub>e</sub> <sub>co</sub>n<sub>s</sub>id<sub>e</sub>r<sub>e</sub>d f<sub>o</sub>r th<sub>e</sub> ML<sub>-</sub>b<sub>ase</sub>d d<sub>a</sub>t<sub>a</sub> <sub>compress</sub>i<sub>on,</sub> <sub>as</sub> <sub>we</sub> <sub>assume</sub> th<sub>a</sub>t th<sub>e</sub> GPU <sub>w</sub>ill <sub>a</sub>l<sub>-</sub> <sub>rea</sub>d<sub>y</sub> b<sub>e ava</sub>il<sub>a</sub>bl<sub>e an</sub>d th<sub>a</sub>t it<sub>s use</sub> f<sub>or</sub> ML<sub>-</sub>b<sub>ase</sub>d d<sub>a</sub>t<sub>a compress</sub>i<sub>on</sub>

Zh<sub>e</sub>n<sub>g</sub>k<sub>a</sub>i S<sub>u</sub>n <sub>z</sub>h<sub>eng</sub>k<sub>a</sub>i<sub>.sun@s</sub>t<sub>u</sub>d<sub>en</sub>t<sub>.manc</sub>h<sub>es</sub>t<sub>er.ac.u</sub>k Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r Manchester<sub>,</sub> UK

will be ne<sub>g</sub>li<sub>g</sub>ible com<sub>p</sub>ared to other uses within a LHC ex<sub>p</sub>eriment. The scenario considered for the ML-based data compression com-<sub>pr</sub>i<sub>ses</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>on</sub> th<sub>e</sub> f<sub>u</sub>ll d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> <sub>one</sub> <sub>compress</sub>i<sub>on</sub> and one decom<sub>p</sub>ression round (on an existin<sub>g</sub> tem<sub>p</sub>orar<sub>y</sub> disk). A b<sub>roa</sub>d<sub>er</sub> <sub>range</sub> <sub>o</sub>f <sub>scenar</sub>i<sub>os</sub> <sub>w</sub>ill b<sub>e</sub> <sub>exp</sub>l<sub>ore</sub>d i<sub>n</sub> f<sub>u</sub>t<sub>ure</sub> <sub>wor</sub>k<sub>.</sub> Si<sub>nce</sub> trainin<sub>g</sub> dominates the ener<sub>gy</sub> consum<sub>p</sub>tion<sub>,</sub> this can be considered an u<sub>pp</sub>er bound in terms of ener<sub>gy</sub> consum<sub>p</sub>tion<sub>,</sub> as in <sub>p</sub>ractice BOA’s <sub>g</sub>eneralisation ca<sub>p</sub>abilit<sub>y</sub> outlined in [8] allows for trainin<sub>g</sub> <sub>on</sub>l<sub>y on a sma</sub>ll d<sub>a</sub>t<sub>ase</sub>t <sub>an</sub>d th<sub>en us</sub>i<sub>ng</sub> thi<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l f<sub>or a</sub> <sub>muc</sub>h l<sub>arger</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub>

## Results and conclusions.

As it can be seen in Fi<sub>gu</sub>re 1<sub>,</sub> the location of the break-e<sub>v</sub>en <sub>p</sub>oint is ver<sub>y</sub> sensitive to the carbon intensit<sub>y</sub> of the countr<sub>y</sub> where the ML-based data com<sub>p</sub>ression is <sub>p</sub>erformed. As shown in <sub>p</sub>revious lit<sub>era</sub>t<sub>ure,</sub> t<sub>ape s</sub>t<sub>orage</sub> h<sub>as a</sub> l<sub>ower</sub> $C O _ { 2 }$ -e<sub>q</sub>uivalent foot<sub>p</sub>rint than HDD<sub>s,</sub> b<sub>u</sub>t <sub>a</sub>t th<sub>e cos</sub>t <sub>o</sub>f <sub>s</sub>l<sub>ower</sub> d<sub>a</sub>t<sub>a access.</sub>

![](images/807bbde7a8ac6dae4c0cef6261552ea6d273880855a06e539d01b6d61cebc552.jpg)  
Figure 1: Results of the break-even analysis in terms of $C O _ { 2 } \mathrm { { - } }$ equivalent for ML-based data compression and subsequent displaced HDD/tape storage.

W<sub>e a</sub>l<sub>so compare</sub>d ML<sub>-</sub>b<sub>ase</sub>d d<sub>a</sub>t<sub>a compress</sub>i<sub>on</sub> t<sub>o s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>a</sub>l<sub>go-</sub> rithms, takin<sub>g p</sub>erformance (com<sub>p</sub>ression ratio) into account: since BOA <sub>g</sub>enerall<sub>y</sub> achieves a better com<sub>p</sub>ression ratio b<sub>u</sub>t at lower throu<sub>g</sub>h<sub>p</sub>ut with res<sub>p</sub>ect to standard al<sub>g</sub>orithms [8], it will be more $C O _ { 2 }$ -intensive <sub>p</sub>er unit of data <sub>p</sub>rocessed than non-ML al<sub>g</sub>orithms. While the team is workin<sub>g</sub> on im<sub>p</sub>rovin<sub>g</sub> the throu<sub>g</sub>h<sub>p</sub>ut<sub>,</sub> it will still be worth examinin<sub>g</sub> whether the com<sub>p</sub>ression <sub>g</sub>ains of ML-based approac<sup>h</sup>es justi<sup>f</sup>y t<sup>h</sup>e a<sup>dd</sup>itiona<sup>l</sup> environmenta<sup>l</sup> cost in <sup>d</sup>ep<sup>l</sup>oyment <sub>a</sub>t <sub>sca</sub>l<sub>e.</sub>

## Appendix: technical configuration

I<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng,</sub> <sub>we</sub> d<sub>e</sub>t<sub>a</sub>il th<sub>e</sub> <sub>compu</sub>ti<sub>ng</sub> h<sub>ar</sub>d<sub>ware</sub> <sub>an</sub>d <sub>so</sub>ft<sub>ware</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>on</sub> <sub>use</sub>d t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>ese</sub> <sub>resu</sub>lt<sub>s,</sub> <sub>as</sub> <sub>we</sub>ll <sub>as</sub> th<sub>e</sub> <sub>se</sub>t<sub>up</sub> f<sub>or</sub> the estimate of HDD and ta<sub>p</sub>e $C O _ { 2 }$ -e<sub>q</sub>uivalent.

## ML data compression

Th<sub>e</sub> BOA ML <sub>co</sub>m<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n <sub>a</sub>l<sub>go</sub>rithm <sub>was</sub> tr<sub>a</sub>in<sub>e</sub>d <sub>a</sub>nd <sub>e</sub>x<sub>ecu</sub>t<sub>e</sub>d <sub>o</sub>n <sub>a</sub> G<sub>oog</sub>l<sub>e</sub> C<sub>o</sub>l<sub>a</sub>b n<sub>o</sub>t<sub>e</sub>b<sub>oo</sub>k <sub>w</sub>ith <sub>a</sub> r<sub>u</sub>ntim<sub>e</sub> N<sub>v</sub>idi<sub>a</sub> T4 GPU<sub>.</sub> Th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l <sub>c</sub>h<sub>osen</sub> <sub>was</sub> <sub>a</sub> t<sub>wo-</sub>l<sub>ayer</sub> M<sub>am</sub>b<sub>a-v</sub>1 b<sub>ac</sub>kb<sub>one</sub> <sub>w</sub>ith <sub>a</sub> hidd<sub>en</sub> di<sub>men-</sub> <sub>s</sub>i<sub>on</sub> <sub>o</sub>f 64<sub>,</sub> <sub>a</sub> b<sub>y</sub>t<sub>e</sub> <sub>voca</sub>b<sub>u</sub>l<sub>ary</sub> <sub>o</sub>f 256 <sub>sym</sub>b<sub>o</sub>l<sub>s,</sub> <sub>a</sub> <sub>sequence</sub> l<sub>eng</sub>th <sub>o</sub>f 10<sub>,</sub>000 b<sub>y</sub>t<sub>es an</sub>d <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>i<sub>ze o</sub>f 5<sub>.</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng was per</sub>f<sub>orme</sub>d f<sub>or e</sub>i<sub>g</sub>ht e<sub>p</sub>ochs in FP32 <sub>p</sub>recision<sub>,</sub> <sub>u</sub>sin<sub>g</sub> a learnin<sub>g</sub> rate of $5 \times 1 0 ^ { - 4 }$ <sub>an</sub>d <sub>a</sub> random seed of 42. The trainin<sub>g</sub> ste<sub>p</sub> included the ei<sub>g</sub>ht trainin<sub>g</sub> <sub>epoc</sub>h<sub>s,</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on,</sub> fi<sub>na</sub>l t<sub>es</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>an</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t <sub>wr</sub>iti<sub>ng.</sub> The com<sub>p</sub>ression ste<sub>p</sub> included the execution of one com<sub>p</sub>ression <sub>an</sub>d d<sub>ecompress</sub>i<sub>on</sub> <sub>roun</sub>d<sub>-</sub>t<sub>r</sub>i<sub>p.</sub> Th<sub>e</sub> fil<sub>e</sub> th<sub>a</sub>t <sub>was</sub> <sub>compresse</sub>d <sub>was</sub> the 49.92 MB "bundled CMS file" (CMS\_DATA\_float32.bin) from [8] that can be found within the BOA GitHub re<sub>p</sub>ositor<sub>y</sub> [7]; each d<sub>a</sub>t<sub>a recor</sub>d <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> 24 <sub>float</sub>32 f<sub>ea</sub>t<sub>ures.</sub> Th<sub>e resu</sub>lt<sub>s</sub> f<sub>rom</sub> thi<sub>s</sub> fil<sub>e</sub> <sub>were sca</sub>l<sub>e</sub>d t<sub>o</sub> th<sub>e max</sub>i<sub>mum</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>s</sub>i<sub>ze s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 1 <sub>a</sub>ft<sub>er</sub> checkin<sub>g</sub> linearit<sub>y</sub> of trainin<sub>g</sub> and inference carbon costs.

Carbon trackin<sub>g</sub> was <sub>p</sub>erformed usin<sub>g</sub> CodeCarbon [6], followin<sub>g</sub> a tutorial in [9]. CodeCarbon was used in one-second <sub>p</sub>owersampling intervals using machine mode to measure the energy <sub>co</sub>n<sub>su</sub>m<sub>p</sub>ti<sub>o</sub>n <sub>o</sub>f CPU<sub>,</sub> GPU <sub>a</sub>nd RAM <sub>use</sub>d in th<sub>e</sub> ML-b<sub>ase</sub>d <sub>co</sub>m <sub>p</sub>ression trainin<sub>g</sub> and inference. The same setu<sub>p</sub> was used for the <sub>co</sub>m<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n-d<sub>eco</sub>m<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n r<sub>ou</sub>nd tri<sub>p us</sub>in<sub>g</sub> ZSTD <sub>a</sub>nd LZMA<sub>.</sub> W<sub>e</sub> are aware that machine mode measures the energy consumption of the hardware stack of GPU, CPU and RAM (which ma<sub>y</sub> as well be shared when usin<sub>g</sub> a Goo<sub>g</sub>le Colab notebook), but it is the onl<sub>y</sub> wa<sub>y</sub> t<sub>o o</sub>bt<sub>a</sub>in <sub>a</sub> GPU <sub>e</sub>n<sub>e</sub>r<sub>gy es</sub>tim<sub>a</sub>t<sub>e us</sub>in<sub>g</sub> C<sub>o</sub>d<sub>e</sub>C<sub>a</sub>rb<sub>o</sub>n<sub>.</sub> Th<sub>e</sub>rm<sub>a</sub>l D<sub>e</sub>- si<sub>g</sub>n Power (TDP) scalin<sub>g</sub> for an Intel(R) Xeon(R) CPU @ 2.00GHz was used for the CPU <sub>p</sub>ower consum<sub>p</sub>tion corres<sub>p</sub>ondin<sub>g</sub> to 8W<sub>,</sub> while the GPU power consumption used the nvidia-ml-py package.

The conversion between ener<sub>gy</sub> use and carbon e<sub>q</sub>uivalent was <sub>p</sub>erformed usin<sub>g</sub> CodeCarbon’s national ener<sub>gy</sub> mix avera<sub>g</sub>ed scenarios [5].

## Storage servers

Th<sub>e</sub> <sub>car</sub>b<sub>on</sub> <sub>cos</sub>t i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d b<sub>o</sub>th <sub>em</sub>b<sub>o</sub>di<sub>e</sub>d <sub>an</sub>d <sub>opera</sub>ti<sub>ona</sub>l <sub>car</sub>b<sub>on</sub> [11, 16].

For the stora<sub>g</sub>e of D GB o<sub>v</sub>er T <sub>y</sub>ears on a medi<sub>u</sub>m m<sub>,</sub> the total stora<sub>g</sub>e carbon cost is

$$
C _ { m } ( D , T ) = C _ { m , \mathrm { e m b o d i e d } } ( D ) + C _ { m , \mathrm { o p e r a t i o n a l } } ( D , T ) .\tag{1}
$$

For HDD stora<sub>g</sub>e<sub>,</sub> a dri<sub>v</sub>e <sub>w</sub>ith <sub>u</sub>sable ca<sub>p</sub>acit<sub>y</sub> $K _ { H D D }$ i<sub>s</sub> th<sub>e</sub> b<sub>as</sub>i<sub>c</sub> h<sub>ar</sub>d<sub>ware</sub> <sub>un</sub>it<sub>.</sub> Th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>powere</sub>d d<sub>r</sub>i<sub>ves</sub> <sub>requ</sub>i<sub>re</sub>d t<sub>o</sub> <sub>s</sub>t<sub>ore</sub> D GB i<sub>s</sub>

$$
N _ { \mathrm { H D D } } ( D ) = \operatorname* { m a x } \left( N _ { \mathrm { H D D , m i n } } , \left\lceil \frac { D } { K _ { \mathrm { H D D } } } \right\rceil \right) ,\tag{2}
$$

<sub>w</sub>h<sub>ere we c</sub>h<sub>oose</sub> $N _ { \mathrm { H D D , m i n } } = 1 _ { \mathrm { : } }$ meanin<sub>g</sub> at least one drive is as-<sub>sume</sub>d t<sub>o</sub> <sub>rema</sub>i<sub>n</sub> i<sub>ns</sub>t<sub>a</sub>ll<sub>e</sub>d <sub>an</sub>d <sub>opera</sub>ti<sub>ona</sub>l<sub>.</sub> Gi<sub>ven</sub> th<sub>e</sub> li<sub>m</sub>it<sub>e</sub>d <sub>scope</sub> and total data volume of this study, the displaced carbon (non-use) for the HDD is allocated <sub>p</sub>ro<sub>p</sub>ortionall<sub>y</sub> to the stora<sub>g</sub>e ca<sub>p</sub>acit<sub>y</sub>

$$
C _ { \mathrm { H D D , n o n - u s e } } ( D ) = D e _ { \mathrm { H D D , n o n - u s e } } ,\tag{3}
$$

<sub>w</sub>h<sub>er</sub> $e e _ { \mathrm { H D D , n o n - u s e } }$ i<sub>s</sub> th<sub>e capac</sub>it<sub>y-a</sub>ll<sub>oca</sub>t<sub>e</sub>d di<sub>sp</sub>l<sub>ace</sub>d <sub>car</sub>b<sub>on</sub> i<sub>n</sub> $g C O _ { 2 } e / k W h$ Th<sub>e opera</sub>ti<sub>ona</sub>l <sub>car</sub>b<sub>on was ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>t<sub>eger num</sub>b<sub>er o</sub>f <sub>powere</sub>d d<sub>r</sub>i<sub>ves</sub>

$$
C _ { \mathrm { H D D , o p e r a t i o n a l } } ( D , T ) = N _ { \mathrm { H D D } } ( D ) \frac { P _ { \mathrm { H D D } } } { 1 0 0 0 } ( 2 4 ) ( 3 6 5 . 2 5 ) T I _ { \mathrm { H D D } } \mathrm { P U E } ,\tag{4}
$$

we ne<sub>g</sub>lect the Power Usa<sub>g</sub>e Efectiveness (PUE) throu<sub>g</sub>hout (and set it e<sub>q</sub>ual to unit<sub>y</sub>). So the total HDD carbon cost is the sum of these t<sub>wo</sub> <sub>qua</sub>ntiti<sub>es.</sub> Th<sub>e</sub> HDD <sub>sce</sub>n<sub>a</sub>ri<sub>o</sub> <sub>use</sub>d th<sub>e</sub> S<sub>eaga</sub>t<sub>e</sub> Ex<sub>os</sub> X18 18 TB drive. Its nominal ca<sub>p</sub>acit<sub>y</sub> and avera<sub>g</sub>e idle <sub>p</sub>ower were taken as 18 TB and 5.3 W, res<sub>p</sub>ectivel<sub>y</sub> [14]. The dis<sub>p</sub>laced $k g C O _ { 2 } e$ <sub>va</sub>l<sub>ue was</sub> t<sub>a</sub>k<sub>en as</sub> 27 $k g C O _ { 2 } e$ <sub>p</sub>er drive<sub>,</sub> corres<sub>p</sub>ondin<sub>g</sub> to a <sub>p</sub>ro<sub>p</sub>ortionalit<sub>y</sub> f<sub>ac</sub>t<sub>or o</sub>f 1<sub>.</sub>50 $g C O _ { 2 } e / G B$

For Ta<sub>p</sub>e stora<sub>g</sub>e<sub>,</sub> cartrid<sub>g</sub>e-related non-use carbon was calculated usin<sub>g</sub> an inte<sub>g</sub>er number of LTO-8 cartrid<sub>g</sub>es<sub>,</sub> while o<sub>p</sub>erati<sub>ona</sub>l <sub>car</sub>b<sub>on</sub> <sub>was</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>as</sub> <sub>propor</sub>ti<sub>ona</sub>l t<sub>o</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> RAL Ta<sub>p</sub>e-service factor in [11]. While inconsistent with the choice made for HDDs<sub>,</sub> this choice reflected existin<sub>g</sub> data availabilit<sub>y</sub>.

Similarl<sub>y</sub> to HDD<sub>,</sub> the n<sub>u</sub>mber of cartrid<sub>g</sub>es is

$$
N _ { \mathrm { T a p e } } ( D ) = \operatorname* { m a x } \left( 1 , \left\lceil \frac { D } { K _ { \mathrm { T a p e } } } \right\rceil \right) .\tag{5}
$$

Th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>car</sub>b<sub>on cos</sub>t i<sub>s ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>as</sub>

$$
C _ { \mathrm { T a p e } } ( D , T ) = N _ { \mathrm { T a p e } } ( D ) E _ { \mathrm { T a p e , n o n - u s e } } ^ { \mathrm { c a r t r i d g e } } + \frac { D } { 1 0 0 0 } T e _ { \mathrm { T a p e , R A L } } ,\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> $E _ { \mathrm { T a p e , n o n - u s e } } ^ { \mathrm { c a r t r i d g e } }$ is the non-use lifec<sub>y</sub>cle <sub>p</sub>rox<sub>y</sub> for one cartrid<sub>g</sub>e in $g C O _ { 2 } e / c a r t r i d g e$ <sub>,</sub> <sub>an</sub>d $e _ { \mathrm { T a p e , R A I } }$ i<sub>s</sub> th<sub>e co</sub>m<sub>p</sub>l<sub>e</sub>t<sub>e</sub> RAL T<sub>ape</sub>-<sub>se</sub>r<sub>v</sub>i<sub>ce</sub> o<sub>p</sub>erational factor in $g C O _ { 2 } e / G B y e a r$ <sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>s car</sub>t<sub>r</sub>id<sub>ge-</sub> related displaced (non-use) carbon, while the second represents <sub>opera</sub>ti<sub>ona</sub>l <sub>em</sub>i<sub>ss</sub>i<sub>ons a</sub>ll<sub>oca</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e s</sub>t<sub>ore</sub>d d<sub>a</sub>t<sub>a vo</sub>l<sub>ume.</sub> Th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l <sub>use</sub>d <sub>a</sub>n LTO-8 <sub>ca</sub>rtrid<sub>ge w</sub>ith <sub>a</sub> n<sub>a</sub>ti<sub>ve capac</sub>it<sub>y o</sub>f 12 TB<sub>.</sub> Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> lif<sub>ecyc</sub>l<sub>e</sub> d<sub>a</sub>t<sub>a repor</sub>t <sub>a</sub> t<sub>o</sub>t<sub>a</sub>l <sub>va</sub>l<sub>ue o</sub>f $1 3 . 7 2 k g C O _ { 2 } e$ <sub>per car</sub>t<sub>r</sub>id<sub>ge, o</sub>f <sub>w</sub>hi<sub>c</sub>h 5<sub>.</sub>70 $k g C O _ { 2 } e$ is assi<sub>g</sub>ned to the o<sub>p</sub>erational <sub>p</sub>hase [10, 11].

$$
E _ { \mathrm { T a p e , n o n - u s e } } ^ { \mathrm { c a r t r i d g e } } = E _ { \mathrm { T a p e , t o t a l } } ^ { \mathrm { c a r t r i d g e } } - E _ { \mathrm { T a p e , u s e } } ^ { \mathrm { c a r t r i d g e } } .\tag{7}
$$

Here, $E _ { \mathrm { T a p e , t o t a l } } ^ { \mathrm { c a r t r i d g e } }$ i<sub>s</sub> th<sub>e repor</sub>t<sub>e</sub>d t<sub>o</sub>t<sub>a</sub>l lif<sub>ecyc</sub>l<sub>e car</sub>b<sub>on em</sub>i<sub>ss</sub>i<sub>on o</sub>f one ta<sub>p</sub>e cartrid<sub>g</sub>e<sub>,</sub> and $E _ { \mathrm { T a p e , u s e } } ^ { \mathrm { c a r t r i d g e } }$ is the corres<sub>p</sub>ondin<sub>g</sub> use-<sub>p</sub>hase contribution. Usin<sub>g</sub> the literature values of 13.72 k<sub>g</sub>CO<sub>2</sub>e/cartrid<sub>g</sub>e and 5.70 k<sub>g</sub>CO<sub>2</sub>e/cartrid<sub>g</sub>e, res<sub>p</sub>ectivel<sub>y</sub> [10, 11], <sub>g</sub>ives

$$
E _ { \mathrm { T a p e , n o n - u s e } } ^ { \mathrm { c a r t r i d g e } } = 1 3 . 7 2 - 5 . 7 0 = 8 . 0 2 \mathrm { k g } \mathrm { C O _ { 2 } e } / \mathrm { c a r t r i d g e } .\tag{8}
$$

## Acknowledgments

This work was funded by EPSRC Studentship EP/W524347/1 (Project 2932638) via MADSIM CDT, the Universit<sub>y</sub> of Manchester Dame Kat<sup>hl</sup>een O<sup>ll</sup>erens<sup>h</sup>aw Fe<sup>ll</sup>ows<sup>h</sup>ip, an<sup>d</sup> is part o<sup>f</sup> a project t<sup>h</sup>at <sup>h</sup>as <sub>rece</sub>i<sub>ve</sub>d f<sub>un</sub>di<sub>ng</sub> f<sub>rom</sub> th<sub>e</sub> E<sub>uropean</sub> R<sub>esearc</sub>h C<sub>ounc</sub>il <sub>un</sub>d<sub>er</sub> th<sub>e</sub> E<sub>u</sub>r<sub>opea</sub>n Uni<sub>o</sub>n’<sub>s</sub> H<sub>o</sub>riz<sub>o</sub>n 2020 r<sub>esea</sub>r<sub>c</sub>h <sub>a</sub>nd inn<sub>ova</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>og</sub>r<sub>a</sub>m (<sub>g</sub>rant a<sub>g</sub>reement 101002463). Summer student fundin<sub>g</sub> was also <sub>p</sub>rovided b<sub>y</sub> the N8CIR summer internshi<sub>p</sub> scheme.

## References

[1] 2022. ATLAS Software and Computing HL-LHC Roadmap. Technical Report. CERN<sub>,</sub> Geneva. htt<sub>p</sub>s://cds.cern.ch/record/2802918

[2] CERN. 2017. Large Hadron Collider FAQ. Technical Report. CERN, Geneva. htt<sub>p</sub>s://cds.cern.ch/record/2255762

[3] Tristan Coi<sub>g</sub>nion, Clément Quinton, and Romain Rouvo<sub>y</sub>. 2025. When Faster Isn’t Greener: The Hidden Costs of LLM-Based Code Optimization. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE) (Seoul, Korea, Re<sub>p</sub>ublic of). IEEE Press, 1655–1666. doi:10.1109/ASE63991.2025. 00139

[4] Yann Collet and Murray Kucherawy. 2021. Zstandard Compression and the ‘application/zstd’ Media Type. RFC 8878. IETF. doi:10.17487/RFC8878

[5] Benoît Court<sub>y</sub> et al. 2023. Global electricit<sub>y</sub> carbon-intensit<sub>y</sub> data for <sub>p</sub>rivate infrastructure. Data file in the CodeCarbon source re<sub>p</sub>ositor<sub>y</sub>. htt<sub>p</sub>s://<sub>g</sub>ithub.c <sub>om</sub>/<sub>m</sub>l<sub>co</sub>2/<sub>co</sub>d<sub>ecar</sub>b<sub>on</sub>/bl<sub>o</sub>b/<sub>mas</sub>t<sub>er</sub>/<sub>co</sub>d<sub>ecar</sub>b<sub>on</sub>/d<sub>a</sub>t<sub>a</sub>/<sub>pr</sub>i<sub>va</sub>t<sub>e\_</sub>i<sub>n</sub>f<sub>ra</sub>/<sub>g</sub>l<sub>o</sub>b<sub>a</sub>l<sub>\_ene</sub> rgy\_mix.json Country-average <sup>f</sup>actors use<sup>d f</sup>or regiona<sup>l</sup> scenarios; accesse<sup>d</sup> 5 <sup>A</sup>u<sub>g</sub>ust 2026.

[6] Benoît Courty et al. 2026. mlco2/codecarbon: v3.3.0. doi:10.5281/zenodo.21791294

[7] Akshat Gupta, Caterina Doglioni, and Thomas Elliott. 2026. Boa Constrictor: A Mamba-based Lossless Compressor for Scientific Data. Technical Report. Zenodo. d<sub>o</sub>i<sub>:</sub>10<sub>.</sub>5281/<sub>zeno</sub>d<sub>o.</sub>1848192

[8] Akshat Gu<sub>p</sub>ta, Caterina Do<sub>g</sub>lioni, and Thomas J Elliott. 2026. BOA constrictor: a Mamba-based lossless compressor for scientific data. Machine Learning: Science and Technology 7, 3 (may 2026), 035014. doi:10.1088/2632-2153/ae64a9

[9] Melanie Hanna and Andrés Feli<sub>p</sub>e Perez Murcia. 2026. Reducin<sub>g y</sub>our Climate Im<sub>p</sub>act when Trainin<sub>g</sub> ML Models. htt<sub>p</sub>s://<sub>g</sub>ithub.com/climatechan<sub>g</sub>e-aitutorials/trackin -ml-emissions. doi:10.5281/zenodo.21828628

[10] Brad Johns. 2021. Improving Information Technology Sustainability with Modern Tape Storage. Technical Report. Brad Johns Consulting and Fujifilm. https:

//asset.<sup>f</sup>uji<sup>fil</sup>m.com/www/<sup>d</sup>e/<sup>f</sup>i<sup>l</sup>es/2023-10/97<sup>dd</sup>c3473883421ce<sup>f</sup>1<sup>fb</sup>820<sup>d</sup>236<sup>df</sup>a 2/Im<sub>p</sub>rovin<sub>g</sub>\_IT\_Sustainabilit<sub>y</sub>\_with\_Ta<sub>p</sub>e\_BJC\_0.<sub>p</sub>df Lifec<sub>y</sub>cle assessment used for the LTO-8 cartrid<sub>g</sub>e <sub>p</sub>rox<sub>y;</sub> accessed 5 A<sub>ugu</sub>st 2026.

[11] Alison Packer and Samuel Cadellin Ski<sub>p</sub>se<sub>y</sub>. 2025. Carbon costs of stora<sub>g</sub>e: A U.K. perspective. EPJ Web of Conferences 337 (2025), 01157. doi:10.1051/epjconf/ 202533701157

[12] I<sub>g</sub>or Pavlov. 2007. LZMA SDK (Software Develo<sub>p</sub>ment Kit). htt<sub>p</sub>s://7-zi<sub>p</sub>.or<sub>g</sub>/sd k<sub>.</sub>ht<sub>m</sub>l

[13] Anna M. M. Scaife. 2020. Bi<sub>g</sub> telesco<sub>p</sub>e, bi<sub>g</sub> data: towards exascale with the S<sub>q</sub>uare Kilometre Array. Philosophical Transactions ofthe Royal Society A: Mathematical, Physical and Engineering Sciences 378, 2166 (2020), 20190060. doi:10.1098/rsta.201 9.0060

[14] Seagate Technology. 2023. Exos X18 18TB Product Sustainability Report. Technical Re<sub>p</sub>ort. Sea<sub>g</sub>ate Technolo<sub>gy</sub>. htt<sub>p</sub>s://www.sea<sub>g</sub>ate.com/content/dam/sea<sub>g</sub>at e/assets/es<sub>g</sub>/<sub>p</sub>lanet/<sub>p</sub>roduct-sustainabilit<sub>y</sub>/ima<sub>g</sub>es/exos-x18-sustainabilit<sub>y</sub>- re<sub>p</sub>ort/files/Exos-X18-18TB-Sustainabilit<sub>y</sub>-Re<sub>p</sub>ort-2023.<sub>p</sub>df Accessed 5 Au<sub>g</sub>ust 2026.

[15] Swamit Tannu and Prashant J. Nair. 2023. The Dirt<sub>y</sub> Secret of SSDs: Embodied Carbon. ACM SIGEnergy Energy Informatics Review 3, 3 (Oct. 2023), 4–9. doi:10.1 145/3630614.3630616

[16] Mattias Wadenstein and Wim Vanderbauwhede. 2025. Life c<sub>y</sub>cle anal<sub>y</sub>sis for emissions of scientific computing centres. Eur. Phys. J. C 85, 8 (2025), 913. arXiv:2506.14365 [he<sub>p</sub>-ex] doi:10.1140/e<sub>p</sub>jc/s10052-025-14650-8