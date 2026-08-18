Graphical Abstract

Predicting, Evaluating, and Explaining Top Misinformation Spreaders via Archetypal User Behavior

Enrico Verdolotti, Luca Luceri, Silvia Giordano

![](images/7a33c7777b7e4342af80e63ce71ff97d47607791a539cae37677644d2b093b50.jpg)  
MISINFORMATION SPREAD  
METHODS DEFINITIONVALIDATION & COMPARISON

## Highlights

Predicting, Evaluating, and Explaining Top Misinformation Spreaders via Archetypal User Behavior

Enrico Verdolotti, Luca Luceri, Silvia Giordano

• Formalization and refinement of user behavioral archetypes: amplifiers, super-spreaders, and coordinated accounts.

• Development, evaluation, and comparison of several methods based on distinct archetypes, and their combination, for predicting top misinformation spreaders.

• Development of a methodology based on explainable AI to provide insights on the diverse behavioral traits of the actors mainly responsible for misinformation propagation.

# Predicting, Evaluating, and Explaining Top Misinformation Spreaders via Archetypal User Behavior

Enrico Verdolotti<sup>a</sup>, Luca Luceri<sup>b</sup>, Silvia Giordano<sup>a</sup>

<sup>a</sup>ISIN - DTI, SUPSI, Lugano, Switzerland

<sup>b</sup>USC Information Sciences Institute, Marina del Rey, California, USA

## Abstract

The spread of misinformation on social networks poses a significant challenge to online communities and society at large. Not all users contribute equally to this phenomenon: a small number of highly efective individuals can exert outsized influence, amplifying false narratives and contributing to significant societal harm. This paper seeks to mitigate the spread of misinformation by enabling proactive interventions, identifying and ranking users according to key behavioral indicators associated with harmful content dissemination. We examine three user archetypes—amplifiers, super-spreaders, and coordinated accounts—each characterized by distinct behavioral patterns in the dissemination of misinformation. These are not mutually exclusive, and individual users may exhibit characteristics of multiple archetypes. We develop and evaluate several user ranking models, each aligned with a specific archetype, and find that super-spreader traits consistently dominate the top ranks among the most influential misinformation spreaders. As we move down the ranking, however, the interplay of multiple archetypes becomes more prominent. Additionally, we demonstrate the critical role of temporal dynamics in predictive performance, and introduce methods that reduce data requirements by minimizing the observation window needed for accurate forecasting. Finally, we demonstrate the utility and benefits of explainable AI (XAI) techniques, integrating multiple archetypal traits into a unified model

to enhance interpretability and ofer deeper insight into the key factors driving misinformation propagation. Our findings provide actionable tools for identifying potentially harmful users and guiding content moderation strategies, enabling platforms to monitor accounts of concern more efectively.

Keywords: social media, misinformation, prediction, prevention, user behavior, content moderation, network analysis, early warning

## 1. Introduction

Social media platforms have transformed the way information circulates, enabling billions of users to create, share, and consume content on an unprecedented scale. This interconnected digital environment encourages open communication and civic engagement but also facilitates the rapid spread of misleading claims and misinformation [1]. During the COVID-19 pandemic, for example, false and low-credibility content proliferated across platforms [2, 3, 4], shaping public perception and leading to harmful behaviors [5, 6, 7, 8]. The consequences of such misinformation are far-reaching: it can distort individual and collective decision-making, polarize public discourse, and erode trust in science, media, and democratic institutions [9, 10].

Despite growing awareness of the dangers posed by misinformation, most moderation strategies employed by social media platforms remain fundamentally reactive. Interventions are often triggered only after a post has been flagged by users, detected by automated filters, or reported by fact-checkers, typically well after the content has begun to circulate widely [11]. By the time action is taken, such as labeling, demotion, or removal, the harmful information may have already reached large audiences or shaped public opinion [12, 13]. This lag in response significantly constrains the efectiveness of moderation eforts, as it allows false or misleading narratives to gain legitimacy, exploit algorithmic amplification [14], and embed themselves in the public consciousness. Moreover, the viral nature of misinformation means that even short delays can have outsized consequences. Studies have shown that false content often spreads faster and more broadly than factual corrections, which struggle to catch up once a narrative has taken root [13, 15, 16].

Bridging this gap calls for proactive, scalable solutions that can detect and counteract misinformation before it gains widespread traction. In this context, a pressing research question arises: Can we estimate the potential of an account to disseminate misinformation based on specific behavioral traits—or, put diferently, does an account’s behavioral footprint reveal latent signals of misinformation risk? Exploring this question could pave the way for early-warning systems that identify high-risk users before false content circulates broadly. By enabling timely, targeted interventions, such approaches have the potential to significantly enhance moderation eforts and reduce the overall harm caused by misinformation proliferation. To explore this research question, we examine three key behavioral archetypes that may characterize accounts responsible for the dissemination of misinformation: (i) Amplifiers: These accounts generate little to no original content but significantly boost the reach of information by re-sharing posts from others [17]. Their defining trait is a high frequency of re-sharing activity. (ii) Super-spreaders: Highly influential users whose original content consistently goes viral [18, 19]. They are characterized by the volume (and virality) of the content they create and the high levels of engagement they receive from other users. (iii) Coordinated accounts: Groups of users who collaborate to promote specific narratives by artificially amplifying targeted content [20, 21]. These accounts typically exhibit synchronized posting behavior and strong behavioral similarity through their orchestrated actions [22, 23].

To support more efective mitigation of misinformation, we build on these archetypes by developing and evaluating several user ranking systems that assess users based on their potential to spread misinformation. Each system is centered on a specific behavioral trait corresponding to one of the archetypes. In addition, we introduce a machine learning–based ranking framework that integrates all three archetypes by using their associated traits as input features. This approach serves two primary objectives: first, to evaluate whether integrating multiple behavioral signals improves predictive performance; and second, to apply explainable AI techniques to determine which traits most significantly influence a user’s ranking.

Our results reveal that the super-spreader archetype dominates the highestranked positions—those most responsible for misinformation dissemination— showing that a small number of content creators play a disproportionate role in the spread of false information, in line with previous findings [4, 18, 19].

In contrast, at middle and lower ranks, multiple archetypal traits frequently co-occur within individual accounts involved in the spread of misinformation. For example, resharing behavior—a defining characteristic of amplifier accounts—becomes increasingly prevalent further down the ranking. This trend suggests that while super-spreaders are instrumental in initiating virality, amplifiers and coordinated accounts play a substantial role in sustaining and extending the reach of misinformation across the network.

This paper ofers three key contributions. First, it formalizes and refines the behavioral archetypes of misinformation actors, addressing a critical gap in the literature where terms like amplifiers remain loosely defined, and research has predominantly focused on super-spreaders and coordinated accounts [20, 24, 25, 26, 27]. Second, it underscores the importance of temporal dynamics in misinformation difusion and introduces the Time-Aware Social H-index (TASH-Index)—a novel, computationally eficient ranking method that performs comparably to more sophisticated approaches while requiring fewer resources. Third, it presents a machine learning-based methodology that integrates features from multiple archetypes and leverages explainable AI techniques to illuminate the behavioral signals driving misinformation propagation. This approach provides a more nuanced understanding of how diferent user behaviors—beyond content generation—contribute to the spread of low-credibility content in networked environments.

Together, these contributions provide actionable tools for identifying potentially harmful users and informing content moderation practices. By surfacing influential spreaders early and shedding light on their behavioral profiles, our approach enables more eficient monitoring and prioritization of accounts of concern. The integration of explainable AI further enhances transparency and interpretability, ofering a foundation for future interdisciplinary research that bridges computational methods and social science to better understand and mitigate the spread of misinformation at scale.

## 2. Related Work

## 2.1. Misinformation Detection: Content vs. User-centered Approaches

The study of misinformation detection has evolved along two main perspectives: content-centered and user-centered. These approaches difer in their conceptual framing of the problem and in the methodologies they employ. This section reviews both perspectives, highlighting their core contributions and limitations, and positioning our work within this landscape.

Content-centered research focuses on analyzing the characteristics of the information being shared. It seeks to answer questions such as: Is this content misinformation? What features make it likely to go viral? and Which modalities contribute to its difusion? Early methods relied on lexical analysis, sentiment detection, and simple keyword-based heuristics. More recent advances use deep learning models, including LSTMs, attention mechanisms, and transformers, to detect linguistic or semantic patterns indicative of misinformation [28, 29, 30]. In multimodal settings, researchers have combined textual and visual signals to capture richer representations of content virality [31]. Recently, the difusion of AI-generated content has drawn comparisons to the spread of misinformation, revealing notable similarities that warrant attention [26]. In both cases, a small number of highly visible actors, such as verified accounts and automated bots, are responsible for a disproportionately large share of the content’s propagation.

While content-centered approaches sometimes incorporate user-related features, such as the identity of the author or engagement metrics, the main analytical focus remains on the content itself. Consequently, they often overlook the role of user behavior and network dynamics in shaping difusion patterns. In contrast, user-centered approaches emphasize the behaviors, attributes, and network positions of individuals participating in the spread of misinformation. These studies focus on identifying key actors, analyzing activity patterns, and modeling interactions within social networks. Rather than treating users as auxiliary features, user-centered models seek to explain difusion phenomena through user behavior itself.

Our work builds on this user-centered perspective, with the objective of modeling misinformation difusion as the emergent outcome of user activity patterns. We propose a framework that focuses on behavioral archetypes rather than content features. This represents a key contribution of our work and will be explored in greater depth in the following sections.

## 2.2. User-Centered Approaches for Misinformation Detection

Research grounded in the user-centered paradigm has examined a wide range of behavioral dimensions to better understand misinformation dynamics. Prior studies have analyzed temporal activity patterns [32], network centrality and influence metrics [33], and exposure patterns linked to misinformation risk [34]. Other studies have investigated ideological alignment, hate speech, network exposure, and psychological traits, such as cognitive reflection and susceptibility to influence, as key factors underlying the adoption of misinformation [35, 36, 37]. For example, Ye et al. [36] found that social media users require fewer exposures to adopt low-credibility content compared to high-credibility content.

From a methodological standpoint, several eforts have leveraged graph neural networks (GNNs) to capture relational structures in social media data. Rath et al. [38] introduced an inductive GNN-based approach for detecting fake news spreaders, which was later extended with attention mechanisms and explanation modules in SCARLET [39]. Sakketou et al. [40] incorporated temporal dynamics into GNNs, enabling forecasts based on evolving behavior. Ullah et al. [41] benchmarked diferent GNN architectures, discussing their trade-ofs between expressiveness and scalability. Minici et al. [42] leveraged a GNN to model similarity networks based on users’ sharing activity and detect coordinated accounts driving information operations.

Beyond GNN-based approaches, research in the misinformation domain has advanced along three main directions, each targeting a distinct objective to deepen our understanding of misinformation dynamics: (i) identifying active spreaders through heuristics or supervised models; (ii) predicting future spreaders using temporal and behavioral predictors; and (iii) characterizing user roles and strategies based on ideological, linguistic, or network features.

Despite these advances, several critical challenges remain. In particular, the distinct roles of user archetypes in misinformation difusion—especially the emergence and impact of amplifiers, who propagate but do not originate false content—remains underexplored. Additionally, user roles tend to be treated as static categories, despite evidence of dynamic shifts over time. Finally, as models grow more complex, issues of transparency and explainability become critical, especially when findings are intended to inform policy deci sions. While explainable AI (XAI) techniques are emerging in this domain, their adoption in misinformation research is still limited.

Our approach contributes to this area of research in two ways. First, we introduce a formalization of three distinct behavioral archetypes that characterize users involved in the spread of misinformation. Second, we enhance a previously proposed influence metric—an H-index-based score introduced by [18]—by making it time-aware, thus capturing temporal patterns in user activity. This refined metric, along with other archetype-specific scores, is integrated into a machine learning framework designed to estimate a user’s potential to disseminate misinformation. By combining features derived from multiple archetypes, the model captures interdependencies between behavioral traits while maintaining interpretability. Leveraging SHAP-based explainability methods, our approach enables the characterization of individual users in terms of their dominant archetypal behaviors. This not only provides deeper insight into the dynamics of misinformation difusion but also ofers actionable evidence that may inform the development of mitigation strategies and platform governance policies.

## 3. Behavioral Archetypes: Definition and Characterization

In this study, we delineate three key archetypes of user behavior on social media: amplifiers, super-spreaders, and coordinated accounts. These archetypes are not mutually exclusive, as individual users may exhibit characteristics of multiple types. They encompass both human-operated and automated accounts, such as social bots [43, 44], and constitute the foundation for the ranking models introduced in the subsequent sections. We define each archetype below, based exclusively on its content difusion behavior:

• Amplifiers are users whose primary strategy for spreading information consists of extensive resharing. While the term “amplifier” is often used in the literature to describe broader platform-level phenomena [45, 46, 47], similarly to [17], we employ it here to characterize accounts that systematically amplify content by re-sharing posts, rather than by creating original information.

• Coordinated accounts represent groups of users that operate in collaborative fashion, often driving orchestrated campaigns. These accounts exhibit behaviors reminiscent of swarm-like coordination, frequently amplifying the same messages to elicit the illusion of public consensus. Coordination on social media has been widely documented [48, 49, 50], with evidence suggesting its employment in multiple influence campaigns aimed at manipulating public opinion and recommendation algorithms [20, 21, 51], even across platforms [52, 23].

• Super-spreaders are users who consistently generate and share original content that achieves viral reach through extensive re-sharing by others on social media platforms. These users play a critical role in initiating and fueling the difusion of narratives. For this reason, they are particularly influential in misinformation dynamics [18].

These three behavioral archetypes encapsulate distinct yet interrelated aspects of user activity in information spread. In the following sections, we formalize methodologies for identifying and ranking users who exhibit these behaviors in the difusion of misinformation. Our models assign each account a score reflecting its potential to propagate low-credibility content, enabling a structured approach to assessing online risks and uncovering its underlying drivers.

## 4. Methodology

In this section, we present the approach to establishing content credibility, and we introduce the ranking methodologies we adopt for analysis, including both archetype-based and machine learning models.

## 4.1. Credibility Assessment

To assess the credibility of social media content, we adopt a domain-based approach that infers credibility from the trustworthiness of URLs embedded in user posts. Following established practice [18], we consider only original posts and reshares, excluding self-reshares, replies, and quotes, as these do not reflect content endorsement or difusion. Posts are labeled using News-Guard’s<sup>1</sup> credibility scores, which assign trust ratings in the range [0–100] to news domains. This platform-agnostic method ensures consistency with prior work and supports scalable, interpretable credibility assessments. Following NewsGuard guidelines, we categorize posts linking to domains with a score of 39 or lower as low-credibility. Tweets without URLs or linking to unknown domains were excluded from the downstream analysis. In cases where tweets contained multiple URLs, each was treated as a separate entry, and the average domain score was assigned to the tweet.

Importantly, our ranking models operate solely on the assigned credibility of shared content, regardless of its modality. As a result, while we focus here on textual content for consistency with [18], the same approach naturally extends to other formats such as images, memes, or videos, provided that a credibility assessment is available. This inherent flexibility allows our methodology to adapt across media types without requiring changes to the core modeling pipeline.

## 4.2. Archetype-Based User Ranking

User activity within a social network ofers rich signals that can be used to evaluate and compare accounts. By analyzing this activity through the lens of specific behavioral archetypes, we can define targeted methods that assign a numerical value (or score) to each user, reflecting their role in the dissemination of misinformation. These scores are then used to generate user rankings, with each ranking capturing a diferent dimension of behavior based on the archetype being modeled (e.g., amplification, influence, coordination).

This section introduces a suite of archetype-based ranking methods, each designed to operationalize a specific archetypal trait. For each archetype, we formally define the corresponding ranking approach and explain the rationale behind its construction, providing a foundation for comparing how diferent user behaviors contribute to misinformation spread.

## 4.2.1. Amplifiers Ranking

This first section focuses on methods aimed at ranking amplifier accounts by analyzing their resharing behavior, which reflects their role in the amplification of misinformation. Since these accounts primarily engage in re-sharing content, we first employ a method that ranks users according to the total number of misinformation posts they reshare. We then introduce a more refined approach that also accounts for the timing of these reshares, giving greater weight to users who consistently reshare earlier in the difusion process. They are both defined as follows:

• Repost Count: This approach ranks users based on their total number of misinformation reshares during the observation period. Formally, we define the Repost Count score as:

$$
\mathrm { R e p o s t \ C o u n t \ } ( u ) = \sum _ { p \in \mathcal { P } _ { \mathrm { m i s i n f o } } } R _ { u } ( p )
$$

where $\mathcal { P } _ { \mathrm { m i s i n f o } }$ is the set of posts with credibility below our predefined threshold, and $R _ { u } ( p ) = 1$ if user u has reshared post $p ,$ otherwise $R _ { u } ( p ) = 0$ . While this approach ofers a simple way to rank users, it overlooks the timing of their reshares—specifically, how early they amplify misinformation within its difusion lifecycle.

• Early Reposter Index (EaR-Index): A more sophisticated approach ranks users based on how early they reshare low-credibility content generated by other users, considering that early re-sharing might indicate a pivotal role in information cascades. Users who systematically appear among the first to reshare viral misinformation posts receive a higher index. The EaR-index is defined as:

$$
\mathrm { E a r l y ~ R e p o s t e r ~ I n d e x } ~ ( u ) = \frac { 1 } { | \mathcal { P } _ { u } | } \sum _ { p \in \mathcal { P } _ { u } } \left( N _ { p } - r _ { u } ( p ) + 1 \right)
$$

where $\mathcal { P } _ { u } \subseteq \mathcal { P } _ { \operatorname* { m i s i n f o } }$ is the set of misinformation posts reshared by user u, $N _ { p }$ is the total number of reshares received by post $p ,$ and $r _ { u } ( p )$ is the ordinal position of user u in the chronologically-sorted sequence of reshares of post $p , \ { \mathrm { e . g . } }$ , if u was the third person to reshare p, then $r _ { u } ( p ) = 3$ . The scoring function assigns users higher values the earlier they appear in the re-sharing sequence, ensuring that those who reshare misinformation early—especially for posts that eventually receive many reshares—are ranked more prominently. The final EaR-index is computed as the average of these scores across all misinformation posts reshared by the user.

## 4.2.2. Coordinated Accounts Ranking

This section presents ranking methods based on coordinated activity, often modeled through behavioral similarity of user actions. The intuition behind this approach is that coordinated accounts tend to reshare, promote, and artificially amplify the same pieces of content, reflecting a shared strategy. Following the methodology described in the literature [21, 20], we construct a co-reshare network where nodes represent users, and undirected edges between users are weighted by the cosine similarity of their resharing activities. In line with prior work, analyzing and pruning the co-reshare similarity network enables the identification of users that play central roles in orchestrated campaigns. To operationalize this, we propose two user ranking methods, each based on a distinct pruning strategy of the similarity network:

• Coordination-Centrality: This method ranks users based on their eigenvector centrality within the co-reshare similarity network, following the rationale proposed by [21], which leverages user centrality in similarity networks to detect coordinated accounts involved in influence operations. A higher score indicates that user u is closely connected to other central—and potentially coordinated—users, suggesting their embeddedness within organized amplification structures.

• Coordination-Edge Weight: This method ranks users based on the maximum weight among all edges incident to them in the similarity network, following the intuition of [20], which suggests that coordinated actors can be uncovered based on the strength of their similarity. A higher score indicates stronger coordinated behavior with at least one other user.

## 4.2.3. Super-Spreaders Ranking

This section presents a suite of approaches for ranking super-spreaders, i.e., users who exert significant influence on social media by attracting engagement from others. Engagement can take various forms, including likes, comments, quotes, and reshares. Among these, we focus specifically on reshares, as they provide a more direct signal of content endorsement and dissemination [53]. Unlike comments or quotes, which may express disagreement, sarcasm, or criticism, reshares more reliably indicate alignment with the content and contribute directly to its amplification.

The following methods are used to assign scores to users and subsequently rank them based on their ability to spread content and elicit engagement from others. We categorize these methods into two groups: state-of-the-art approaches and our proposed methods:

• State-of-the-art methods, as proposed by DeVerna et al., [18], use the volume of retweets received as a proxy for user influence. Specifically, they introduce two approaches, namely:

– Influence Score: This method calculates the total number of reshares received by an account’s low-credibility posts, providing a direct measure of the user’s overall reach in spreading such content:

$$
{ \mathrm { I n f l u e n c e ~ S c o r e ~ } } ( u ) = \sum _ { p \in \mathcal { P } _ { u } } { \mathrm { R e p o s t s } } _ { p }
$$

where $\mathcal { P } _ { u }$ is the set of low-credibility posts created by user $u ,$ and Reposts is the number of reshares gathered by post $p .$

– H-Index: This method treats posts as publications and reshares as citations. The H-Index, h, for a user u is defined as the largest number of posts h that have received at least h reshares each. Formally:

$$
{ \mathrm { H } } { \mathrm { - I n d e x ~ } } ( u ) = \operatorname* { m a x } \{ h \mid u { \mathrm { ~ h a s ~ } } h { \mathrm { ~ p o s t s } } \in \mathcal { P } _ { u } { \mathrm { ~ w i t h ~ r e s h a r e ~ c o u n t } } \geq h \}
$$

This method ensures that a user’s influence is not determined by a single viral post but instead reflects a consistent ability to generate engagement.

However, these two methods compute user scores within a fixed, and often arbitrarily chosen, observation period, meaning that the resulting rankings may be influenced by both the length of the window and the frequency of user activity. This can introduce inconsistencies when comparing users with diferent posting patterns. Notably, the H-index ofers a more stable alternative, as it is less sensitive to temporal constraints: users must maintain consistent engagement over time to increase their H-index, making it more robust to variability.

• Our Proposed Time-Aware Methods: Since user activity varies over time, we partition the observation window into non-overlapping, contiguous time intervals and compute scores within each slot. To aggregate these values, we apply an exponential moving average, which smooths the scores across time while giving more weight to recent behavior. This approach balances historical influence with recent activity, capturing long-term trends while reducing the impact of short-term fluctuations. Based on this framework, we define two time-aware variants of the state-of-the-art methods:

– Time-Aware Influence Score, which applies the Influence Score method for each time slot. These per-slot scores are then smoothed over time using an exponential moving average, resulting in a final score that reflects both historical and recent influence in terms of content virality. The Time-Aware Influence (TAI) Score is defined as follows:

$$
\left\{ \begin{array} { l l } { \mathrm { T A I - S c o r e } \ ( u ) _ { t } = \alpha \cdot \mathrm { T A I - S c o r e } \ ( u ) _ { t - 1 } + ( 1 - \alpha ) \cdot \mathrm { I n f l u e n c e } \ \mathrm { S c o r e } \ ( u ) _ { t } } \\ { \mathrm { T A I - S c o r e } \ ( u ) _ { 0 } = \mathrm { I n f l u e n c e } \ \mathrm { S c o r e } \ ( u ) _ { 0 } } \end{array} \right.
$$

where α is the smoothing factor, and Influence Score (u)<sub>t</sub> is the Influence Score for user u at time slot t spanning δ days. Intuitively, a higher α assigns more weight to past influence, while a lower α emphasizes recent activity.

– Time-Aware Social H-Index: Similar to the TAI-Score, this method applies an exponential moving average to the H-index computed within each time slot. This produces a smoothed score that captures a user’s sustained ability to generate widely reshared content over time, balancing historical consistency and recent activity. The Time-Aware Social Hirsch (TASH) Index is defined as follows:

$$
\left\{ \begin{array} { l } { \mathrm { T A S H - I n d e x } \left( u \right) _ { t } = \alpha \cdot \mathrm { T A S H - I n d e x } \left( u \right) _ { t - 1 } + \left( 1 - \alpha \right) \cdot \mathrm { H - I n d e x } \left( u \right) _ { t } } \\ { \mathrm { T A S H - I n d e x } \left( u \right) _ { 0 } = \mathrm { H - I n d e x } \left( u \right) _ { 0 } } \end{array} \right.
$$

where H-Index $( u ) _ { t }$ is the user u’s H-Index as proposed by [18], computed at time slot t spanning δ days, while α controls the trade-of between long-term and short-term influence.

In conclusion, both time-aware methods require tuning two parameters: δ, the size of the time slots t in days, and α, the smoothing factor. When $\alpha = 1$ , the TASH-Index remains constant, equal to the H-index from the first observed time slot, efectively ignoring any updates over time. As α increases, it adjusts the balance between historical estimates and recent influence. When $\alpha = 0$ instead, we consider only the H-index for the user in the most recent time slot of size δ. To determine the best values for α and δ, we conduct a grid search using the normalized Discounted Cumulative Gain metric [54] to evaluate diferent ranking performances. Further details on the search space are provided in Appendix A.1.

## 4.3. Machine Learning-based Ranking

The ranking methods introduced thus far rely on distinct behavioral traits, each designed to capture a specific archetypal aspect of user behavior. Building on this foundation, we explore whether combining these traits can improve ranking accuracy. Moreover, by integrating multiple archetypal traits into a unified model, we aim to interpret the model’s decisions and gain deeper insights into the key factors driving misinformation spread.

Ranking items is a well-established task in machine learning, commonly referred to as learning to rank [55], and it aligns naturally with our objective.

To comprehensively evaluate the efectiveness of diferent approaches, we frame the user ranking problem as a regression task—also known as pointwise ranking [56, 57]. By modeling the ranking as a regression problem, we aim to predict each user’s future contribution to misinformation based on their historical behavior and extracted features. This formulation provides a systematic framework for assessing user influence and enables meaningful comparisons with alternative ranking methods in our evaluation.

The target variable for this regression task quantifies a user’s strength in the misinformation reshare network. Specifically, it is defined as the total number of reshares involving low-credibility content in which the user is engaged—either as the original author whose posts are amplified by others, or as a user who actively reshares such content. This metric captures both the user’s influence and their activity level in the propagation of misinformation, ofering a comprehensive indicator of their role in its difusion.

Each user is represented by a feature vector comprising behavioral indicators drawn from the archetypes previously introduced, including:

• Repost Count (Repost Count)

• Early Reposter Index (EaR-Index)

• Coordination-Centrality (Coord-Centrality)

• Coordination-Edge Weight (Coord-EdgeWeight)

• Time-Aware Influence Score (TAI-Score)

• Time-Aware Social H-Index (TASH-Index)

We train and test several of-the-shelf ML models, with Random Forest resulting in the most accurate one. Finally, we apply SHAP (SHapley Additive exPlanations) to interpret the model’s predictions by identifying the features that most significantly influence the output. This analysis not only highlights the behavioral traits driving misinformation spread but also helps surface the users who play the most critical roles in its difusion.

## 5. Evaluation

In this section, we first describe the data sources used in our study, detailing their structure and relevance to the analysis. We then present the evaluation methodology and define the metrics employed to assess the performance of the ranking models introduced in the previous section.

![](images/7b564b81481e074631eccd986db88d4a5f1c6139cbbc26b53208281f665c8c5b.jpg)

![](images/6784caea3fe0f0477436f62432d7cbe5e23a2765fe43ca851abf48ed64af6e6a.jpg)  
Figure 1: Credibility distribution for labeled retweets for each dataset under scrutiny: VaccinItaly (left) and COVID-19 Multilanguage (right). Credibility score is assigned to each tweet based on NewsGuard’s ratings, as described in Section 4.1

## 5.1. Datasets

This study relies on two datasets, both collected from Twitter during a period of heightened misinformation difusion related to the COVID-19 pandemic. To evaluate the consistency of our findings across diferent contexts, we selected datasets focused on the same topic but covering distinct geographic areas: one limited to a single country (Italy), and the other reflecting the global conversation—albeit with a notable U.S. bias, given the platform’s user base. These datasets are described as follows:

• VaccinItaly: This dataset, collected by Pierri et al., [58], contains COVID-19-related tweets from Italian-speaking accounts, focusing on discussions related to vaccines and public health from December 20, 2020, to October 22, 2021 (306 days). Considering only reshares that can be scored—i.e., those containing one or more URLs—the final dataset includes 371,586 retweets from 54,916 distinct original posts and comprising 51,962 unique users.

• COVID-19 Multilanguage: This larger multilingual dataset of tweets was collected during the pandemic [59] and spans 372 days, from November 1, 2020, to November 8, 2021. The dataset includes 1,118,697 reshares from 103,250 distinct original posts and comprises 129,507 unique users.

Figure 1 presents the distribution of credibility scores associated with shared URLs in the two datasets. Both plots capture the credibility landscape of online content circulated during the COVID-19 pandemic, reflecting diferent linguistic and geographic dynamics. In the VaccinItaly dataset (left panel), the distribution is bimodal, with clear peaks in the very low credibility range (0–10) and the very high range (90–100). However, the low-credibility peak is less sharp and more spread across the 0–40 range than in the Multilanguage dataset. The presence of a secondary concentration around credibility scores in the 70–80 range suggests a more varied credibility spectrum within the Italian conversation, rather than strict polarization. Still, the data reveal a tendency for users to share content from both low- and high-credibility sources, with relatively less activity in the mid-range (40–70). In the Multilanguage dataset (right panel), the distribution is more sharply skewed, with a dominant peak in the lowest credibility bracket (0–10) and a strong resurgence at the top end (90–100). Unlike the Italian dataset, there is less evidence of mid-credibility sharing, and the low-credibility activity is more concentrated around the extreme values. This indicates a more polarized global discourse, particularly driven by content from highly questionable or highly reputable domains, with fewer intermediate sources in between.

By leveraging two datasets that difer in scale, geographic focus, and linguistic diversity, we ensure that our evaluation captures variability in user behavior and information difusion dynamics across distinct ecosystems.

Data splitting: Observation vs. Evaluation Periods. We adopt a temporal split strategy to ensure that no information from the future leaks into model training. Specifically, the observation period is used as the training set—that is, all data the model can access—while the evaluation period serves as the test set, following a standard 80–20 proportion. This split is applied samplewise in chronological order, selecting the first 80% of resharing events in their natural temporal sequence. Such a split preserves temporal consistency and respects causality, ensuring that the model never uses future events to predict past behavior. Within the training portion, we apply a sliding-window approach to construct temporally valid validation sets, following the crossvalidation scheme described in Appendix C.1.

## 5.2. Network Dismantling

To evaluate the accuracy of distinct ranking strategies, we apply a network dismantling approach, ensuring a fair comparative analysis with previous work [18]. We construct a reshare network by modeling it as a weighted, directed graph where nodes represent users and directed edges represent reshare activities. The weight of each edge corresponds to the total number of distinct pieces of content reshared between two users. For instance, if an edge exists from user A to user B with weight 3, this indicates that B reshared three distinct pieces of content originally created by A.

To ensure consistency across platforms, we count reshares at the level of unique content. This decision accounts for diferences in how platforms handle resharing: some, like X (formerly Twitter), prevent users from resharing the same post multiple times, while others, such as Facebook, allow repeated sharing of the same content. As a result, the edge A <sup>3</sup>−→ B indicates that user B has reshared exactly three distinct pieces of content originally posted by user A, regardless of how many times each piece was shared. From this network, we construct a low-credibility reshare network by retaining only content with credibility scores below a predefined threshold (39, based on NewsGuard guidelines). This filtered network captures the dissemination patterns of misinformation among users over the analyzed time period.

The network dismantling process involves iteratively removing users from the low-credibility reshare network, based on a given ranking. Removing a node eliminates all its incident edges, efectively preventing the associated misinformation reshares. We quantify the impact of removing a specific user by summing the weights of all incident edges (both incoming and outgoing) and normalizing this value as a fraction of the total network weight.

While this method does not account for potential cascading efects—such as whether removing one user influences others’ behavior—it serves as a practical proxy for estimating the impact of targeted user moderation. Specifically, it quantifies the share of misinformation that would no longer circulate if certain users were deplatformed.

## 5.3. Evaluation Metrics

We assess the performance of the diferent ranking models using two complementary metrics: Quality@k, which we derived from the network dismantling procedure introduced in prior work [18] (and illustrated in Figure 2), and nDCG@k (normalized Discounted Cumulative Gain) [54]. Both metrics are computed by evaluating model-generated user rankings against the lowcredibility reshare network constructed from the test set, representing unseen misinformation dissemination patterns. The dismantling procedure shown in Figure 2 simulates the sequential removal of top-ranked users from the reshare network and measures the resulting drop in misinformation volume.

Misinformation re-post network: dismantling  
![](images/c28f6408b1a633aa20210f1338ea28fe9e76bc875bd9cffd3c32843eaee78056.jpg)  
Figure 2: Illustration of the network dismantling process. The figure illustrates how the removal of highly ranked users impacts the overall volume of misinformation. Each point on the curve corresponds to one user removed from the network, in descending order according to their ranking. The observed trend reveals that a small subset of users is responsible for a disproportionately large share of misinformation difusion.

The curve represents the optimal dismantling, obtained by ranking users in descending order according to ground truth from the evaluation (test) set—specifically, their total node strength in the low-credibility reshare network. Node strength is defined as the sum of a user’s incoming and outgoing reshare connections, reflecting both their role in amplifying others and being amplified. Each point on the curve corresponds to the removal of a user according to this ranking, highlighting the cumulative reduction in misinformation volume as top-ranked users are sequentially removed.

Quality@k estimates the proportion of misinformation that would be removed by moderating the top-k ranked users in the test set network. For example, Quality@5 represents the share of low-credibility content eliminated by suppressing the top 5 users in the ranking. This metric provides an intuitive measure of ranking efectiveness for practical moderation scenarios.<sup>2</sup> nDCG@k [54] is a standard information retrieval metric that evaluates ranked output quality by considering both item relevance and position, assigning higher scores when relevant items appear earlier in the ranking. In our context, an item correspond to a user and each user’s relevance is determined by their volume of misinformation dissemination during the test period. The truncated version focuses only on the top-k users in both predicted and ground-truth rankings, ensuring comparability with Quality@k.

These metrics provide complementary perspectives: Quality@k captures the potential moderation benefit of each method when applied to future misinformation networks, while nDCG@k reflects the consistency between predicted and actual user influence in misinformation spread during the test period.

Evaluation setup. Our models produce user rankings based on training data, and we measure their efectiveness by applying these rankings to dismantle the misinformation reshare network observed in the test period. This evaluation design reflects the practical scenario where moderation decisions must be made based on historical user behavior to prevent and mitigate future misinformation spread.

Unlike standard recommendation tasks with static item sets, our scenario involves a dynamic user base, which can lead to discrepancies between training and test sets. We address this by distinguishing two user categories:

• Users present in training but absent in test are assumed to contribute no future misinformation. They are retained in the predicted ranking and assigned a ground-truth relevance score of zero.

• Users appearing only in the test set are considered unrecoverable by the model, as they were unobserved during training. These users are excluded from evaluation.

This design ensures that predicted and ground-truth rankings are aligned, matching in length and referring to the same set of users. While this setup introduces a systematic downward bias by omitting unseen future users, it reflects a realistic modeling constraint: a model can only rank users it has previously observed.

## 6. Results

This section presents the performance of each model using the metrics previously outlined.

Special emphasis is placed on explainability, utilizing the SHAP technique to ofer insights into model decisions and feature contributions to the model outcome. In addition, we examine the influence of data scarcity on the performance of the best-performing methods. This analysis sheds light on the robustness and adaptability of the models under varying data availability scenarios.

## 6.1. Ranking Performance

This subsection presents a comparative analysis of the best-performing ranking methods for each user archetype, using the dismantling evaluation framework introduced in Section 5.2. A more extensive evaluation involving all the ranking methods is provided in Appendix A. Figure 3 illustrates the dismantling trajectories obtained by progressively removing users from the VaccinItaly dataset according to diferent ranking strategies. Each curve tracks the cumulative fraction of misinformation removed as more users are blocked, simulating a moderation intervention.

Figure 3 includes the best method per archetype, as well as an optimal dismantling track (star-marked curve) that serves as our ground truth. The objective is to evaluate the efectiveness of each method in mitigating misinformation and, in turn, to identify which user archetypes are most relevant or most reliably detected within the analyzed framework. In practical terms, we are most interested in the early portion of each curve: the left-hand side indicates how much misinformation is mitigated by removing only the topranked users. Conversely, the tail of the curve reflects diminishing returns and is less relevant for real-world scenarios. These removals assume complete blocking of users, without modeling the dynamic efects of moderation on user behavior or network structure. Nevertheless, they provide useful insights into the relative efectiveness of diferent approaches.

A key finding from this analysis is that the ML-based method and TASH (a method grounded in the super-spreader archetype) achieve performance comparable to the optimal ranking, particularly for the top-ranked users. Moreover, both methods consistently outperform alternative strategies in identifying the most influential actors in the spread of misinformation.

This can be attributed to how these rankings align with the structure of the reshare network: the most impactful nodes in this network are those with high node strength, defined as the sum of incoming and outgoing edge weights, which in turn indicate produced and received retweets. Nodes with high strength, therefore, indicate users who are highly efective at spreading content and receive substantial engagement from others. Therefore, ranking strategies that prioritize such properties, such as the volume of reshares received, naturally result in better performance.

Misinformation re-post network: dismantling  
![](images/0d7f6438ca1e168a59820efccb45c014b7bac1955fa95c813e429ba75d44f010.jpg)  
Figure 3: Performance of the best performing ranking methods for each archetype. Each marker represents a user removed from the misinformation reshare network of the VaccinItaly dataset. The x-axis represents the user’s rank, while the y-axis indicates the estimated reduction in misinformation achieved by removing that user.

Table 1 presents a comparative, quantitative evaluation of the methods presented in Figure 3, using the two key metrics introduced before: nDCG and Quality score. Overall, the ML model exhibits the best ranking performance, achieving the highest nDCG score (0.904) and outperforming all other methods, particularly at small and moderate intervention levels (k = 3 to 100). The TASH-Index, despite its relative simplicity, closely follows it in terms of ranking performance, showing strong results particularly in top rankings. Quality scores at these levels also indicate that ML may mitigate the largest share of misinformation with fewer interventions, followed by TASH-Index. This aligns with previous findings that show that a limited number of users (small k) is responsible for a large amount of misinformation [4, 19]. As the value of k increases, the TAI-Score emerges as particularly efective, attaining the highest Quality score (83.9%) at k = 1000, suggesting its potential strength for broad-scale interventions. In contrast, the H-Index and Influence Score consistently underperform across both evaluation metrics. This further underscores the importance of incorporating temporal dynamics. Such temporal awareness proves particularly efective in detecting users who contribute to the rapid spread of harmful content, making time-aware methods substantially more informative than static ones.

<table><tr><td rowspan="2">@k</td><td colspan="2">ML</td><td colspan="2">TASH-Index</td><td colspan="2">H-Index</td><td colspan="2">TAI-Score</td><td colspan="2">Influence Score</td></tr><tr><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td></tr><tr><td>3</td><td>0.963</td><td>34.0%</td><td>0.949</td><td>33.3%</td><td>0.892</td><td>32.6%</td><td>0.938</td><td>32.6%</td><td>0.892</td><td>32.6%</td></tr><tr><td>5</td><td>0.967</td><td>39.8%</td><td>0.912</td><td>35.7%</td><td>0.858</td><td>35.0%</td><td>0.901</td><td>35.0%</td><td>0.865</td><td>35.7%</td></tr><tr><td>10</td><td>0.939</td><td>45.8%</td><td>0.931</td><td>45.8%</td><td>0.832</td><td>40.4%</td><td>0.873</td><td>40.3%</td><td>0.824</td><td>40.4%</td></tr><tr><td>20</td><td>0.890</td><td>49.0%</td><td>0.906</td><td>52.8%</td><td>0.790</td><td>43.7%</td><td>0.848</td><td>49.2%</td><td>0.783</td><td>43.2%</td></tr><tr><td>100</td><td>0.848</td><td>62.3%</td><td>0.832</td><td>60.0%</td><td>0.800</td><td>66.7%</td><td>0.829</td><td>65.8%</td><td>0.793</td><td>66.9%</td></tr><tr><td>1000</td><td>0.841</td><td>81.8%</td><td>0.782</td><td>82.4%</td><td>0.742</td><td>81.7%</td><td>0.774</td><td>83.9%</td><td>0.735</td><td>82.7%</td></tr><tr><td>Overall</td><td>0.904</td><td>97.1%</td><td>0.868</td><td>97.1%</td><td>0.829</td><td>97.1%</td><td>0.855</td><td>97.1%</td><td>0.821</td><td>97.1%</td></tr></table>

Table 1: Comparison of methods using nDCG and Quality scores at diferent @k levels on the VaccinItaly Dataset. The maximum Quality score remains identical across methods when all the nodes are removed from the misinformation reshare network, except those that appear after the training phase; consequently, 2.9% of misinformation cannot be mitigated due to the emergence of new users in the test set. Bold values denote the best results, underlined values indicate the second best.

Overall, our analysis highlights a clear trade-of between model complexity and efectiveness: while ML ofers superior performance, heuristic-based methods like TASH-Index can approximate its impact at lower computational costs. The TAI-Score, however, demonstrates value when wide-scale intervention is possible or necessary.

The results from the COVID-19 Multilanguage dataset closely mirror those observed in the VaccinItaly dataset (see Table A.2 in the Appendix), reinforcing the robustness and generalizability of our findings across diferent linguistic and geographic contexts.

## 6.2. Machine Learning Explanation

The machine learning component of our framework is designed to rigorously evaluate the efectiveness of combining multiple behavioral archetype features to characterize online user behavior—an area that remains underexplored in prior literature. By leveraging interpretable and well-established modeling techniques, our approach not only facilitates accurate ranking of users by their misinformation spreading potential, but also provides explanatory insight into the relative influence of diferent behavioral traits. This enables a deeper understanding of the mechanisms behind misinformation dissemination and ofers actionable directions for intervention strategies.

![](images/63f7d679ea65af73c207121ae6eeb814014d3ced7ea8b4b1a913c64ca8a29785.jpg)  
Figure 4: SHAP summary plot: The plot displays the impact of each feature on the machine learning model’s predictions. Each dot represents a SHAP value for a particular user-feature instance, with color indicating the actual feature value (from low in blue to high in red). SHAP values quantify the direction and magnitude of a feature’s contribution to the model’s output. Values are shown on a logarithmic scale to enhance visibility and interpretability.

To better understand and interpret the predictions of the machine learning model, we adopt SHAP [60], a widely used framework that quantifies the contribution of each input feature to the model’s output. This enhances both interpretability and transparency, allowing us to uncover which behavioral archetypes most strongly drive the ranking outcomes.

While the TASH-Index is inherently interpretable, machine learning models are typically opaque. SHAP mitigates this limitation by decomposing each prediction into additive contributions from individual features. In our setting, the model is intentionally constrained to use only archetypal features. This constraint ensures that the model’s decisions are grounded in the predefined behavioral profiles. Crucially, this setup allows for a meaningful comparison between the performance and interpretability of the TASH-Index and the ML model, ofering complementary insights into the behavioral signals that characterize key misinformation spreaders.

Figure 4 shows the SHAP summary plot for the ML regression model. Each point represents a user, with the color indicating the feature value (red for high, blue for low), and the position along the x-axis representing the impact of that feature on the user’s predicted ranking. The most influential feature is the Repost Count, associated with the amplifier archetype, which quantifies the volume of low-credibility content a user reshared.

The color gradient in the SHAP summary plot adds interpretability by linking feature values to their impact on model predictions. In the case of the Repost Count feature, the plot reveals a robust positive correlation between a user’s frequency of amplifying low-credibility content and their predicted ranking. That is, the model consistently learns that frequent resharing behavior is a strong signal to detect pivotal actors in misinformation dissemination.

Interestingly, although the Repost Count feature is not directly linked to the super-spreader archetype, it emerges as the most important feature in the prediction model. This finding is somewhat unexpected, as one might assume that features associated with influence—such as those defining the super-spreader archetype—would exert the strongest efect. However, this finding underscores the critical role of amplifiers in the information ecosystem and shows that resharing alone can serve as a dominant behavioral indicator for identifying relevant actors in the spread of misinformation.

The next two most influential features identified by the model are the TAI-Score and the TASH-Index, both linked to the super-spreader archetype. In both cases, higher values correspond to higher predicted rankings, indicating that the model efectively identifies users who generate original content that triggers wide reshare cascades and play a central role in initiating the spread of misinformation. This reinforces the role of super-spreaders as central actors in the misinformation ecosystem. Other features, such as Coordination-Centrality, EaR-Index, and Coordination-EdgeWeight, exhibit lower but still meaningful contributions to the model outcome.

Overall, the SHAP summary analysis confirms that the model ranks users based on nuanced combinations of archetypal features. While it assigns substantial importance to behaviors typical of the super-spreader archetype— such as high TAI-Score and TASH-Index—it consistently prioritizes features linked to the amplifier archetype, with the Repost Count emerging as the dominant predictor. This is particularly interesting, as amplifier-like behavior alone may not fully characterize high-risk users, yet when combined with other signals, it plays a critical role in the model’s decision-making. By integrating multiple archetypal dimensions, the model surpasses simpler strategies based on a single archetype.

To better understand how the model leverages diferent features across the ranking spectrum, we analyze SHAP contributions separately for four contiguous intervals in the rank (1–10, 10–100, 100–1000, 1000–3000). This stratified view reveals how the decision process shifts from dominant archetypal profiles to hybrid archetypal profiles as we move down the ranking.

![](images/c969f4329bfbe0c2aa510a2184a779c3c8c14c8cb102a7c90fed0b90930ba15a.jpg)

![](images/175baea874b277351d18c7a9d4c33378f99bff0550192fd6a16e98d971757136.jpg)

![](images/411f7cf0e9398336cc8c00d77dd95d032eb653e9c0a1f27d41e4ea73ebe2fc71.jpg)

![](images/0d90abeed687ab343a59a4f8063f7adeeb91a77774eaa7dd2a7c09bdb3ecee32.jpg)  
Figure 5: SHAP decision plots showing how archetypal features contribute to model predictions for users in diferent ranking intervals. Each line represents a user, starting from the base value and accumulating SHAP values (bottom up) to reach the final prediction on the x-axis. Line color indicates the total SHAP contribution (log-scaled), revealing how the influence of diferent archetypes varies across the ranking.

In the top ranks (1–10), the model predominantly relies on the TASH-Index and TAI-Score, which are the key archetypal features linked to superspreader behavior. This indicates that the highest-ranked users are not merely active in the network, but instead exhibit behaviors that align with the dynamics of influential super-spreaders.

Between positions 10 and 100, these archetypal features remain important, but features such as Repost Count and Coordination-Centrality begin to play a stronger role. This suggests that the model is integrating both coordinated and amplifier behaviors, possibly capturing hybrid user types.

In the lower ranks (100–1000), the Repost Count emerges as the most influential feature. This reflects the typical behavior of users who primarily engage in amplifying content rather than originating it. While these users may not exhibit high individual impact, their consistent engagement still plays a meaningful role in shaping the overall dynamics of misinformation difusion—a pattern the model efectively captures. Notably, the increasing relevance of the Coordination-Centrality feature suggests that structured and coordinated activity—rather than isolated individual behavior—becomes a key driver of influence. This highlights the importance of coordination signals in identifying actors who contribute collectively to the spread of harmful content, even if their individual metrics are modest. This layered explanation highlights how the model shifts its decision-making across the ranking: features associated with the super-spreader archetype — most notably the TASH-Index — dominate the top-tier predictions, underscoring the model’s ability to capture key high-risk patterns. As we move down the ranks, signals linked to amplifiers (such as Repost Count) and to coordinated accounts (Coordination-Centrality) gain relative importance, while other indicators like the EaR-Index (amplifiers) and Coord-EdgeWeight (coordinated accounts) remain marginal throughout.

These findings may inform diverse moderation strategies based on distinct behavioral archetypes. Top-ranked users, whose behavior strongly aligns with the super-spreader archetype, might warrant targeted preventive interventions, such as increased moderation scrutiny, fact-checking overlays, or temporary content restrictions. In contrast, mid- and lower-ranked users, whose behavior reflects coordinated or repetitive resharing activity patterns, could be better addressed through broader systemic measures like limiting the amplification of their posts, adjusting recommendation exposure, or applying friction mechanisms (e.g., share limits or warning prompts).

## 6.3. Data Scarcity Robustness

To assess the robustness of the most promising methodologies identified in our evaluation (the TASH-Index and the ML approach) in more realistic conditions, we compare their performance in a data-scarce scenario. We aim to answer the following question: To what extent does delaying the start of data collection impact model performance? Data scarcity is a common challenge for many predictive algorithms. The more data a method requires, the longer the waiting time before action can be taken. To address this, we analyze the robustness of the TASH-Index and the machine learning model when faced with limited data.

![](images/c2bbf792222553bfe15baeae42fe032c2ef0901689e8d2b3e2a23d41aa5f1a4d.jpg)  
Figure 6: Performance of the top-performing models under data scarcity conditions. Each point corresponds to a training of the model with X days of data, obtaining Y as nDCG in evaluation over the same period.

Figure 6 reports the performance of four ranking models under varying data availability conditions. The x-axis shows the number of days of activity used for training, while the y-axis reports the corresponding nDCG@10 score on a fixed test set. Each point represents a complete training and evaluation cycle on a distinct subset of the training data.

The H-Index curve confirms previous results. In this scenario, its performance slightly improves as the observation window shortens, likely because shorter windows capture only the most recent (and thus most predictive) user activity. The TASH-Index, on the other hand, demonstrates remarkable stability and consistently outperforms the H-Index, indicating its robustness to data scarcity. The machine learning model was evaluated in two conditions. The re-trained ML model, i.e., trained from scratch on each data subset, exhibits instability—its performance fluctuates and degrades as less data becomes available. Conversely, the pre-trained ML model, which does not require any re-training, maintains stable and high performance across most of the data reduction range. Notably, the pre-trained model’s performance only starts degrading when trained on the smallest window (30 days). This drop likely occurs because the underlying user archetypal feature vectors become too sparse, impairing the model’s ability to make reliable predictions.

## 7. Conclusion

## 7.1. Discussion

This study presents a formal taxonomy of three behavioral archetypes of misinformation actors—super-spreaders, amplifiers, and coordinated accounts—and develops corresponding ranking models to assess their influence in the dissemination process. By focusing on observable behavioral patterns rather than user intent, our approach aligns with a core principle of governance: interventions should address systemic harm, regardless of individual motivation. Still, intent-aware approaches could complement our models. For example, integrating influence-based methods such as the one proposed by Zhou et al. [61] may enhance ground truth annotations and provide a richer basis for evaluation. Such hybrid models could support more nuanced, tiered moderation strategies that respond diferently based on user intent.

A central contribution of this work lies in demonstrating that combining distinct archetypal features within an interpretable model can yield both competitive predictive performance and actionable insights into misinformation dynamics.

Methodologically, our framework targets behavioral patterns rather than account identity, treating human and automated actors alike. This reflects the increasing overlap between human and bot-like behaviors driven by generative AI, which challenges traditional classification schemes. We argue that behavioral impact ofers a reliable signal for moderation, though future work should assess how archetypal behaviors may change with the increasing use of generative AI tools as large language and vision models.

Another core contribution of our work is the TASH-Index, a novel adaptation of the H-Index designed to capture both temporal and structural aspects of misinformation difusion in a computationally eficient and interpretable manner. Our experiments show that the TASH-Index performs competitively even in low-data settings and enhances the performance of machine learning models when used as a feature. This highlights the value of simple, behavior-based metrics that remain robust and explainable.

From a deployment perspective, our approach is best suited for integration within human-in-the-loop moderation systems, where model-generated rankings act as decision-support tools rather than automated enforcement mechanisms. This allows for scalable interventions, such as prioritizing content for review or limiting visibility, while preserving oversight and accountability. To ensure transparency and reduce potential for misuse, such systems should include safeguards such as auditability and mechanisms for appeal.

## 7.2. Limitations

We acknowledge several limitations of our study. First, the reshare network analyzed captures only direct interactions between original content creators and resharers, without accounting for potential indirect exposure pathways. This limitation stems from platform-level constraints, as most social media APIs do not provide access to complete reshare cascades [62]. Second, our labeling of low-credibility content relies on simple yet established techniques based on domain-level assessments. While efective for our purposes, this approach could be extended to other modalities, such as images and videos, contingent on the availability of appropriate credibility annotations. Third, our ranking models are limited to users observed during training, which restricts their applicability to newly emerging accounts. While the TASH-Index can be updated incrementally with each reshare, enabling realtime applicability, developing analogous, incrementally computable metrics for other archetypal behaviors remains an open research challenge. Finally, our evaluation framework relies on static metrics, which do not account for the dynamic, system-level efects of interventions. Real-world actions, such as content takedowns or account suspensions, can induce behavioral adaptations that reshape the network structure and difusion processes. Although recent work has begun to explore these dynamics [63, 64, 65], there is still a lack of scalable simulation environments for robust evaluation of intervention strategies.

## 7.3. Future Work

Several promising research directions arise from this work. First, we aim to further refine the TASH-Index and extend its integration into expressive learning architectures such as graph neural networks, which ofer the potential to capture more nuanced temporal and relational structures underlying misinformation difusion. Second, we plan to enhance the parameter tuning process of the TASH-Index. Future work will explore adaptive optimization strategies, including gradient-based tuning, Bayesian optimization, or evolutionary algorithms, to enhance generalizability across platforms. Third, access to richer, multi-platform datasets would enable more comprehensive modeling of user behavior and facilitate the generalization of archetype-based approaches. Such datasets would also support the development of inductive models capable of classifying newly emerging users—an essential step toward real-time moderation pipelines. Finally, we underscore the importance of simulation-based evaluation frameworks that can model the reactive dynamics of online platforms. These tools are critical for assessing the long-term impact, fairness, and efectiveness of interventions, and for informing the responsible design and deployment of predictive moderation systems.

## Acknowledgements

This work was supported by the Swiss National Science Foundation through the CARISMA project (Sinergia grant CRSII5 209250). The authors wish to thank Gianluca Nogara for providing data and supporting the related analysis, Matt DeVerna for his insightful suggestions, Fil Menczer for ofering invaluable insights to this research and all the CARISMA’s participants for meaningful discussions.

## Appendix A. Rankings Methods

This appendix provides additional details on the ranking methods evaluated in our study. We begin with a subsection on the optimization of time-aware approaches, describing how their parameters were tuned. Subsequent subsections present detail ranking methods per each archetype. For each method, we include a visualization of its dismantling track performance, highlighting its efectiveness in reducing misinformation spread over time.

## Appendix A.1. Time-Aware Methods Optimization

To determine the optimal values of α and δ, we performed a grid search over a predefined range. This resulted in $\alpha \ : = \ : 0 . 5$ and δ = 14 days for the TASH-Index model, and $\alpha = 0 . 6$ and δ = 18 days for the TAI-index model. A visualization of the search space for TASH-Index is represented in Figure A.7.

## Appendix A.2. Ranking Amplifiers

Figure A.8 evaluates two statistical methods designed to detect amplifiers, as introduced in Section 4.2.1.

![](images/233232b40318951311bbdf6d87658360f2d240e4b1657d3864e74c460389f5a7.jpg)  
Figure A.7: Optimization grid of α and δ for the TASH-Index using nDCG as the evaluation metric.

Surprisingly, the Early Reposter method—based on the idea that early sharers are influential—performs worse than the simpler Repost Count method. This challenges the initial hypothesis that early resharing is a strong proxy for influence in misinformation difusion.

Figure A.8 shows that reshare frequency is a more efective ranking strategy for identifying amplifiers than early reposting.

## Appendix A.3. Ranking Coordinated Accounts

This subsection evaluates coordinated users using two methods based on the co-reshare similarity network (Section 4.2.2). The network is undirected, where edge weights represent cosine similarity between users’ reshare patterns.

Misinformation re-post network: dismantling  
![](images/a581863aff3f38cadf8514e6568bd37947cc50b4950587701eac338d107cad22.jpg)  
Figure A.8: Performance comparison of amplifiers-ranking strategies.

Figure A.9 shows the results for:

• Coordination Centrality: Ranks users by their centrality in the coreshare similarity network.

• Coordination Max Weight: Ranks each user by their highest cosine similarity edge.

The former method outperforms the latter, suggesting that being embedded in a densely connected coordination cluster is more indicative of influence than having a single strong link. Centrality-based rankings better capture structural importance within coordination networks.

## Appendix A.4. Ranking Super-Spreaders

The Figure A.10 shows the dismantling trajectories of time-aware and time-agnostic methods within the super-spreaders archetype presented in 4.2.3. Both the TASH-Index and TAI-Score show clear improvements when temporal dynamics are included.

![](images/7cdc9a0a089bf418e2ffa86c0731010ae20a13bd6cf74a4a351c59dde6d2ab5d.jpg)  
Figure A.9: Performance of coordinated user detection strategies based on similarity networks.  
Appendix A.5. Ranking Performance in the Multilingual COVID-19 Dataset

The results on the COVID-19 Multilanguage dataset in Table A.2 closely mirrors those observed in the VaccinItaly dataset (Table 1, ensuring the robustness of the findings across diferent contexts. Consistently, the ML model achieves the highest nDCG scores at the top of the ranking, indicating its ability to identify highly influential users. In particular, ML and TASH-index yield the highest nDCG scores, outperforming all other methods, confirming that both models capture strong predictive signals.

## Appendix B. Sensitivity Analyses

This appendix investigates two complementary forms of model sensitivity. The first examines how varying the credibility threshold used to select training data afects model performance. The second explores whether the models tend to prioritize users who consistently share misinformation over those with a mixed content history, ofering insight into post-training model behavior.

## Appendix B.1. Sensitivity to Credibility Threshold

We assess how model performance changes when the credibility threshold applied during training is varied. This threshold determines which reshare actions are included in the training data: only actions involving content with credibility scores less than or equal to the threshold are used. For example, a threshold of 20 means that only reshares of content with credibility scores ≤ 20 are included during training.

Misinformation re-post network: dismantling  
![](images/5bd5981a667294b0bab21a2e7c115e418e67524205d49a4160f98e796d2fc782.jpg)  
Figure A.10: Comparison of misinformation removal efectiveness across super-spreader identification methods.

Crucially, the test set remains the same and always includes actions labeled using a credibility threshold of 39.0 — which corresponds to content considered “unreliable because it severely violates basic journalistic standards”, according to NewsGuard. This fixed reference allows us to isolate the efect of varying the threshold during training.

As shown in Figure B.11, we evaluate three representative models: the Hindex approach, the best-performing archetype-based method (TASH-index), and the machine learning model. Overall, results indicate that varying the threshold has limited efects on models performance, with only small fluctuations observed across thresholds.

An exception occurs at the extreme case of a threshold of 0.0. In this setting, only reshare actions involving content with a credibility score of exactly 0 are retained for training. This aggressive filtering results in significantly worse performance, likely due to the sharp reduction in the training data size.

Interestingly, when the credibility threshold is set to its maximum value (i.e., no filtering is applied and all available data are used for training), the performance of the ML-based model slightly drops, approaching that of the H-index. In contrast, the TASH-index maintains a favorable performance level.

<table><tr><td rowspan="2">@k</td><td colspan="2">ML</td><td colspan="2">TASH-Index</td><td colspan="2">H-Index</td><td colspan="2">TAI-Score</td><td colspan="2">Influence Score</td></tr><tr><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td><td>nDCG</td><td>Quality</td></tr><tr><td>3</td><td>0.986</td><td>41.7%</td><td>0.986</td><td>41.7%</td><td>0.880</td><td>34.1%</td><td>0.914</td><td>36.2%</td><td>0.932</td><td>37.6%</td></tr><tr><td>5</td><td>0.939</td><td>42.7%</td><td>0.932</td><td>42.0%</td><td>0.905</td><td>41.5%</td><td>0.932</td><td>43.0%</td><td>0.886</td><td>38.4%</td></tr><tr><td>10</td><td>0.918</td><td>46.8%</td><td>0.904</td><td>45.2%</td><td>0.919</td><td>51.3%</td><td>0.931</td><td>49.5%</td><td>0.918</td><td>49.0%</td></tr><tr><td>20</td><td>0.932</td><td>56.9%</td><td>0.926</td><td>56.8%</td><td>0.899</td><td>54.9%</td><td>0.925</td><td>56.7%</td><td>0.904</td><td>54.0%</td></tr><tr><td>100</td><td>0.896</td><td>68.6%</td><td>0.880</td><td>66.2%</td><td>0.872</td><td>68.4%</td><td>0.890</td><td>69.5%</td><td>0.884</td><td>70.7%</td></tr><tr><td>1000</td><td>0.884</td><td>81.5%</td><td>0.829</td><td>82.0%</td><td>0.823</td><td>83.7%</td><td>0.840</td><td>84.8%</td><td>0.834</td><td>85.1%</td></tr><tr><td>Overall</td><td>0.926</td><td>95.2%</td><td>0.886</td><td>95.2%</td><td>0.876</td><td>95.2%</td><td>0.890</td><td>95.2%</td><td>0.884</td><td>95.2%</td></tr></table>

Table A.2: Comparison ofmethods using nDCG and Quality scores at diferent @k levels on COVID-19 Multilanguage dataset. The maximum Quality score remains identical across methods because all the nodes are removed from the misinformation reshare network except those that appear after the training phase; consequently, 4.8% of misinformation cannot be mitigated due to the emergence of new users in the test phase. Bold values denote the best results, underlined values indicate the second best.

## Appendix B.2. Sensitivity to Credibility in Ranking Composition

To better understand how diferent ranking systems prioritize users in terms of the credibility of their content, we perform a sensitivity analysis focused on the top-100 users produced by each method. Figure B.12 presents a comparative visualization, where each column corresponds to one ranking method, and each point (bubble) represents a user appearing in the top-100 for that method.

Each user is shown twice: once for their original posts and once for their reshares, distinguished by color: orange for posts and blue for reshares across all rankings, except for the Optimal ranking, which uses green (posts) and pink (reshares) to visually mark its special status.

The vertical axis reports the user’s average credibility score for that specific action type, while the bubble size indicates the user’s position within the ranking (larger bubbles denote higher ranks).

Importantly, only the Optimal ranking (fourth from the left) is computed using future data from the test set: users are ranked based on the actual volume of misinformation they disseminate. This makes it a kind of oracle, ofering a behavioral benchmark for comparison. For all methods, including

Credibility threshold sensitivity  
![](images/813d072a40fe0b8cb7220d6c72af49531d9826f20d24cc4ea10985cfb59c89bb.jpg)  
Figure B.11: Impact of varying the credibility threshold during training. The x-axis indicates the threshold applied, and the y-axis shows model performance (nDCG). For each threshold x, only training data involving content with credibility scores ≤ x was used.

Optimal, credibility scores are always computed from the training data to ensure consistency.

This analysis is not meant to compare performance, but rather to reveal the credibility profiles of the users that each method elevates. Notably, some methods, such as TAI-Score or ML, show wide variability in the credibility of highly ranked users, while others like Coord-EdgeWeight appear to concentrate more around specific ranges.

## Appendix C. Machine Learning

This appendix provides additional details on the machine learning method ology used in our study. We describe both the training procedure and the model selection process that guided our final choice. The first subsection covers how the data was split and how model tuning was performed, including the evaluation strategy. The second subsection presents a comparison of diferent models considered, highlighting their relative performance and justifying the selection of the final model.

## Appendix C.1. Model Tuning

All data was chronologically ordered and processed using a sliding window approach, ensuring strict temporal consistency throughout the pipeline (i.e., no leakage of future information into the training data). Importantly, we adopted a sample-wise data split, where the split is defined over the number of resharing events (i.e., rows), not over a fixed duration of time (e.g., days or weeks). This choice allows for more granular control over the training set size while preserving the natural temporal order of events.

![](images/3223186f9988b443a4717c29e64d8acba8f27b025b690a98859ce470bc21ece6.jpg)  
Figure B.12: Credibility profiles of top-100 ranked users across diferent ranking methods. Each column represents a ranking method, with users shown as bubbles positioned according to their average credibility scores for original posts (orange) and reshares (blue). For the Optimal ranking, we use distinct colors (green for posts, pink for reshares) to empha size its role as the ground-truth baseline. Bubble size reflects ranking position, with larger bubbles indicating higher ranks. While all credibility scores are computed from training data, only the Optimal ranking is derived using test-set information based on actual future misinformation spread. This visualization reveals how each method prioritizes users with diferent credibility profiles, providing insight into the types of actors emphasized by diferent ranking strategies.

Each sliding window was defined over a contiguous, time-ordered sequence of resharing events. Within each window, models were trained on a prefix of the data and evaluated on the next 20% of events, simulating forward prediction in real-world deployment settings. Figure C.13 illustrates this procedure.

This process was repeated across all windows and for every hyperparameter configuration. We selected the configuration that achieved the best average performance across these temporally consistent folds. The procedure ensures robustness to diferent data regions and guards against overfitting on early or late time periods.

![](images/cd5854a578058429e337e6ab48500c64df32218728ccdc9b0ac9c4c280f41604.jpg)  
Figure C.13: Sliding window procedure for model tuning and evaluation.

## Appendix C.2. Model Selection

While user behavior often overlaps across archetypes, machine learning can help disentangle this complexity. We tested multiple models using a hyperparameter grid search, with performance on the test set reported in Table C.3. Given these results, we adopt the Random Forest model as our ML baseline in Section 4.3, particularly to assess how TASH-Index contributes when used alongside other user features.

<table><tr><td>Model</td><td>MAE↓</td><td>MSE↓</td><td>nDCG@3↑</td><td>nDCG@5↑</td><td>nDCG@10↑</td><td>nDCG ↑</td></tr><tr><td>Linear Model</td><td>1.073</td><td>343.34</td><td>0.955</td><td>0.920</td><td>0.851</td><td>0.890</td></tr><tr><td>Random Forest</td><td>1.108</td><td>397.55</td><td>0.970</td><td>0.968</td><td>0.939</td><td>0.910</td></tr><tr><td>XGBoost</td><td>1.079</td><td>641.62</td><td>0.750</td><td>0.860</td><td>0.850</td><td>0.864</td></tr><tr><td>Neural Network</td><td>1.287</td><td>361.50</td><td>0.970</td><td>0.928</td><td>0.910</td><td>0.900</td></tr></table>

Table C.3: Performance comparison of tested models on the test set. Random Forest emerges as the most stable and reliable model overall, especially in managing high target skewness and inter-feature correlation.

Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author(s) used DeepL (DeepL SE), ChatGPT (OpenAI) and Claude (Anthropic) in order to enhance the language and readability. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

## References

[1] S. Bhattacharya, A. Singh, Unravelling the infodemic: a systematic review of misinformation dynamics during the covid-19 pandemic, Frontiers in Communication 10 (2025) 1560936.

[2] F. Pierri, M. R. DeVerna, K.-C. Yang, D. Axelrod, J. Bryden, F. Menczer, One year of covid-19 vaccine misinformation on twitter: Longitudinal study, Journal of Medical Internet Research 25 (2023) e42227. doi:10.2196/42227. URL http://dx.doi.org/10.2196/42227

[3] G. Verma, A. Bhardwaj, T. Aledavood, M. De Choudhury, S. Kumar, Examining the impact of sharing covid-19 misinformation online on mental health, Scientific Reports 12 (1) (2022) 8045.

[4] G. Nogara, P. S. Vishnuprasad, F. Cardoso, O. Ayoub, S. Giordano, L. Luceri, The Disinformation Dozen: An Exploratory Analysis of Covid-19 Disinformation Proliferation on Twitter, Proc. 14th ACM Web Science Conference 2022 (2022). URL https://doi.org/10.1145/3501247.3531573

[5] M. Cinelli, W. Quattrociocchi, A. Galeazzi, C. M. Valensise, E. Brugnoli, A. L. Schmidt, P. Zola, F. Zollo, A. Scala, The covid-19 social media infodemic, Scientific Reports 10 (1) (Oct. 2020). doi: 10.1038/s41598-020-73510-5. URL http://dx.doi.org/10.1038/s41598-020-73510-5

[6] K.-C. Yang, F. Pierri, P.-M. Hui, D. Axelrod, C. Torres-Lugo, J. Bryden, F. Menczer, The COVID-19 Infodemic: Twitter versus Facebook, Big Data & Society 8 (1) (2021).

[7] V. L. Gatta, L. Luceri, F. Fabbri, E. Ferrara, The interconnected nature of online harm and moderation: Investigating the cross-platform spread of harmful content between youtube and twitter, in: Proceedings of the 34th ACM conference on hypertext and social media, 2023, pp. 1–10.

[8] M. A. Gisondi, R. Barber, J. S. Faust, A. Raja, M. C. Strehlow, L. M. Westafer, M. Gottlieb, A deadly infodemic: social media and the power of covid-19 misinformation (2022).

[9] C. Erisen, E. Erisen, Populist attitudes and misinformation challenging trust: The case of turkey, International Journal of Public Opinion Research 37 (1) (2025) edae056. arXiv:https://academic.oup.com/ ijpor/article-pdf/37/1/edae056/62374318/edae056.pdf, doi:10. 1093/ijpor/edae056. URL https://doi.org/10.1093/ijpor/edae056

[10] L. Luceri, S. Cresci, S. Giordano, Social media against society: Information manipulation in the 2020 election, in: J. Baumgartner, T. Towner (Eds.), The Internet and the 2020 election, 2021, pp. 3–23.

[11] L. Edelson, B. Kovba, H. Yershova, A. Botelho, D. McCoy, T. Lauinger, Measurement and metrics for content moderation: The multidimensional dynamics of engagement and content removal on facebook, Journal of Online Trust and Safety 2 (5) (2025).

[12] G. Pennycook, Z. Epstein, M. Mosleh, A. A. Arechar, D. Eckles, D. G. Rand, Shifting attention to accuracy can reduce misinformation online, Nature 592 (7855) (2021) 590–595.

[13] B. T. Truong, S. Kim, G. Nogara, E. Verdolotti, E. S. Sahneh, F. Saurwein, N. Just, L. Luceri, S. Giordano, F. Menczer, Delayed takedown of illegal content on social media makes moderation inefective. arXiv:2502.08841[cs], doi:10.48550/arXiv.2502.08841. URL http://arxiv.org/abs/2502.08841

[14] R. Meerson, K. Koban, J. Matthes, Platform-led content moderation through the bystander lens: a systematic scoping review, Information, Communication & Society (2025) 1–18.

[15] S. Vosoughi, D. Roy, S. Aral, The spread of true and false news online, science 359 (6380) (2018) 1146–1151.

[16] K. Solovev, N. Pröllochs, Moral emotions shape the virality of covid-19 misinformation on social media (2022). arXiv:2202.03590. URL https://arxiv.org/abs/2202.03590

[17] E. L. Wang, L. Luceri, F. Pierri, E. Ferrara, Identifying and characterizing behavioral classes of radicalization within the qanon conspiracy on twitter, Proceedings of the International AAAI Conference on Web and Social Media (ICWSM) abs/2209.09339 (2022). URL https://doi.org/10.48550/arXiv.2209.09339

[18] M. R. DeVerna, R. Aiyappa, D. Pacheco, J. Bryden, F. Menczer, Identifying and characterizing superspreaders of low-credibility content on twitter 19 (5) e0302201. doi:10.1371/journal.pone.0302201. URL https://dx.plos.org/10.1371/journal.pone.0302201

[19] S. Baribi-Bartov, B. Swire-Thompson, N. Grinberg, Supersharers of fake news on twitter, Science (New York, N.Y.) 384 (2024) 979–982. doi: 10.1126/science.adl4435.

[20] D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, F. Menczer, Uncovering coordinated networks on social media: Methods and case studies 15 455–466. doi:10.1609/icwsm.v15i1.18075. URL https://ojs.aaai.org/index.php/ICWSM/article/view/ 18075

[21] L. Luceri, V. Pantè, K. Burghardt, E. Ferrara, Unmasking the web of de ceit: Uncovering coordinated activity to expose information operations on twitter, in: Proceedings of the 2024 ACM Web Conference, 2024.

[22] S. Tardelli, L. Nizzoli, M. Tesconi, M. Conti, P. Nakov, G. Da San Martino, S. Cresci, Temporal dynamics of coordinated online behavior: Stability, archetypes, and influence, Proceedings of the National Academy of Sciences 121 (20) (2024) e2307038121.

[23] F. Cinus, M. Minici, L. Luceri, E. Ferrara, Exposing cross-platform coordinated inauthentic activity in the run-up to the 2024 us election, in: Proceedings of the ACM on Web Conference 2025, 2025, pp. 541–559.

[24] K. Sharma, Y. Zhang, E. Ferrara, Y. Liu, Identifying coordinated accounts on social media through hidden influence and group behaviours

(2021). arXiv:2008.11308. URL https://arxiv.org/abs/2008.11308

[25] K. W. Ng, A. Iamnitchi, Coordinated information campaigns on social media: A multifaceted framework for detection and analysis (2023). arXiv:2309.12729. URL https://arxiv.org/abs/2309.12729

[26] Z. Chen, J. Ye, B. Tsai, E. Ferrara, L. Luceri, Synthetic politics: Prevalence, spreaders, and emotional reception of ai-generated political images on x, in: Proceedings of the 36th ACM conference on hypertext and social media, 2025.

[27] V. Pantè, D. Axelrod, A. Flammini, F. Menczer, E. Ferrara, L. Luceri, Beyond interaction patterns: Assessing claims of coordinated inter-state information operations on twitter/x (2025). arXiv:2502.17344. URL https://arxiv.org/abs/2502.17344

[28] P. Shaeri, A. Katanforoush, A semi-supervised fake news detection using sentiment encoding and LSTM with self-attention. arXiv:2407. 19332[cs], doi:10.48550/arXiv.2407.19332. URL http://arxiv.org/abs/2407.19332

[29] Q. Guo, Z. Kang, L. Tian, Z. Chen, TieFake: Title-text similarity and emotion-aware fake news detection. arXiv:2304.09421[cs], doi:10. 48550/arXiv.2304.09421. URL http://arxiv.org/abs/2304.09421

[30] A. A. J. Karim, K. H. M. Asad, A. Azam, Strengthening fake news detection: Leveraging SVM and sophisticated text vectorization techniques. defying BERT? arXiv:2411.12703[cs], doi:10.48550/arXiv.2411. 12703. URL http://arxiv.org/abs/2411.12703

[31] S. Abdali, S. shaham, B. Krishnamachari, Multi-modal misinformation detection: Approaches, challenges and opportunities. arXiv: 2203.13883[cs], doi:10.48550/arXiv.2203.13883. URL http://arxiv.org/abs/2203.13883

[32] E. Stockinger, R. Gallotti, C. I. Hausladen, The connection between the spread of misinformation, time of day, and individual user activity patterns. arXiv:2307.11575[cs], doi:10.48550/arXiv.2307.11575. URL http://arxiv.org/abs/2307.11575

[33] F. Zhou, L. Lü, J. Liu, M. S. Mariani, Beyond network centrality: Individual-level behavioral traits for predicting information superspreaders in social media. arXiv:2112.03546[cs], doi:10.1093/nsr/ nwae073. URL http://arxiv.org/abs/2112.03546

[34] M. Avram, N. Micallef, S. Patil, F. Menczer, Exposure to social engagement metrics increases vulnerability to misinformationarXiv:2005. 04682[cs], doi:10.37016/mr-2020-033. URL http://arxiv.org/abs/2005.04682

[35] M. Karami, T. H. Nazer, H. Liu, Profiling fake news spreaders on social media through psychological and motivational factors, in: Proceedings of the 32st ACM Conference on Hypertext and Social Media, pp. 225– 230. arXiv:2108.10942[cs], doi:10.1145/3465336.3475097. URL http://arxiv.org/abs/2108.10942

[36] J. Ye, L. Luceri, J. Jiang, E. Ferrara, Susceptibility to unreliable information sources: Swift adoption with minimal exposure, in: Proceedings of the ACM Web Conference 2024, 2024, pp. 4674–4685.

[37] L. Luceri, J. Ye, J. Jiang, E. Ferrara, The susceptibility paradox in online social influence, in: Proceedings of the International AAAI Conference on Web and Social Media, Vol. 19, 2025, pp. 1122–1138.

[38] B. Rath, A. Salecha, J. Srivastava, Detecting fake news spreaders in social networks using inductive representation learning. arXiv:2011. 10817[cs], doi:10.48550/arXiv.2011.10817. URL http://arxiv.org/abs/2011.10817

[39] B. Rath, X. Morales, J. Srivastava, SCARLET: Explainable attention based graph neural network for fake news spreader prediction. arXiv: 2102.04627[cs], doi:10.48550/arXiv.2102.04627. URL http://arxiv.org/abs/2102.04627

[40] F. Sakketou, J. Plepi, H.-J. Geiss, L. Flek, Temporal graph analysis of misinformation spreaders in social media.

[41] A. Ullah, R. A. Abbasi, A. S. Khattak, A. Said, Identifying misinformation spreaders: A graph-based semi-supervised learning approach. arXiv:2303.03704[cs], doi:10.48550/arXiv.2303.03704. URL http://arxiv.org/abs/2303.03704

[42] M. Minici, L. Luceri, F. Fabbri, E. Ferrara, Iohunter: Graph foundation model to uncover online information operations, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 28258– 28266.

[43] E. Ferrara, O. Varol, C. Davis, F. Menczer, A. Flammini, The rise of social bots, Communications of the ACM 59 (7) (2016) 96–104.

[44] L. Luceri, A. Deb, A. Badawy, E. Ferrara, Red bots do it better: Comparative analysis of social bot partisan behavior, in: Companion proceedings of the 2019 world wide web conference, 2019, pp. 1007–1012.

[45] S. L. Lim, P. J. Bentley, Opinion amplification causes extreme polarization in social networks, Scientific Reports 12 (2022). URL https://api.semanticscholar.org/CorpusID:253163946

[46] E. Nowak-Teter, B. Łódzki and, What makes news shared on facebook? social media logic and content-related factors of shareability, Digital Journalism 12 (4) (2024) 451–475. doi:10.1080/21670811.2023. 2218902.

[47] A. Peck, A problem of amplification: Folklore and fake news in the age of social media, Journal of American Folklore 133 (2020) 329 – 351. URL https://api.semanticscholar.org/CorpusID:243130538

[48] K. Hristakieva, S. Cresci, G. D. S. Martino, M. Conti, P. Nakov, The spread of propaganda by coordinated communities on social media, Proceedings of the 14th ACM Web Science Conference 2022 (2021). URL https://api.semanticscholar.org/CorpusID:237940264

[49] K. Sharma, Y. Zhang, E. Ferrara, Y. Liu, Identifying coordinated accounts on social media through hidden influence and group behaviours, Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (2020). URL https://api.semanticscholar.org/CorpusID:234771738

[50] P. S. Vishnuprasad, G. Nogara, F. Cardoso, S. Cresci, S. Giordano, L. Luceri, Tracking fringe and coordinated activity on twitter leading up to the us capitol attack, in: Proceedings of the international AAAI conference on web and social media, Vol. 18, 2024, pp. 1557–1570.

[51] L. Luceri, T. V. Salkar, A. Balasubramanian, G. Pinto, C. Sun, E. Ferrara, Coordinated inauthentic behavior on tiktok: Challenges and opportunities for detection in a video-first ecosystem, arXiv preprint arXiv:2505.10867 (2025).

[52] M. Minici, L. Luceri, F. Cinus, E. Ferrara, Uncovering coordinated crossplatform information operations threatening the integrity of the 2024 us presidential election online discussion, First Monday (2024).

[53] P. Metaxas, E. Mustafaraj, K. Wong, L. Zeng, M. O’Keefe, S. Finn, What do retweets indicate? results from user survey and meta-review of research, in: Proceedings of the international AAAI conference on web and social media, Vol. 9, 2015, pp. 658–661.

[54] K. Järvelin, J. Kekäläinen, Cumulated gain-based evaluation of ir techniques, ACM Trans. Inf. Syst. 20 (4) (2002) 422–446. doi:10.1145/ 582415.582418. URL https://doi.org/10.1145/582415.582418

[55] T.-Y. Liu, Learning to rank for information retrieval, Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval (2009). URL https://api.semanticscholar.org/CorpusID:28826624

[56] H. Li, Learning to Rank for Information Retrieval and Natural Language Processing, Second Edition, Vol. 4, 2011. doi:10.2200/ S00348ED1V01Y201104HLT012.

[57] V. Melnikov, E. Hüllermeier, D. Kaimann, B. Frick, P. Gupta, Pairwise versus pointwise ranking: A case study, Schedae Informaticae 25 (2016). doi:10.4467/20838476si.16.006.6187.

[58] F. Pierri, A. Tocchetti, L. Corti, M. D. Giovanni, S. Pavanetto, M. Brambilla, S. Ceri, Vaccinitaly: monitoring italian conversations around vaccines on twitter and facebook (2021). arXiv:2101.03757. URL https://arxiv.org/abs/2101.03757

[59] M. Di Giovanni, F. Pierri, C. Torres-Lugo, M. Brambilla, Vaccineu: Covid-19 vaccine conversations on twitter in french, german and italian, Proceedings of the International AAAI Conference on Web and Social Media (ICWSM) (2022).

[60] S. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions (2017). arXiv:1705.07874. URL https://arxiv.org/abs/1705.07874

[61] X. Zhou, K. Shu, V. Phoha, H. Liu, R. Zafarani, "this is fake! shared it by mistake": Assessing the intent of fake news spreaders (02 2022). doi:10.48550/arXiv.2202.04752.

[62] M. R. DeVerna, F. Pierri, R. Aiyappa, D. Pacheco, J. Bryden, F. Menczer, Information difusion assumptions can distort our understanding of social network dynamics. arXiv:2410.21554[cs], doi: 10.48550/arXiv.2410.21554. URL http://arxiv.org/abs/2410.21554

[63] L. H. Butler, P. X. Lamont, D. L. Y. Wan, T. Prike, M. Nasim, B. Walker, N. Fay, U. K. H. Ecker, The (mis)information game: A social media simulator, Behavior Research Methods 56 (2023) 2376 – 2397. URL https://api.semanticscholar.org/CorpusID:259832118

[64] B. T. Truong, X. Lou, A. Flammini, F. Menczer, Quantifying the vulnerabilities of the online public square to adversarial manipulation tactics (2024). arXiv:1907.06130. URL https://arxiv.org/abs/1907.06130

[65] A. Sittar, S. Münker, F. Sartori, A. Reitenbach, A. Rettinger, M. Mäs, A. Guček, M. Grobelnik, Agent-based simulations of online political discussions: A case study on elections in germany (2025). arXiv:2503. 24199. URL https://arxiv.org/abs/2503.24199