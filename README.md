# Papers for Event Extraction
Papers from top conferences and journals for event extraction in recent years.

In order to show more information for each paper, we take a sentence from the abstract which can express the main purpose in the paper.

# Table of Contents


<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [ACL](#ACL)
- [EMNLP](#EMNLP)
- [NAACL](#NAACL)
- [COLING](#COLING)
- [AAAI](#AAAI)
- [Other Conferences](#Other Conferences)
- [Journals](#Journals)
</p></blockquote>

## ACL

### 2019

1. **Exploring Pre-trained Language Models for Event Extraction and Generation.**
  _Sen Yang, Dawei Feng, Linbo Qiao, Zhigang Kan, Dongsheng Li._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-1522.pdf)
  
      > We first propose an event extraction model to overcome the roles overlap problem by separating the argument prediction in terms of roles. Moreover, to address the problem of insufficient training data, we propose a method to automatically generate labeled data by editing prototypes and screen out generated samples by ranking the quality.
  
  
2. **Distilling Discrimination and Generalization Knowledge for Event Detection via ∆-Representation Learning.**
  _Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-1429.pdf)
  
    > This paper proposes a ∆-learning approach to distill discrimination and generalization knowledge by effectively decoupling, incrementally learning and adaptively fusing event representation.


3. **Cost-sensitive Regularization for Label Confusion-aware Event Detection.**
  _Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-1521.pdf)
  
    > This paper proposes cost-sensitive regularization, which can force the training procedure to concentrate more on optimizing confusing type pairs. Specifically, we introduce a costweighted term into the training loss, which penalizes more on mislabeling between confusing label pairs. Furthermore, we also propose two estimators which can effectively measure such label confusion based on instance-level or population-level statistics.


4. **Rapid Customization for Event Extraction.**
  _Yee Seng Chan, Joshua Fasching, Haoling Qiu, and Bonan Min._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-3006.pdf)
  
    > We present a system for rapidly customizing event extraction capability to find new event types (what happened) and their arguments (who, when, and where).
    
5. **Literary Event Detection.**
  _Matthew Sims, Jong Ho Park, David Bamman._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-1353.pdf)
  
    > In this work we present a new dataset of literary events—events that are depicted as taking place within the imagined space of a novel.
    
6. **Open Domain Event Extraction Using Neural Latent Variable Models.**
  _Xiao Liu, Heyan Huang, Yue Zhang._
  ACL 2019.
  [paper](https://www.aclweb.org/anthology/P19-1276.pdf)
  
    > We consider open domain event extraction, the task of extracting unconstraint types of events from news clusters. A novel latent variable neural model is constructed, which is scalable to very large corpus.   
    

### 2018

1. **Document Embedding Enhanced Event Detection with Hierarchical and Supervised Attention.**
  _Yue Zhao, Xiaolong Jin, Yuanzhuo Wang, Xueqi Cheng._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/P18-2066.pdf)
  
    > We proposed a hierarchical and supervised attention based and document embedding enhanced Bi-RNN method, called DEEB-RNN, for event detection. We explored different strategies to construct gold word and sentence-level attentions to focus on event information. 
    

2. **Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection.**
  _Yu Hong Wenxuan Zhou Jingli Zhang Qiaoming Zhu Guodong Zhou._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/P18-1048.pdf)
  
    > We propose a self-regulated learning approach by utilizing a generative adversarial network to generate spurious features. On the basis, we employ a recurrent network to eliminate the fakes.
    

3. **Zero-Shot Transfer Learning for Event Extraction.**
  _Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare R. Voss._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/P18-1201.pdf)
  
    > We design a transferable architecture of structural and compositional neural networks to jointly represent and map event mentions and types into a shared semantic space.
 
 
4. **Nugget Proposal Networks for Chinese Event Detection.**
  _Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/P18-1145.pdf)
  
    > We propose Nugget Proposal Networks (NPNs), which can solve the word-trigger mismatch problem by directly proposing entire trigger nuggets centered at each character regardless of word boundaries.


5. **DCFEE: A Document-level Chinese Financial Event Extraction System based on Automatically Labeled Training Data.**
  _Hang Yang, Yubo Chen, Kang Liu, Yang Xiao, Jun Zhao._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/P18-4009.pdf)
  
    >We present an event extraction framework to detect event mentions and extract events from the document-level financial news.
    
    
6. **Biomedical Event Extraction Using Convolutional Neural Networks and Dependency Parsing.**
  _Jari Bjorne, Tapio Salakoski._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/W18-2311.pdf)
  
    >We develop a convolutional neural network that can be used for both event and relation extraction. We use a linear representation of the input text, where information is encoded with various vector space embeddings. Most notably, we encode the parse graph into this linear space using dependency path embeddings.  
    
    
7. **Economic Event Detection in Company-Specific News Text.**
  _Gilles Jacobs, Els Lefever, Veronique Hoste._
  ACL 2018.
  [paper](https://www.aclweb.org/anthology/W18-3101.pdf)
  
    >This paper presents a dataset and supervised classification approach for economic event detection in English news articles.  
 

### 2017

1. **Exploiting Argument Information to Improve Event Detection via Supervised Attention Mechanisms.**
  _Shulin Liu, Yubo Chen, Kang Liu, Jun Zhao._
  ACL 2017.
  [paper](https://www.aclweb.org/anthology/P17-1164.pdf)
  
    >We propose to exploit argument information explicitly for ED via supervised attention mechanisms.   
    
    
2. **Automatically Labeled Data Generation for Large Scale Event Extraction.**
  _Yubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, Jun Zhao._
  ACL 2017.
  [paper](https://www.aclweb.org/anthology/P17-1038.pdf)
  
    >We propose to automatically label training data for event extraction via world knowledge and linguistic knowledge, which can detect key arguments and trigger words for each event type and employ them to label events in texts automatically.  
    
    
3. **English Event Detection With Translated Language Features.**
  _Sam Wei, Igor Korostil, Joel Nothman, Ben Hachey._
  ACL 2017.
  [paper](https://www.aclweb.org/anthology/P17-2046.pdf)
  
    >We propose novel radical features from automatic translation for event extraction.  


### 2016

1. **A Language-Independent Neural Network for Event Detection.**
  _Xiaocheng Feng, Lifu Huang, Duyu Tang, Bing Qin, Heng Ji, Ting Liu._
  ACL 2016.
  [paper](https://www.aclweb.org/anthology/P16-2011.pdf)
  
    > We develop a hybrid neural network to capture both sequence and chunk information from specific contexts, and use them to train an event detector for multiple languages without any manually encoded features.
  
  
2. **Event Nugget Detection with Forward-Backward Recurrent Neural Networks.**
  _Reza Ghaeini, Xiaoli Z. Fern, Liang Huang, Prasad Tadepalli._
  ACL 2016.
  [paper](https://www.aclweb.org/anthology/P16-2060.pdf)
  
    > We instead use forward-backward recurrent neural networks (FBRNNs) to detect events that can be either words or phrases. 


3. **Leveraging FrameNet to Improve Automatic Event Detection.**
  _Shulin Liu, Yubo Chen, Shizhu He, Kang Liu, Jun Zhao._
  ACL 2016.
  [paper](https://www.aclweb.org/anthology/P16-1201.pdf)
  
    > We propose a global inference approach to detect events in FN. Further, based on the detected results, we analyze possible mappings from frames to event-types. 


4. **Liberal Event Extraction and Event Schema Induction.**
  _Lifu Huang, Taylor Cassidy, Xiaocheng Feng, Heng Ji, Clare R. Voss, Jiawei Han, Avirup Sil._
  ACL 2016.
  [paper](https://www.aclweb.org/anthology/P16-1025.pdf)
  
    > We propose a brand new “Liberal” Event Extraction paradigm to extract events and discover event schemas from any input corpus simultaneously.
    
    
5. **RBPB: Regularization-Based Pattern Balancing Method for Event Extraction.**
  _Lei Sha, Jing Liu, Chin-Yew Lin, Sujian Li, Baobao Chang, Zhifang Sui._
  ACL 2016.
  [paper](https://www.aclweb.org/anthology/P16-1116.pdf)
  
    > This paper proposes a Regularization-Based Pattern Balancing Method (RBPB). Inspired by the progress in representation learning, we use trigger embedding, sentence-level embedding and pattern features together as our features for trigger classification so that the effect of patterns and other useful features can be balanced.    
    

### 2015

1. **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks.**
  _Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, Jun Zhao._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-1017.pdf)
  
    > This paper proposes a novel event-extraction method, which aims to automatically extract lexical-level and sentence-level features without using complicated NLP tools.


2. **Event Detection and Domain Adaptation with Convolutional Neural Networks.**
  _Thien Huu Nguyen, Ralph Grishman._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-2060.pdf)
  
    > We study the event detection problem using convolutional neural networks (CNNs) that overcome the two fundamental limitations of the traditional feature-based approaches to this task: complicated feature engineering for rich feature sets and error propagation from the preceding stages which generate these features.
    
    
3. **A Domain-independent Rule-based Framework for Event Extraction.**
  _Marco A. Valenzuela-Escarcega, Gus Hahn-Powell, Thomas Hicks, Mihai Surdeanu._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-4022.pdf)
  
    > We describe the design, development, and API of ODIN (Open Domain INformer), a domainindependent, rule-based event extraction (EE) framework.


4. **Disease Event Detection based on Deep Modality Analysis.**
  _Yoshiaki Kitagawa, Mamoru Komachi, Eiji Aramaki, Naoaki Okazaki, and Hiroshi Ishikawa._
  ACL 2015.
  [paper](https://www.aclweb.org/anthology/P15-3005.pdf)
  
    > This study proposes the use of modality features to improve disease event detection from Twitter messages, or “tweets”.
    


## EMNLP

### 2019

1. **HMEAE: Hierarchical Modular Event Argument Extraction.**
  _Xiaozhi Wang, Ziqi Wang, Xu Han, Zhiyuan Liu, Juanzi Li, Peng Li, Maosong Sun, Jie Zhou, Xiang Ren._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1584.pdf)
  
    > We propose a Hierarchical Modular Event Argument Extraction (HMEAE) model, to provide effective inductive bias from the concept hierarchy of event argument roles.

2. **Event Detection with Multi-Order Graph Convolution and Aggregated Attention.**
  _Haoran Yan, Xiaolong Jin, Xiangbin Meng, Jiafeng Guo, Xueqi Cheng._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1582.pdf)
  
    > This paper proposes a new method for event detection, which uses a dependency tree based graph convolution network with aggregative attention to explicitly model and aggregate multi-order syntactic representations in sentences.
    

3. **Event Detection with Trigger-Aware Lattice Neural Network.**
  _Ning Ding, Ziran Li, Zhiyuan Liu, Hai-Tao Zheng, Zibo Lin._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1033.pdf)
  
    > We propose a novel framework TLNN for event detection, which can simultaneously address the problems of trigger-word mismatch and polysemous triggers.


4. **Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction.**
  _Shun Zheng, Wei Cao, Wei Xu, Jiang Bian._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1032.pdf)
  
    >  We propose a novel end-to-end model, Doc2EDAG, which can generate an entity-based directed acyclic graph to fulfill the document-level EE(DEE) effectively.


5. **Entity, Relation, and Event Extraction with Contextualized Span Representations.**
  _David Wadden, Ulme Wennberg, Yi Luan, Hannaneh Hajishirzi._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1585.pdf)
  
    > Our framework (called DYGIE++) accomplishes all tasks by enumerating, refining, and scoring text spans designed to capture local (withinsentence) and global (cross-sentence) context.


6. **Neural Cross-Lingual Event Detection with Minimal Parallel Resources.**
  _Jian Liu, Yubo Chen, Kang Liu, Jun Zhao._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1068.pdf)
  
    >  We propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources.


7. **Financial Event Extraction Using Wikipedia-Based Weak Supervision.**
  _Liat Ein-Dor, Ariel Gera, Orith Toledo-Ronen, Alon Halfon, Benjamin Sznajder, Lena Dankin, Yonatan Bilu, Yoav Katz and Noam Slonim._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-5102.pdf)
  
    >  This work is in line with this latter approach, leveraging relevant Wikipedia sections to extract weak labels for sentences describing economic events.


8. **Cross-lingual Structure Transfer for Relation and Event Extraction.**
  _Ananya Subburathinam, Di Lu1, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1030.pdf)
  
    >  We exploit relation- and event-relevant language-universal features, leveraging both symbolic (including part-of-speech and dependency path) and distributional (including type representation and contextualized representation) information.


9. **Open Event Extraction from Online Text using a Generative Adversarial Network.**
  _Rui Wang, Deyu Zhou, Yulan He._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-1027.pdf)
  
    >  We propose an event extraction model based on Generative Adversarial Nets, called Adversarial-neural Event Model (AEM). 


10. **Extending Event Detection to New Types with Learning from Keywords.**
  _Viet Dac Lai, Thien Huu Nguyen._
  EMNLP 2019.
  [paper](https://www.aclweb.org/anthology/D19-5532.pdf)
  
    >  We introduce a novel feature-based attention mechanism for convolutional neural networks for event detection in the new formulation.
 
 
### 2018 

1. **Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation.**
  _Xiao Liu, Zhunchen Luo, Heyan Huang._
  EMNLP 2018.
  [paper](https://www.aclweb.org/anthology/D18-1156.pdf)
  
    > We propose a novel Jointly Multiple Events Extraction (JMEE) framework to jointly extract multiple event triggers and arguments by introducing syntactic shortcut arcs to enhance information flow and attention-based graph convolution networks to model graph information.


2. **Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms.**
  _Yubo Chen, Hang Yang, Kang Liu, Jun Zhao, Yantao Jia._
  EMNLP 2018.
  [paper](https://www.aclweb.org/anthology/D18-1158.pdf)
  
    > This paper proposes a novel framework for event detection, which can automatically extract and dynamically integrate sentence-level and documentlevel information and collectively detect multiple events in one sentence.
    

3. **Exploiting Contextual Information via Dynamic Memory Network for Event Detection.**
  _Shaobo Liu, Rui Cheng, Xiaoming Yu, Xueqi Cheng._
  EMNLP 2018.
  [paper](https://www.aclweb.org/anthology/D18-1127.pdf)
  
    > We proposed the TD-DMN model which utilizes the multi-hop mechanism of the dynamic memory network to better capture the contextual information for the event trigger detection task. 
    

4. **Event Detection with Neural Networks: A Rigorous Empirical Evaluation.**
  _J. Walker Orr, Prasad Tadepalli, Xiaoli Fern._
  EMNLP 2018.
  [paper](https://www.aclweb.org/anthology/D18-1122.pdf)
  
    > We present a novel GRU-based model that combines syntactic information along with temporal structure through an attention mechanism. 
    
  
5. **Similar but not the Same: Word Sense Disambiguation Improves Event Detection via Neural Representation Matching.**
  _Weiyi Lu, Thien Huu Nguyen._
  EMNLP 2018.
  [paper](https://www.aclweb.org/anthology/D18-1517.pdf)
  
      > We propose a method to transfer the knowledge learned on WSD to ED by matching the neural representations learned for the two tasks. 


### 2016 

1. **Modeling Skip-Grams for Event Detection with Convolutional Neural Networks.**
  _Thien Huu Nguyen, Ralph Grishman._
  EMNLP 2016.
  [paper](https://www.aclweb.org/anthology/D16-1085.pdf)
  
    > We propose to improve the current CNN models for ED by introducing the non-consecutive convolution. 


2. **Event Detection and Co-reference with Minimal Supervision.**
  _Haoruo Peng, Yangqiu Song, Dan Roth._
  EMNLP 2016.
  [paper](https://www.aclweb.org/anthology/D16-1038.pdf)
  
    > This paper proposes a novel event detection and co-reference approach with minimal supervision, addressing some of the key issues slowing down progress in research on events, including the difficulty to annotate events and their relations.
    
    
### 2015 

1. **Joint Event Trigger Identification and Event Coreference Resolution with Structured Perceptron.**
  _Jun Araki, Teruko Mitamura._
  EMNLP 2015.
  [paper](https://www.aclweb.org/anthology/D15-1247.pdf)
  
    > This paper proposes a document-level structured learning model that simultaneously identifies event triggers and resolves event coreference.


2. **Event Detection and Factuality Assessment with Non-Expert Supervision.**
  _Kenton Lee, Yoav Artzi, Yejin Choi, and Luke Zettlemoyer._
  EMNLP 2015.
  [paper](https://www.aclweb.org/anthology/D15-1189.pdf)
  
    > We studied event detection and scalar factuality prediction, demonstrating that non-expert annotator can, in aggregate, provide high-quality data and introducing simple models that perform well on each task. 



## NAACL

### 2019

1. **Adversarial Training for Weakly Supervised Event Detection.**
  _Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, Peng Li._
  NAACL 2019.
  [paper](https://www.aclweb.org/anthology/N19-1105.pdf)
  
    > We build a large event-related candidate set with good coverage and then apply an adversarial training mechanism to iteratively identify those informative instances from the candidate set and filter out those noisy ones.


2. **Event Detection without Triggers.**
  _Shulin Liu, Yang Li, Xinpeng Zhou, Tao Yang, Feng Zhang._
  NAACL 2019.
  [paper](https://www.aclweb.org/anthology/N19-1080.pdf)
  
    > We propose a novel framework dubbed as Type-aware Bias Neural Network with Attention Mechanisms (TBNNAM), which encodes the representation of a sentence based on target event types.
    

3. **Multilingual Entity, Relation, Event and Human Value Extraction.**
  _Manling Li, Ying Lin, Joseph Hoover, Spencer Whitehead, Clare R. Voss, Morteza Dehghani, Heng Ji._
  NAACL 2019.
  [paper](https://www.aclweb.org/anthology/N19-4019.pdf)
  
    > This paper demonstrates a state-of-the-art endto-end multilingual (English, Russian, and Ukrainian) knowledge extraction system that can perform entity discovery and linking, relation extraction, event extraction, and coreference.


4. **SEDTWik: Segmentation-based Event Detection from Tweets using Wikipedia.**
  _Keval M. Morabia, Neti Lalita Bhanu Murthy, Aruna Malapati, Surender S. Samant._
  NAACL 2019.
  [paper](https://www.aclweb.org/anthology/N19-3011.pdf)
  
    > This paper presents the problems associated with event detection from tweets and a tweet-segmentation based system for event detection called SEDTWik, an extension to a previous work, that is able to detect newsworthy events occurring at different locations of the world from a wide range of categories.


5. **Biomedical Event Extraction Based on Knowledge-driven Tree-LSTM.**
  _Diya Li, Lifu Huang, Heng Ji, Jiawei Han._
  NAACL 2019.
  [paper](https://www.aclweb.org/anthology/N19-1145.pdf)
  
    > We show the effectiveness of using a KB-driven tree-structured LSTM for event extraction in biomedical domain.
    
    
### 2018

1. **Semi-Supervised Event Extraction with Paraphrase Clusters.**
  _James Ferguson, Colin Lockard, Daniel S. Weld, Hannaneh Hajishirzi._
  NAACL 2018.
  [paper](https://www.aclweb.org/anthology/N18-2058.pdf)
  
    > We present a method for self-training event extraction systems by bootstrapping additional training data.   


2. **Neural Events Extraction from Movie Descriptions.**
  _Alex Tozzo, Dejan Jovanovic, Mohamed R. Amer._
  NAACL 2018.
  [paper](https://www.aclweb.org/anthology/W18-1507.pdf)
  
    > We formulate our problem using a recurrent neural network, enhanced with structural features extracted from syntactic parser, and trained using curriculum learning by progressively increasing the difficulty of the sentences.
    
    
### 2016

1. **Joint Event Extraction via Recurrent Neural Networks.**
  _Thien Huu Nguyen, Kyunghyun Cho, Ralph Grishman._
  NAACL 2016.
  [paper](https://www.aclweb.org/anthology/N16-1034.pdf)
  
    > We propose to do event extraction in a joint framework with bidirectional recurrent neural networks, thereby benefiting from the advantages of the two models as well as addressing issues inherent in the existing approaches.    


2. **Joint Extraction of Events and Entities within a Document Context.**
  _Bishan Yang, Tom Mitchell._
  NAACL 2016.
  [paper](https://www.aclweb.org/anthology/N16-1033.pdf)
  
    > We propose a novel approach that models the dependencies among variables of events, entities, and their relations, and performs joint inference of these variables across a document. 


3. **Bidirectional RNN for Medical Event Detection in Electronic Health Records.**
  _Abhyuday N Jagannatha, Hong Yu._
  NAACL 2016.
  [paper](https://www.aclweb.org/anthology/N16-1056.pdf)
  
    > We have shown that RNNs models like LSTM and GRU are valuable tools for extracting medical events and attributes from noisy natural language text of EHR notes. 


### 2015

1. **Diamonds in the Rough: Event Extraction from Imperfect Microblog Data.**
  _Ander Intxaurrondo, Eneko Agirre, Oier Lopez de Lacalle, Mihai Surdeanu._
  NAACL 2015.
  [paper](https://www.aclweb.org/anthology/N15-1066.pdf)
  
    > We introduce a distantly supervised event extraction approach that extracts complex event templates from microblogs.
    
    

## COLING

### 2018

1. **Open-Domain Event Detection using Distant Supervision.**
  _Jun Araki, Teruko Mitamura._
  COLING 2018.
  [paper](https://www.aclweb.org/anthology/C18-1075.pdf)
  
    > This paper introduces open-domain event detection, a new event detection paradigm to address issues of prior work on restricted domains and event annotation.


### 2016

1. **Leveraging Multilingual Training for Limited Resource Event Extraction.**
  _Andrew Hsi, Yiming Yang, Jaime Carbonell, Ruochen Xu._
  COLING 2016.
  [paper](https://www.aclweb.org/anthology/C16-1114.pdf)
  
    > We propose a new event extraction approach that trains on multiple languages using a combination of both language-dependent and language-independent features, with particular focus on the case where target domain training data is of very limited size.


2. **Incremental Global Event Extraction.**
  _Alex Judea, Michael Strube._
  COLING 2016.
  [paper](https://www.aclweb.org/anthology/C16-1215.pdf)
  
    > We present an incremental approach to make the global context of the entire document available to the intra-sentential, state-of-the-art event extractor.



## AAAI

### 2020

1. **Image Enhanced Event Detection in News Articles.**
  _Meihan Tong, Shuai Wang, Yixin Cao, Bin Xu, Juaizi Li, Lei Hou, Tat-Seng Chua._
  AAAI 2020.
  [paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-TongM.4888.pdf)
  
    > In this paper, we first contribute an image dataset supplement to ED benchmarks (i.e., ACE2005) for training and evaluation. We then propose a novel Dual Recurrent Multimodal Model, DRMM, to conduct deep interactions between images and sentences for modality features aggregation. 


2. **A Human-AI Loop Approach for Joint Keyword Discovery and Expectation Estimation in Micropost Event Detection.**
  _Akansha Bhardwaj, Jie Yang, Philippe Cudre-Mauroux._
  AAAI 2020.
  [paper](https://arxiv.org/pdf/1912.00667.pdf)
  
    > This paper introduces a Human-AI loop approach to jointly discover informative keywords for model training while estimating their expectation.
    
    
### 2019    
    
1. **Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection.**
  _Jian Liu, Yubo Chen, Kang Liu._
  AAAI 2019.
  [paper](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4649)
  
    > We propose an adversarial imitation based knowledge distillation approach, for the first time, to tackle the challenge of acquiring knowledge from rawsentences for event detection.   


2. **One for All: Neural Joint Modeling of Entities and Events.**
  _Trung Minh Nguyen, Thien Huu Nguyen._
  AAAI 2019.
  [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4661)
  
    > We propose a novel model to jointly perform predictions for entity mentions, event triggers and arguments based on the shared hidden representations from deep learning. 


### 2018   
    
1. **Graph Convolutional Networks with Argument-Aware Pooling for Event Detection.**
  _Thien Huu Nguyen, Ralph Grishman._
  AAAI 2018.
  [paper](https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf)
  
    > We investigate a convolutional neural network based on dependency trees to perform event detection.
    

2. **Event Detection via Gated Multilingual Attention Mechanism.**
  _Jian Liu, Yubo Chen, Kang Liu, Jun Zhao._
  AAAI 2018.
  [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16371/16017)
  
    > We propose a novel multilingual approach — dubbed as Gated MultiLingual Attention (GMLATT) framework — to address the two issues simultaneously. 
    
    
3. **Jointly Extracting Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction.**
  _Lei Sha, Feng Qian, Baobao Chang, Zhifang Sui._
  AAAI 2018.
  [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16222/16157)
  
    > We propose a novel dependency bridge recurrent neural network (dbRNN) for event extraction.     


4. **Scale Up Event Extraction Learning via Automatic Training Data Generation.**
  _Ying Zeng, Yansong Feng, Rong Ma, Zheng Wang, Rui Yan, Chongde Shi, Dongyan Zhao._
  AAAI 2018.
  [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16119/16173)
  
    > This paper has presented a novel, fast approach to automatically construct training data for event extraction with little human involvement, which in turn allows effective event extraction modeling.
    

### 2016   
    
1. **From Tweets to Wellness: Wellness Event Detection from Twitter Streams.**
  _Mohammad Akbari, Xia Huc, Nie Liqiang, Tat-Seng Chua._
  AAAI 2016.
  [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11931/11568)
  
    > We proposed a learning framework that utilizes content information of microblogging texts as well as the relation between event categories to extract PWE from users social posts.
    
    
    
## Other Conferences

### CoNLL 2019

1. **Exploiting the Entity Type Sequence to Benefit Event Detection.**
  _Yuze Ji, Youfang Lin, Jianwei Gao, Huaiyu Wan._
  CoNLL 2019.
  [paper](https://www.aclweb.org/anthology/K19-1057.pdf)
  
    > We propose a novel ED approach which learns sequential features from word sequences and entity type sequences separately, and combines these two types of sequential features with the help of a trigger-entity interaction learning module.     
    

2. **Contextualized Cross-Lingual Event Trigger Extraction with Minimal Resources.**
  _Meryem M’hamdi , Marjorie Freedman, Jonathan May._
  CoNLL 2019.
  [paper](https://www.aclweb.org/anthology/K19-1061.pdf)
  
    > We treat event trigger extraction as a sequence tagging problem and propose a cross-lingual framework for training it without any hand-crafted features.
    
    
    
## Journals

### arxiv 2020 

1. **Event Detection with Relation-Aware Graph Convolutional Networks.**
  _Shiyao Cui, Bowen Yu, Tingwen Liu, Zhenyu Zhang, Xuebin Wang, Jinqiao Shi._
  arxiv 2020 [cs.CL].
  [paper](https://arxiv.org/pdf/2002.10757.pdf)
  
    >  We investigate a novel architecture named Relation-Aware GCN (RA-GCN), which efficiently exploits syntactic relation labels and models the relation between words specifically.     


2. **Joint Event Extraction along Shortest Dependency Paths using Graph Convolutional Networks.**
  _Ali Balali, Masoud Asadpour, Ricardo Campos, Adam Jatowt._
  arxiv 2020 [cs.LG].
  [paper](https://arxiv.org/ftp/arxiv/papers/2003/2003.08615.pdf)
  
    >  We propose a novel joint event extraction framework that aims to extract multiple event triggers and arguments simultaneously by introducing shortest dependency path (SDP) in the dependency graph. 
    
    
3. **MAVEN: A Massive General Domain Event Detection Dataset.**
  _Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi Li, Peng Li, Yankai Lin, Jie Zhou._
  arxiv 2020 [cs.CL].
  [paper](https://arxiv.org/pdf/2004.13590.pdf)
  
    > We present a MAssive eVENt detection dataset (MAVEN), which contains 4,480 Wikipediadocuments, 117,200 event mention instances, and 207 event types.
    

4. **Event Extraction by Answering (Almost) Natural Questions.**
  _Xinya Du, Claire Cardie._
  arxiv 2020 [cs.CL].
  [paper](https://arxiv.org/pdf/2004.13625.pdf)
  
    > We introduce a new paradigm for event extraction by formulating it as a question answering (QA) task, which extracts the event arguments in an end-to-end manner.


### arxiv 2019 [cs.CL]

1. **Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection.**
  _Shumin Deng, Ningyu Zhang, Jiaojian Kang, Yichi Zhang, Wei Zhang, Huajun Chen._
  arxiv 2019 [cs.CL].
  [paper](https://arxiv.org/pdf/1910.11621.pdf)
  
    >  We propose a Dynamic-Memory-Based Prototypical Network (DMB-PN), which exploits Dynamic Memory Network (DMN) to not only learn better prototypes for event types, but also produce more robust sentence encodings for event mentions.
    
    
2. **Extending Event Detection to New Types with Learning from Keywords.**
  _Viet Dac Lai, Thien Huu Nguyen._
  arxiv 2019 [cs.LG].
  [paper](https://arxiv.org/pdf/1910.11368.pdf)
  
    >  We study a novel formulation of event detection that describes types via several keywords to match the contexts in documents. This facilitates the operation of the models to new types. We introduce a novel feature-based attention mechanism for convolutional neural networks for event detection in the new formulation.   
    

3. **Event Detection in Twitter: A Keyword Volume Approach.**
  _Ahmad Hany Hossny, Lewis Mitchell._
  arxiv 2019 [cs.SI].
  [paper](https://arxiv.org/pdf/1901.00570.pdf)
  
    >  We propose an efficient method to select the keywords frequently used in Twitter that are mostly associated with events of interest such as protests. 
    

4. **Financial Event Extraction Using Wikipedia-Based Weak Supervision.**
  _Liat Ein-Dor, Ariel Gera, Orith Toledo-Ronen, Alon Halfon, Benjamin Sznajder, Lena Dankin, Yonatan Bilu, Yoav Katz, Noam Slonim._
  arxiv 2019 [cs.CL].
  [paper](https://arxiv.org/pdf/1911.10783.pdf)
  
    >  This work is in line with this latter approach, leveraging relevant Wikipedia sections to extract weak labels for sentences describing economic events.
    

5. **CONTEXT AWARENESS AND EMBEDDING FOR BIOMEDICAL EVENT EXTRACTION.**
  _Shankai Yan, Ka-Chun Wong._
  arxiv 2019 [cs.CL].
  [paper](https://arxiv.org/pdf/1905.00982.pdf)
  
    >   We proposed a bottom-up event detection framework using deep learning techniques. We built an LSTM-based model VecEntNet to construct argument embeddings for each recognized entity.
