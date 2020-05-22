# Papers for Event Extraction
Papers from top conferences and journals for event extraction in recent years

# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [ACL](#ACL)
- [EMNLP](#EMNLP)
- [NAACL](#NAACL)
- [COLING](#COLING)
</p></blockquote></details>

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
  
  </i></summary><blockquote><p align="justify">
  We propose a method to transfer the knowledge learned on WSD to ED by matching the neural representations learned for the two tasks.
  </p></blockquote></details>
    
    
