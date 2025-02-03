1 Introduction
Image encoders have evolved significantly with the emergence of **self-supervised learning** techniques. Recent advancements like **DINO** and **DINOv2** have demonstrated strong performance in learning feature representations with- out labeled data. In this post, we explore self-attention, transformers, and how they lead to advanced vision encoders.
2 Self-Attention Mechanism
Self-attention is a key component of transformer architectures and is responsi- ble for capturing **long-range dependencies** in input sequences. It enables the model to assign different importance weights to various parts of the input, allowing for more dynamic feature extraction.
2.1 Mathematical Formulation
Given an input sequence of token embeddings X, we compute **Query (Q), Key (K), and Value (V)** matrices:
Q=XWQ, K=XWK, V =XWV (1)
where WQ, WK, and WV are learnable weight matrices. These matrices help in computing attention scores and refining feature representations.
The **scaled dot-product attention** is computed as:
QKT 
Attention(Q, K, V ) = softmax √
where dk is the dimension of the key vectors, which stabilizes gradient updates.
1
 dk
V (2)
2.2 Multi-Head Attention
Instead of computing a single attention function, transformers use **multi-head attention**, which allows the model to focus on different parts of the sequence simultaneously. The attention heads are computed as:
MultiHead(Q, K, V ) = Concat(head1, head2, ..., headh)WO (3) where each attention head computes attention independently before being con-
catenated and linearly transformed.
3 Transformer Encoder Architecture
The **Transformer encoder** consists of multiple stacked layers, each contain- ing:
• Multi-head self-attention mechanism
• Feed-forward network (FFN)
• Residual connections and Layer Normalization
Each encoder block follows this sequence:
1. Input embeddings + Positional Encoding 2. Multi-head self-attention
3. Layer Normalization
4. Feed-forward network
5. Layer Normalization (again)
The encoder processes the input in parallel, unlike RNNs, making it more effi- cient for long sequences.
3.1 Masked Language Modeling (MLM)
MLM is commonly used in natural language processing (NLP) tasks, where a fraction of the input tokens are randomly masked, and the model learns to predict the missing tokens based on their context. The objective function for MLM is:
LMLM = − X logP(xi|x\i) (4) i∈M
where M represents the set of masked positions, and P(xi|x\i) is the probability of correctly predicting the masked token.
2

4 Vision Transformers (ViTs)
ViTs apply **self-attention** to image data, treating images as sequences of patches rather than using convolutional operations.
4.1 Patch Extraction and Encoding
An image is split into patches, flattened, and passed through a **linear projec- tion**:
Xpatch = WE · Xinput + bE (5) where WE is the embedding matrix and bE is a bias term. These embeddings
are then processed using self-attention layers to extract contextual information.
5 Contrastive Language-Image Pretraining (CLIP)
CLIP is a multimodal self-supervised learning model designed to learn joint representations of images and text. It is trained on a large dataset of image- text pairs using a contrastive learning approach.
5.1 Architecture
CLIP consists of two main components:
• An image encoder (ResNet or Vision Transformer) • A text encoder (Transformer-based model)
Both encoders project images and text into a shared embedding space, where semantically similar images and text have higher similarity scores.
5.2 Training Objective
CLIP uses a contrastive learning objective where it maximizes the cosine sim- ilarity between matching image-text pairs while minimizing similarity between mismatched pairs:
X ecos(Ii ,Ti )/τ
LCLIP = −
where Ii and Ti are corresponding image-text pairs, and τ is a temperature
log Pj ecos(Ii,Tj)/τ (6) parameter that controls the sharpness of the distribution.
 i
3

5.3 Applications of CLIP
• Zero-shot image classification
• Image retrieval and search
• Generating text-based descriptions of images • Few-shot learning for downstream vision tasks
CLIP’s ability to understand images and text jointly makes it a powerful model for many vision-language applications.
6 DINO: Self-Supervised Vision Encoding
DINO (Self-Distillation without Labels) is a **self-supervised learning method** that trains a **student** network to predict the output of a momentum **teacher** network.
6.1 Key Features of DINO
• Works with both CNNs and ViTs
• Learns representations using **contrastive learning**
• Uses a momentum teacher-student setup
• Multi-crop augmentation for efficient feature learning
• Achieves competitive performance in unsupervised representation learning
6.2 DINO Training Process
DINO minimizes the cross-entropy loss between teacher and student outputs: LDINO =−XpTi logpSi (7)
i
where pTi and pSi are teacher and student predictions. The teacher network
updates using an exponential moving average of the student’s weights.
7 DINOv2: Improved Vision Encoding
DINOv2 extends DINO with **Masked Image Modeling (MIM)** and **Self- Supervised Instance Discrimination**, improving its ability to capture both local and global features.
4

7.1 Key Improvements in DINOv2
• Introduces **masked image modeling (MIM)** to learn local textures
• Uses **multi-crop augmentation** to enhance representation learning
• No reliance on **text supervision** (unlike CLIP)
• Improved transferability to **dense prediction tasks** (e.g., segmenta- tion)
• Asymmetric teacher-student design for more effective training
• Incorporates KoLeo regularization to improve feature uniformity
7.2 Masked Image Modeling (MIM)
MIM extends the concept of MLM to images by randomly masking patches in an image and training the model to reconstruct the missing information. The objective function for MIM is:
LMIM = − X logP(Ii|I\i) (8) i∈M
where Ii represents the masked image patch, and I\i denotes the observed patches.
7.3 Training Objectives
DINOv2 uses a **hybrid objective**:
LDINOv2 = λ1LMIM + λ2LInstance (9)
where:
• LMIM is the loss for **masked image modeling**, focusing on reconstruct- ing masked patches
• LInstance is the loss for **instance discrimination**, ensuring robust fea- ture alignment across augmentations
DINOv2 leverages a **vision-only training approach**, making it highly effec- tive for self-supervised learning in image-based tasks.
8 Conclusion
DINO and DINOv2 represent significant advances in **self-supervised learning for vision tasks**. Their ability to learn rich representations **without labeled data** makes them valuable for diverse applications such as **image classifica- tion, retrieval, segmentation, and biomedical imaging**.
5
