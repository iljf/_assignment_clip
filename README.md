# Introduction
### Learning Transferable Visual Models From Natural Language Supervision, CLIP
![Clip](https://github.com/iljf/Assignment_CLIP/assets/94291960/78e0f6a9-0a76-4dd1-a3cf-71d86e73770a)

### CLIP: Contrastive Language-Image Pre-training

https://arxiv.org/pdf/2103.00020.pdf

Contrastive Language-Image Pre-Training (CLIP) is proposed to have the pre-training task of predicting which caption to learn image representations from scratch on a dataset of 400 million (image, text) pairs.

Unlike traditional models that are typically trained on either text or image data, Clip is designed to understand and generate meaningful representations of both text and images simultaneously. The underlying idea behind Clip is to leverage large-scale datasets containing image and text pairs to learn a shared embedding space, where similar images and their corresponding descriptions are placed close to each other.
- CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N²-N incorrect pairings.
- CLIP is trained from scratched, and does not use any non-linear projection layers, instead only a linear projection is used to map from each encoder’s representation to the multi-modal embedding space.
- Image encoder : 5 ResNets and 3 ViTs with some modifications
- Text encoder  : Transformer as a base size, 63M-parameter 12-layer 512-wide model with 8 attention heads
- All models are trained for 32 epochs. A very large minibatch size of 32,768 is used. Mixed-Precision Training. The calculation of embedding similarities was also sharded with individual GPUs
 
The resulting model can then be used for a wide range of tasks that require understanding and processing of both images and text. For example, given an image, Clip can generate textual descriptions or classify it into categories. Conversely, given a text prompt, Clip can retrieve relevant images or rank a set of images based on their relevance to the text.

### Pseudocode
```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T)  #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = 12_normalize(np.dot(I_f, W_i), axis=1)
T_e = 12_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) /2
```
*clip.py is coded based on pseudocode
- given a batch of N (image, text) pairs
- predict which of the N x N possible pairs across a batch actually occured
- CLIP learns the multi-modal embedding space by jointly training an image and text encoder to:
    - maximize the cosine similarity of the N real embedding pairs
    - minimize the cosine similarity of the N^2 - N incorrect embedding pairs
- optimize a symmetric cross entropy loss over the similarity scores

# Installation
- written in Window os, python 3.9
- other requirements are in reqiurements.txt
```
pip install -r requirements.txt
```
- Datasets Flicker-8k, https://www.kaggle.com/datasets/adityajn105/flickr8k
  - After downloading the datasets, resave captions.txt to captions.csv
  - in config.py, change the image path and captions path to your dataset location
```
image_path = "C:/Users/wlwrl/PycharmProjects/Assignment_CLIP/Datasets/Flicker-8k/Images"
captions_path = "C:/Users/wlwrl/PycharmProjects/Assignment_CLIP/Datasets/Flicker-8k"
```

# Run
- For traning process, simply run 'main.py'
- For inference, run 'inference.py'

### Models

ResNet50 For Image-encoder, DistiLBERT as text-encoder
```
model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200
```
For projection head, used for both image and text encoders
```
num_projection_layers = 1
projection_dim = 256
dropout = 0.1
Used parameters
```

Used parameters
```
batch_size = 32
num_workers = 4
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 1
```

### Inference
With model, image_embeddings, and text query; It displays the most relevant images from the validation set
