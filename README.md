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
 
CLIP can be used for a wide range of tasks that require understanding and processing of both images and text. For example, given an image, Clip can generate textual descriptions or classify it into categories. Conversely, given a text prompt, Clip can retrieve relevant images or rank a set of images based on their relevance to the text.

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
```
find_matches(model, 
             image_embeddings,
             query="one dog sitting on the grass",
             image_filenames=valid_df['image'].values,
             n=9)
```
![Figure_1](https://github.com/iljf/Assignment_CLIP/assets/94291960/098f6e20-3a02-40f8-9fe8-184ed951267e)

# Modified (FInal Exam)
## Possible way to improve the model
### Data Augmentation
- Since the original CLIP model is pre-trained based on 400 million data and the flcker-8k dataset (the one we used) only 8,091 data, I assumed that More data should be used to enhance the quality of the model
- I thought that Ensembling Generative model and CLIP will give me better result due to CLIP being a Pre-trained model that learns from cosine similarity of image and captions of the image; CLIP model can adjust better by learning from the loss of the generative model while calculating it's loss 
- The way I tried is to simply ensemble GAN with CLIP to Create fake images, add the fake image to the original dataset before going through process of Encoding; leading to maximize the number of the data
- Train the data using GAN, load the weight of the pre-trained GAN and apply the data transformation to the original dataset. Then generate augmented images using only the Generator in GAN.

### Modified Pesudocode
```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# G             - GAN ***
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I+G) #[n, d_i] ***
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
* Lines with *** at the end is modified
* GAN.py is added for the modified version

# Result
### Learning from Failure
Being straight Forward, I worked on the code for 2 weeks and couldn't find a way to fix the errors I was having (lacking of coding skills and lacking of understanding the model)
- like a mentioned before CLIP uses its image+text(captions) to pre-train
- Adding only images generated by GAN will not improve the result of the Model (Maybe it will lead to a poor performance since there are no matching caption with generated image )
- I could not find the way to generate captions(text) without me hand labeling each image (Tried to Augment text data with BERT but failed because it generated text crazy randomly)
- I could not fine the way to add BCE loss from the GAN and CE loss from the CLIP mathematically (did not fully understand how add them through code)
- I was really frustrated that I couldn't elaborate my modifying ideas because of my lack of skills but I believe What doesn't kill you make you stronger

### Things to improve
- Understanding the model's code better before improving
- skills on pytorch
- Try to use StyleGAN for subset to check the performance on Zero-shot-learning
- Study on ensembling models and how they are ensembled by code