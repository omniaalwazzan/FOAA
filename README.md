# FOAA: Flattened Outer Arithmetic Attention For Multimodal Tumor Classification

Fusion models implemented in our paper 'FOAA: Flattened Outer Arithmetic Attention For Multimodal Tumor Classification' Omnia Alwazzan, Ioannis Patras, Gregory Slabaugh, ISBI 2024


<p align="center">
  <img width="1000" height="500" src="https://github.com/omniaalwazzan/FOAA/blob/main/ISBI_pipline.png">
</p>

### The paper has been accepted by The IEEE International Symposium on Biomedical Imaging (ISBI) and will be available soon.

This paper serves as an extension to our previous methodology, [MOAB](https://github.com/omniaalwazzan/MOAB). Here, we introduce a novel approach where the MOAB block is integrated into the attention mechanism, replacing the standard dot product operation between Query and Key in a cross-attention manner for multimodality fusion. For more comprehensive insights, please take a look at the paper's arXiv [link](https://arxiv.org/abs/2403.06339).
This repository provides all functional fusion methods that can be applied to any domain with any CNN. 

* #### Available fusion models
  * The file [FOAA_one_branch](https://github.com/omniaalwazzan/FOAA/blob/main/FOAA_one_branch.py) can be adjusted by a user to incorporate
    * Cross Attention Outer Addition (OA)
    * Cross Attention Outer Product (OP)
    * Cross Attention Outer Division (OD)
    * Cross Attention Outer Subtraction (OS)
  * Cross OA+OP
  * Cross OA+OP+OS
  * CNN FOAA SA
  * FOAA

