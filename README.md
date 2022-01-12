# Continuous Learning In Automatic Speech Recognition (ASR)

Code repo 

## TL;DR

Using language models and different word representations to allow continuous learning in automatic speech recognition.
We focus on new/rare words to improve the overall performance of SOTA speech recognition systems like Wav2Vec2.

## Abstract

Today's speech recognition models achieve good transcription performance on most publicly accessible test sets using state-of-the-art deep learning technologies. The systems even attain a human-like error rate. However, in a real-world setting, further challenges must be overcome. One significant issue is that human vocabularies are always changing. This leads to the frequent occurrence of words that did not or only appear sporadically in the training data.
In this work, we perform an in-depth investigation on the ability of ASR systems to transcribe rare and unknown words. 
Continuous learning is one possible strategy for tackling this issue. To evaluate the ability of the ASR system and its individual components to accomplish continuous learning, we propose a simulated continuous learning scenario. We show that continuous learning of the language model component is crucial for a strong system, and that the acoustic component can generalize effectively to unknown words using a widely used wav2vec2-based model.  
We have shown that depending on the chosen word representation, the model focuses on new or rare words. Our best word error rates for new and rare words were achieved by a text-based fine-tuning of a sub-word language model. It was irrelevant whether the acoustic model had been previously fine-tuned with audio samples on the new words.

