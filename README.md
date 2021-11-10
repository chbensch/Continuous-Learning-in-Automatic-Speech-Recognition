# Continuous Learning In Automatic Speech Recognition (ASR)

Code repo for my thesis

## TL;DR

Using language models and different word representations to allow continuous learning in automatic speech recognition.
We focus on new/rare words to improve the overall performance of SOTA speech recognition systems like Wav2Vec2.

## Abstract

Human vocabulary is evolving constantly. Social or cultural changes were often the  underlying cause in the past. However, technology is the driving factor nowadays. Words like "botnet", "selfie" or "blogging" represent an indispensable part of today’s vocabulary and demonstrate the introduction of novel words, which have been of no significance two decades ago. Furthermore, redefinitions of words such as "cookie", "hotspot" or "cloud" expose how our language has evolved in recent years.

Therefore, continuous learning became a crucial factor in Automatic Speech Recognition (ASR). Today's speech recognition models achieve good recognition rates on known words. Thus, only rare- (one-/few-shot-learning) or new words (zero-shot-learning) are a significant challenge. In consequence, recognition rates of these word groups play a critical role when increasing overall ASR system performance. This work explores and evaluates different methods to improve automatic speech recognition of new and rare words.

The results show that the use of an n-gram Language Model (LM) improves the overall performance of our baseline ASR model, by decreasing its WER by 1.8\%. Especially the recognition of rare words was significantly improved. For example, words seen once (one-shot-learning) improved in accuracy from 57\% up to 84\%. However, due to the fixed vocabulary in the language model, zero-shot learning (recognition of new words) was eliminated.

To include these unseen words, the word representation was changed. This was accomplished by training the acoustic baseline model on sub-words using a modified dataset. This did not have a major impact on the WER of the model.  However, this gave us the opportunity to train an n-gram language model based on our newly defined sub-word vocabulary, which significantly improved the overall performance of the model, as the WER decreased by 4.0\% to 10.7\%. In contrast to the word-based language model, the accuracy of the unseen words decreased only slightly. On the other hand, the recognition of the one-shot-learning words has also increased, up to 15\% for more frequent words. While the increase in accuracy is not as significant as for the word-based language model, the much lower WER shows that the model benefits from the subword approach. This can be attributed to the fact that unseen or rare words can now be composed of familiar word parts and are therefore better recognized.

