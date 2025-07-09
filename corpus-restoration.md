# Open Books Restoration
The two open book copora available on huggingface lack proper punctuation or capitalization. I discovered this issue as I was training an autoregressive model using the corpus used to train BERT. My initual intuition had been that this clean corpus would quickly get to low loss. While it is true that the model trained quickly, it was highly specialized to the simple formatting of the training corpus. In parallel, I retrained a model using fineweb-edu and started this project to restore Open Book's structure by fine tuning a base model.

## Project Goals

1. Work with pre-trained / opensource models
2. Use existing APIs for ML scafolding (data loading, training, model save / restore, logging)
3. Leverage SFT to solve novel problem


## Comparing Base Models 
| Attribute | **[BERT](https://huggingface.co/google-bert/bert-base-uncased)** | **[GPT-2](https://huggingface.co/openai-community/gpt2)** | **[T5](https://huggingface.co/google-t5/t5-base)** | **[BART](https://huggingface.co/facebook/bart-base)** |
|-----------|----------|-----------|---------|-----------|
| **Paper** | [Devlin et al. 2018](https://arxiv.org/abs/1810.04805) | [Radford et al. 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [Raffel et al. 2019](https://arxiv.org/abs/1910.10683) | [Lewis et al. 2019](https://arxiv.org/abs/1910.13461) |
| **Type** | Encoder-only | Decoder-only | Encoder-decoder | Encoder-decoder |
| **Parameters** | 110M, 340M | 117M, 345M, 762M, 1.5B | 60M, 220M, 770M, 3B, 11B | 140M, 400M |
| **Positional Encoding** | Learned absolute | Learned absolute | Relative | Learned absolute |
| **Pretraining Data** | Books + Wikipedia | WebText (40GB) | C4 (750GB) | Books + Wikipedia + news |
| **Pretraining Objective** | Masked language modeling + Next sentence prediction | Next token prediction | Span corruption | Denoising (text infilling, sentence shuffling) |
| **Vocabulary Size** | 30k | 50k | 32k | 50k |
| **Context Length** | 512 | 1,024 | 512 | 1,024 |
| **Attention Pattern** | Bidirectional | Causal (unidirectional) | Encoder: bidirectional, Decoder: causal | Encoder: bidirectional, Decoder: causal |
| **Generation Capability** | None (requires task reformulation) | Native autoregressive | Seq2seq generation | Seq2seq generation |
| **Corruption During Pretraining** | 15% random masking | None | Span masking | Text infilling, deletion, permutation |
| **Community Support** | 22k likes | 18k likes | 29k likes | 9k likes |
| **Approach for Task** | BIO tagging with classification heads (B-PUNCT, I-PUNCT, ...) | Fine-tune on degraded→restored pairs with autoregressive generation | Fine-tune with "restore text:" prefix using seq2seq framework | Fine-tune on text infilling task similar to pretraining |
| **Summary** | Hard - no native generation capability, would need token classification | Suboptimal - unidirectional context misses future information | Works - text-to-text framework matches task | Works - denoising pretraining aligns well with restoration |

## Key Terms

**Span corruption**: T5's pretraining objective where contiguous sequences of tokens are replaced with special sentinel tokens (`<extra_id_0>`, `<extra_id_1>`). The model learns to predict the original spans. Example: "The cat sat on the mat" -> "The cat `<extra_id_0>` mat" with target "`<extra_id_0>` sat on the `<extra_id_1>`".

**Text infilling**: BART's pretraining task where arbitrary spans of text are deleted and the model must reconstruct the complete original sequence. Unlike span corruption, no sentinel tokens mark deletion positions - the model just infers what's missing from context.

**Masked language modeling (MLM)**: BERT's pretraining where 15% of tokens are randomly masked and the model predicts the original tokens. Uses `[MASK]` tokens during training but sees clean text during fine-tuning.

**Next sentence prediction (NSP)**: BERT's secondary objective predicting whether two sentences appeared consecutively in the corpus. 

**Autoregressive**: Generation approach where the model predicts the next token given all previous tokens. Standard for language models like GPT-2 which only sees left context.

## Technical Strategy

### SFT Objectives

I aimed to:
1. Split sentences using punctuation and spaces.
2. Restore capitalization to named entities (proper nouns)
3. Use commas, appostrophes, periods, question marks, and exlimination points correctly.
4. Retain any deliberate punctuation while also removing extra spacing.

### Base Model
T5 and BART are the most suitable base models for this task, with BART having a slight edge given it's training includes denoising. However, T5 is a model I'm familiar with at Google and it is also much more widely adopted by the open source community. I choose T5 as my base model.

### Training Data

### Evaluation

## Solution

## Learnings
1. Unsurprisingly, high quality data that provided enough signal to the SFT process to meet my objectives was critical
2. I noticed that even though the restoration worked as I wanted and expected, there were other artifacts such as added spaces that were not cleaned up. This was a let down at first, but given that these examples were not in my training data, not surprising. I corrected the trainig data to include deletion.
3. SFT was surprisingly quick. While my GPT2 model required more than 24 hours to train, SFT completed in minutes.

### Qualitative Evaluation
