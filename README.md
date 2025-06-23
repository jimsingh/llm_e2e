# LLM End-to-End: An Experimentation and Learning Framework for Decoder-Style Language Models

## Abstract

Large language models (LLMs) have become increasingly accessible and put into production across diverse applications. While the adaptability of modern pretrained models is remarkable, it is crucial to understand and debug the fundamentals of how these models work to apply them effectively. This project is a complete implementation and instrumentation framework for decoder-style language models, with a focus on the well-understood GPT-2 architecture [1]. The key components are modularized to enable systematic exploration of model behavior under different architectural and training hyperparameters. Through the extensibility of each component, the design serves as both an educational resource and a platform for conducting experiments on autoregressive language modeling. As a demonstration, I pre-train my own GPT-2 model from scratch, extract and visualize attention weights to highlight how the model interprets patterns of text, and show how the model can be aligned with human preferences.

## Introduction

Understanding the inner workings of large language models is critical as these systems play a more prominent role in a wide range of applications, many going well beyond simple natural language processing and token prediction. While pre-trained models are readily available, the ability to build, train, and analyze these models from first principles provides invaluable insights into their behavior, limitations, and optimization.

This project has five key objectives:

1. **Architecture Replication**: Reproduce the GPT-2 model architecture [1] in PyTorch with clear, documented code that connects theoretical concepts to practical implementation. Verify the implementation by loading and running OpenAI's pre-trained weights.

2. **Training Framework Development**: Build an end-to-end training pipeline that including efficient data loaders optimized for next-token prediction, streaming dataset support for large-scale training, integration with [Weights & Biases](https://wandb.ai/) for ML Ops and experiment tracking, and model state checkpointing for restartability.

3. **Representation Analysis**: Visualize learned representations, with particular emphasis on attention patterns [4], to provide insights into how the model processes and relates tokens within its context window.

4. **Fine-tuning and Alignment**: Apply supervised fine-tuning (SFT) and alignment techniques to adapt the model to human preferences, demonstrating how base language models can be shaped for specific tasks and behaviors.

5. **Performance**: Implement the highest ROI strategies to significantly improve model training and inferences performance including word alignment, quantization, KV caching, and quantization.

## Training

### Pre-training from Scratch

I pre-trained a 33.6M parameter GPT-2 from scratch using a curated Wikipedia + BookCorpus dataset. This corpus was originally prepared for BERT training, with all text lowercased, markdown removed, and special punctuation cleaned. My hypothesis: could a cleaner, more consistent corpus enable a model 4x smaller than GPT-2 124M to achieve similar language understanding? The preprocessed text—free from formatting distractions and inconsistent capitalization—should theoretically require less model capacity to learn core linguistic patterns.

### Training Dynamics

<img src="assets/validation_loss.png" alt="Validation Loss" width="600"/>

The validation loss chart reveals three distinct phases:

1. **High Learning Rate Phase** (LR=0.009, Steps 0-100k): Rapid initial descent from ~9.4 to ~7.0
2. **Manual Intervention** (LR=0.001, Steps 100k-200k): cut learning rate by 90% to stabalize training
3. **Cosine Annealing** (Steps 200k-370k): Smooth convergence to best loss of 3.6432 at step 310,811

The chart also shows two other attemps, one with too high a learning rate (gradients explode) and a too low learning rate with a fast initial decent, but very slow improvement afterwards.

**Training Duration**: Approximately 5 hours for 370k steps on a single GPU (~1,200 steps/minute)

### Headroom Analysis

<img src="assets/gradient_sum.png" alt="Gradient Sum" width="600"/>

The monotonically increasing gradient magnitude (0.48→0.54) indicates headroom remains. This steady growth in gradient norms, even as validation loss improved, demonstrates the model was still finding meaningful parameter updates and exploring productive regions. With 170k steps of cosine annealing completed, the healthy gradient signal suggests extended training would likely yield further improvements. Perhaps into the 3.5-3.6 range.

### Language Acquisition

Monitoring the model's completions of "The quick brown fox jumps over the..." shows progress of linguistic development:

**Step 197k** (syntax, but still complete nonsense):
"...carpet and proceeds to train him for the skunks last business"

**Step 227k** (an action sequence):
"...window after crossing a downhill rock indoors a kick blob"

**Step 289k** (complex grammar, surealism):
"...ladder flipping across the roof and the fans throw their feet against the ground killing themselves"

**Step 273k** (brevity - model knows when to stop!):
"...wire"

The progression has a clear pattern: structure -> narrative patterns -> length control. The model mastered grammatical rules before semantic coherence. This is characteristic of pure autoregressive pre-training without grounding.

### Comparison with GPT-2 124M

I downloaded OpenAI's open sourced weights for GPT2 124M. Although the model architectures are very similar, there  For context, here are GPT-2 124M's completions of the same prompt:

"...curb. It's ready to go. I glance around for sinkholes..."
"...fence."
"...cat. drawing to me the all colourful bowl of splendor..."

While GPT-2 124M shows more grounded, everyday completions, my 33.6M parameter model exhibits similar grammatical competence despite being 4x smaller. The cleaner training data appears to have enabled efficient learning of syntactic patterns, though semantic grounding remains a challenge at this scale.

### Key Observations

- **Perplexity-Quality Gap**: Best validation loss (3.6432) corresponded to grammatically correct but semantically unusual generations
- **Emergent Conciseness**: Later training stages showed increasingly focused completions
- **Data Quality Impact**: The preprocessed corpus enabled stable training dynamics despite aggressive initial learning rates

Configuration: `config/gpt2-bert-corpus-gpu.yaml` | Single GPU | bfloat16 precision | torch.compile optimization

# Model Interpretability

## Attention Visualization

The [visualization notebook](notebooks/05_visualize_attention.ipynb) instruments the model and extracts attention layers during inference. We can plots these layers as a heatmap to visualize the attention relationship between tokens.

![Attention Patterns for the second to last layer](assets/attention_gpt2_774M_layer11.png)
*Attention patterns in the second-to-last layer. The model correctly attends to 'brown suitcase' when generating 'a larger', as shown by the highlighted rows for 'a' and 'larger'.*

## Out Of Sample Evaluation
```bash

python -m llm_e1e.eval --task simple-wikipedia training-corpus c4 --max-samples 1000
```

I compared model perplexity based on next token prediction against the training corpus, simple-wikipedia, and c4 (common crawl). 

```bash
[BERT-CORPUS]
-----------------
  perplexity: 66.779
  loss: 4.201
  samples: 1000
  
[C4]
----
  perplexity: 5850.555
  loss: 8.674
  samples: 1000
```
The model performs as expected on text similar to its training data, but performs poorly on out of corpus text including
simple-wikipedia and c4. Inspecting samples from open books and c4 show differences in:
- **punctuation**: the training corpus has all punctuation removed
- **capitalization**: the training corpus is all lower case while C4 is mixed case, allowing for identification of entities, sentence structure, and emphasis.
- **style**: narrative and conversational vs. informational and commercial 

Toronto Open Books:
```commandline
she said you may find this hard to believe but there was very little acting it was
horrible we became those people we were those people she said that today people would
probably call it method acting but added we didnt know what method acting was we just
called it getting on with it syms said that during the scene where the ambulance rolls
backwards down the hill narrowly avoiding her the actors assumed there would be a hawser
to stop the vehicle if anything went wrong but there was not the actress said she was
pretty sure mills quayle and andrews angrily upbraided director j lee thompson for
this risky approach she added he liked to push actors a bit
```

C4
```
Biomedics 1 Day Extra are daily replacement disposable contact lenses by CooperVision
Hydron. Buy one box of 90 lenses. Biomedics 1 Day Extra contacts give you all the
convenience of a daily disposable lens with no need for solutions, cases or cleaning
and are perfect for the occasional wear. These lenses have greater comfort handling with
superior ease of insertion and removal. Biomedic 1 Day Extra are also marketed under
various other brand names including Clear Choice 1-day, Ascend 1-day, easyvision CLARISION
SPHERE, Clearsight 1 Day and ProView Daily Disposable.
```


## Loading OpenAI's Pretrained Weights

To validate my implementation, I loaded OpenAI's pretrained GPT-2 124M weights into my model architecture. This required careful weight manipulation to account for minor difference between implementations.

- OpenAI GPT2 uses a **Fused QKV matrix** while my model uses separate Q, K, V matrices, which is a more direct interpretation of the design from Attention is All You need. I acknowledge that the fused matrix implementation may be more performance.
- Transpose all weights to convert from TensorFlow's format to PyTorch
- Added QKV biases to my model. (I originally omitted these, but added later bias terms for compatability)
- Tediously munge various layer names to align naming conventions. OpenAI used fairly terse (but standard) naming, but I choose to be a bit more descriptive. 

### Weight Transformation Process

The [weight loading notebook](notebooks/04_load_pretrained_weights.ipynb) handles the conversion:

```python
# Downloads model-124M from OpenAI's released checkpoints
download_gpt2_model(model_size='124M', models_dir='models')

...

# OpenAI uses shape [768, 2304] where 2304 = 768 * 3 (Q, K, V)
qkv = loaded_weights[f'model/h{i}/attn/c_attn/w']
q, k, v = qkv.split(n_embed, dim=1)

# Map to separate parameters in my model
state_dict[f'h.{i}.attn.q_proj.weight'] = q.T
state_dict[f'h.{i}.attn.k_proj.weight'] = k.T  
state_dict[f'h.{i}.attn.v_proj.weight'] = v.T
```

### Generate text using loaded OpenAI weights
generate_text(m_gpt2, enc, "The gold trophy would not fit in the brown suitcase because it was too",
              max_new_tokens=10, temperature=0.8)

Output: "big, so it went into a box"

It took several tries to get weights loaded and for the model produces coherent completions. But, this confirmed successful weight transfer despite the architectural differences. 


# Code Pointers

* **`llm_e2e/config.py`**: defines the `GPT2Config` dataclass, which manages configuration parameters for the GPT-2 model and training process. This grew overtime to handle all settings related to data (dataset path, name, block size), model architecture (vocab size, embedding dimensions, number of layers/heads), training hyperparameters (learning rate, batch size, epochs), and system settings (device, model compilation). The configuration can be loaded from and saved to YAML files. It also includes a utility to estimate the total number of parameters in the model based on the configuration.

* **`src/llm_e2e/dataset.py`**: focuses on data loading and preprocessing. It includes implementations for:
    * `ShakespeareDataloader` that loads 'karpathy/tiny_shakespeare' dataset fully into memory, tokenizes it using `tiktoken`, and prepares batches (x,y pairs) for training.
    * A `StreamingDatasetGenerator` class to stream data from Hugging Face datasets. This wrapper handles shuffling, tokenization, and batch creation and exposes a python iterator. 
    * Utilities for checking data quality, such as analyzing token frequencies and vocabulary coverage, and printing sample input/output pairs.

* **`src/llm_e2e/model.py`**: (deprecates notebooks/02_gpt2_model.ipynb) PyTorch implementation of the GPT-2 model architecture. Key components defined include:
    * `GPTModel`: The main model class, which combines token and positional embeddings, a series of transformer blocks, a final layer normalization, and an output linear layer (tied to token embeddings, like GPT2). 
    * `TransformerBlock`: Implements the core transformer block with multi-head self-attention and a position-wise feed-forward network, using pre-normalization.
    * `MultiHeadAttention`: The multi-head self-attention mechanism, including causal masking for autoregressive training.
    * `FeedForward`: The position-wise feed-forward network, using GELU activation.
    * `LayerNorm`: A standard layer normalization implementation.
    * `GELU`: The Gaussian Error Linear Unit activation function.
    The implementation references OpenAI's GPT-2 paper and Andrej Karpathy's nanoGPT. The model also includes `generate` method for text generation and following Andrej's example, can product both logits and loss in the forward pass if Ys are provided. Instrumentation hooks were added to enable visualization of attention vectors.

* **`src/llm_e2e/trainer.py`**: (deprecates notebooks/03_gpt2_training.ipynb) Executes training inclusive of logging and checkpointing. 
    * A training loop that iterates through epochs and batches, performs forward and backward passes, and updates model parameters.
    * Functions to estimate training and validation loss (`estimate_loss`, `evaluate_model`).
    * Generating sample text during training to observe model progress (`generate_text`).
    * Logging training progress, including running loss and evaluation metrics.
    * Saving the trained model's state dictionary and training state.
    * Logging to wandb and stdout

* **`train.py`**: main orchestration training script. Loads configuration, checkpointed model state, sets up the optimizer, and starts training.

* **`notebook/04_load_openai_gpt2_123M.ipynb`**: downloads and loads the model parameters released by open ai
    * T2T and the GPT2 model use a fused QKV parameter and split after the matrix multiply. This might have
      some performance benefits, but it's having separate parameters maps better to the AIAYN paper. But, I had to split
      the params manually during loading. Also worth noting is that QKV bias is used in the GPT2 model and I had to load
      these parameters. The rest is basically just key mapping.
    * Added some basic tests to ensure the model is coherent.

* **`tests/test_*.py`**: Unit and integration test for config, data loading, model training, and the model itself. 

---

# TODO List:

- [x] **Data Loading**: Get a very small dataset, imdb, and fineweb loading and set up for training / validation
    - loaded tiny shakespeare to test the model, ensuring it can be overfit to one batch
    - choose fineweb-edu as a high quality curated dataset
    - implemented streamed loading / shuffling and wrapped the stream in an iterator class for training 
- [x] **Architecture**: Basic GPT2 model structure, probably in pytorch (later jax + flax)
    - implemented GPT2Model top down (from model definition, to transformer, to causal attention)
    - using openai gpt2 and micro gpt2 for reference, implemented in PyTorch, adding specific comments
      connecting gpt2 TF implementation with my rewrite
    - added dropout to avoid overfitting and improve generalization
    - added generate method to watch the model learn while training
    - wrote basic model dict saving / restore
- [x] **Pre-Training**: Basic pre-training including eval logging, checkpointing, and logging for wandb 
    - write a basic training script from pytorch references
    - made changes required to move model/tensors to gpu, using tensor cores, bfloat16, compiling model
    - improved output stauts printing
- [x] **Evaluation**: Implement eval for next token completion (perplexity)
- [X] **Visualization**: Extract attention weights and visualize to identify patterns.
- [ ] **Finetuning**: train for sentiment classification and instruction following
- [X] **ML Ops**: Deploy wandb for productionalization and checkpointing for restarts


## References

[1] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). *OpenAI blog*, 1(8), 9.

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762).

[3] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805).

[4] Alammar, J. (2018). [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/). 

[5] Karpathy, A. (2022). [nanoGPT](https://github.com/karpathy/nanoGPT). GitHub repository.

[6] OpenAI. (2019). [GPT-2: 1.5B release](https://github.com/openai/gpt-2). GitHub repository.

[7] Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). [What does BERT look at? An analysis of BERT's attention](https://arxiv.org/abs/1906.04341). *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.

[8] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361). *arXiv preprint arXiv:2001.08361*.

[9] Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2021). [A mathematical framework for transformer circuits](https://transformer-circuits.pub/2021/framework/index.html). *Transformer Circuits Thread*.
