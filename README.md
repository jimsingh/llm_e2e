# LLM End-to-End: An Experimentation and Learning Framework for Decoder-Style Language Models

# Abstract

Large language models (LLMs) have become increasingly accessible and put into production across diverse applications. While the adaptability of modern pretrained models is remarkable, it is crucial to understand and debug the fundamentals of how these models work to apply them effectively. This project is a complete implementation and instrumentation framework for decoder-style language models, with a focus on the well-understood GPT-2 architecture [1]. The key components are modularized to enable systematic exploration of model behavior under different architectural and training hyperparameters. Through the extensibility of each component, the design serves as both an educational resource and a platform for conducting experiments on autoregressive language modeling. As a demonstration, I pre-train my own GPT-2 model from scratch, extract and visualize attention weights to highlight how the model interprets patterns of text, and show how the model can be aligned with human preferences.

## Introduction

Understanding the inner workings of large language models is critical as these systems play a more prominent role in a wide range of applications, many going well beyond simple natural language processing and token prediction. While pre-trained models are readily available, the ability to build, train, and analyze these models from first principles provides invaluable insights into their behavior, limitations, and optimization.

This project has five key objectives:

1. **Architecture Replication**: Reproduce the GPT-2 model architecture [1] in PyTorch with clear, documented code that connects theoretical concepts to practical implementation. Verify the implementation by loading and running OpenAI's pre-trained weights.

2. **Training Framework Development**: Build an end-to-end training pipeline that including efficient data loaders optimized for next-token prediction, streaming dataset support for large-scale training, integration with [Weights & Biases](https://wandb.ai/) for ML Ops and experiment tracking, and model state checkpointing for restartability.

3. **Representation Analysis**: Visualize learned representations, with particular emphasis on attention patterns [4], to provide insights into how the model processes and relates tokens within its context window.

4. **Fine-tuning and Alignment**: Apply supervised fine-tuning (SFT) and alignment techniques to adapt the model to human preferences, demonstrating how base language models can be shaped for specific tasks and behaviors.

5. **Performance**: Implement the highest ROI strategies to significantly improve model training and inferences performance including word alignment, quantization, KV caching, and quantization.

## Text Generation

```python
gen_f = lambda m: generate_text(m, enc, "The gold trophy would not fit in the brown suitcase because it was too large. I needed a larger"
gen_f(m)

'The gold trophy would not fit in the brown suitcase because it was too large. I needed a larger suitcase.'
```

## Attention Visualization

We'll look at the second to last layer of the model and visualize the attention heads. We'll clearly see that 'a larger' is strong attention to 'brown suitcase', indicating that the model understands the relationship. The 2 vertical blocks corresponding to the rows for 'a' and 'larger'.

![Attention Patterns for the second to last layer](assets/attention_gpt2_774M_layer11.png)

## Code pointers

* **`llm_e2e/config.py`**: defines the `GPT2Config` dataclass, which manages configuration parameters for the GPT-2 model and training process. This grew overtime to handle all settings related to data (dataset path, name, block size), model architecture (vocab size, embedding dimensions, number of layers/heads), training hyperparameters (learning rate, batch size, epochs), and system settings (device, model compilation). The configuration can be loaded from and saved to YAML files. It also includes a utility to estimate the total number of parameters in the model based on the configuration.

* **`src/llm_e2e/dataset.py`**: focuses on data loading and preprocessing. It includes implementations for:
    * `ShakespeareDataloader` that loads 'karpathy/tiny_shakespeare' dataset fully into memory, tokenizes it using `tiktoken`, and prepares batches (x,y pairs) for training.
    * A `StreamingDatasetGenerator` class to stream data from Hugging Face datasets. This wrapper handles shuffling, tokenization, and batch creation and exposes a python iterator. 
    * Utilities for checking data quality, such as analyzing token frequencies and vocabulary coverage, and printing sample input/output pairs.

* **`notebooks/02_gpt2_model.ipynb`**: PyTorch implementation of the GPT-2 model architecture. Key components defined include:
    * `GPTModel`: The main model class, which combines token and positional embeddings, a series of transformer blocks, a final layer normalization, and an output linear layer (tied to token embeddings, like GPT2). 
    * `TransformerBlock`: Implements the core transformer block with multi-head self-attention and a position-wise feed-forward network, using pre-normalization.
    * `MultiHeadAttention`: The multi-head self-attention mechanism, including causal masking for autoregressive training.
    * `FeedForward`: The position-wise feed-forward network, using GELU activation.
    * `LayerNorm`: A standard layer normalization implementation.
    * `GELU`: The Gaussian Error Linear Unit activation function.
    The implementation references OpenAI's GPT-2 paper and Andrej Karpathy's nanoGPT. The model also includes `generate` method for text generation and following Andrej's example, can product both logits and loss in the forward pass if Ys are provided.

* **`notebooks/03_gpt2_training.ipynb`**: Orchestrates the training process, but still pretty rough. It handles: 
    * Initializing the model, optimizer (AdamW), and data loaders.
    * A training loop that iterates through epochs and batches, performs forward and backward passes, and updates model parameters.
    * Functions to estimate training and validation loss (`estimate_loss`, `evaluate_model`).
    * Generating sample text during training to observe model progress (`generate_text`).
    * Logging training progress, including running loss and evaluation metrics.
    * GPU memory statistics logging if CUDA is used.
    * Saving the trained model's state dictionary and the full model.
    * TODO: logging for wandb

* **`notebook/04_load_openai_gpt2_123M.ipynb`**: downloads and loads the model parameters released by open ai
    * T2T and the GPT2 model use a fused QKV parameter and split after the matrix multiply. This might have
      some performance benefits, but it's having separate parameters maps better to the AIAYN paper. But, I had to split
      the params manually during loading. Also worth noting is that QKV bias is used in the GPT2 model and I had to load
      these parameters. The rest is basically just key mapping.
    * Added some basic tests to ensure the model is coherent.

* **`notebooks/99_tests.ipynb`**: This notebook contains some integration tests for the project to ensure things stay sane.
        * Sets up a test configuration (`GPT2Config`) with smaller parameters for quick testing.
        * Initializes the `ShakespeareDataloader` with the test configuration and tokenizer.
        * Initializes the `GPTModel` with the test configuration.
        * Fetches a batch of data from the dataloader.
        * Performs a forward pass of the batch through the model.
        * Asserts the shapes, device, and dtypes of the output logits.

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
- [ ] **Evaluation**: Implement eval strategies for next token completion and instruction handling
- [ ] **Visualization**: Extract attention weights and visualize to identify patterns.
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
