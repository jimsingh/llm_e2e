# LLM end-to-end project 

**Project Goal:** Build an end-to-end pipeline to pre-train, fine-tune, and evaluate a language model that starts with a basic GPT2
implementation and introduces features over time.

**Status**: using fineweb-edu, trained a 124M parameter model for ~10 hours on a RTX 6000 Ada 48 GB, resulting in a loss 3.7 (perplexity of 40 of ~50000). Here is what that looks like:

```python
gen_f = lambda m: generate_text(m, enc, "tomorrow is")
gen_f(m)

'tomorrow is twice extreme times six dual in 7 price for e decrypted down into the ssh. asked you visualize'
```

... whatever you say GPT2.

## Prototype code in Jupyter Notebooks:

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
    [ ] TODO: implement checkpointing and logging for reporting
- [ ] **Finetuning**: train for sentiment classification and instruction following
- [ ] **Evaluation**: Implement eval strategies for next token completion and instruction handling
- [ ] **ML Ops**: Deploy wandb for productionalization

# Future Scope 
- Sparse Attention, Flash Attention
- LR Warmup / Cosine Decay  
- Distriuted Data Parallel / Full Sharded Data Parallel
- Optimizing training for TPU trillium (v6e) - (but VRAM is < RTX 6000 Ada)
