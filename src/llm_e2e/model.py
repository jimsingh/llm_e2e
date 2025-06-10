"""
Moved from notebooks/02_gpt_model.py

OpenAI's GPT2 paper, github repo, and tf code were used as a reference for this design.
My goal was to match the GPT2 design closely enough that I could load the OpenAI 
opensource weights and run inference. The comments in the model code below are
directly pulled from the gpt2 source code.

The OpenAI paper and github don't provide information on how they trained
this model (datasets, training parameters, etc ...). I pull information from pytorch 
documentation, Andrej Karpathy's implementation, and a fair bit of trial and error
to pull together training data and training parameters.

The training datasets include mini-shakespear (good for testing overfit), 
fineweb-edu (noisy), and wikipedia simple. BERT was trained on a (refined) Wikipedia EN
and OpenBooks. 

Separately, I added instrumentation to inspect attention weights in the forward pass.

References:
* paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* repo: https://github.com/openai/gpt-2
* kaparthy nanoGPT: https://github.com/karpathy/nanoGPT
* https://huggingface.co/blog/bert-101
* https://huggingface.co/datasets/wikimedia/wikipedia

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTModel(nn.Module):
    """
    The high-level tensorflow model structure released by OpenAI translated to pytorch.
    
    src: https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L147
    
    def model(hparams, X, past=None, scope='model', reuse=False):
        results = {}
        batch, sequence = shape_list(X)
                
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results 
    """
    def __init__(self, cfg):
        super().__init__()
        
        # positional embeddings - each position in the input gets a learned positional embedding to capture relationships
        # between words. Worth noting, AIAYN used fixed embeddings while GPT and BERT uses learned embeddings.
        #
        # wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd], initializer=tf.random_normal_initializer(stddev=0.01))
        self.position_embeddings = nn.Embedding(num_embeddings=cfg.context_length, embedding_dim=cfg.emb_dim)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.01)

        # token embeddings - map token ids to learned token embeddings
        #
        # wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        self.token_embedding = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.emb_dim)
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        self.emb_dropout = nn.Dropout(cfg.dropout_rate)
        
        # positional and token embeddings are added together to represent both the word and where it
        # is in the input. this is the input to the transformer blocks.

        # transformer model is a stack of transformer blocks. the hparam cfg.n_layers tells us how many
        #
        #    presents = []
        #    for layer, past in enumerate(pasts):
        #        h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
        #        presents.append(present)
        #    results['present'] = tf.stack(presents, axis=1)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # normalize (stabilize) to unit variance and learn gain and bias
        # 
        #    norm from h = norm(h, 'ln_f')
        self.final_norm = LayerNorm(cfg.emb_dim)
    
        # project hidden states (h) to logits tying weights to token_embeddings (so we end up with 1 set of
        # embeddings for tokens used in both input and output.
        # Corresponds to TF: `logits = tf.matmul(h_flat, wte, transpose_b=True)`

        # F.linear(input, weight) applies a linear transformation to the last dimension of the hidden layer
        # h: (B, T, C), self.token_embedding.weight: (vocab_size, C) -> (B, T, vocab_size)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        
        # Tie the weights of the output projection layer with the token embedding layer
        self.out_head.weight = self.token_embedding.weight

        # tiktoken gpt2 encoding, this is the token id of '<|endoftext|>'
        self.eos_token_id = cfg.eos_token_id


    def forward(self, in_idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape # token indices (B, T)

        # get the token embeddings for each token index (B, T) -> (B, T, C)
        tok_embeds = self.token_embedding(in_idx)

        # get position embeddings for each sequence index (T)
        pos_indices = torch.arange(seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.position_embeddings(pos_indices) # (T) -> (T, C)

        # Combines token and position embeddings for input to transformer blocks
        # 
        #    h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))
        x = tok_embeds + pos_embeds # (B, T, C) + (T, C) -> (B, T, C) - broadcasting works right to left
        s = self.emb_dropout(x)
        
        # replaces the for layer, past in enumerate(pasts) loop
        # n.Sequential replaces the `for layer, past in enumerate(pasts)` loop
        h = self.transformer_blocks(x)

        # h = norm(h, 'ln_f')
        h = self.final_norm(h)
        
        # hidden state to logits, linear C -> cfg.vocab_size
        logits = self.out_head(h) # (B, T, C) -> (B, T, cfg.vocab_size)

        # compute the loss if given targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)
        
        # todo: optimize logits calculation if we're only performing inference
        
        return logits, loss

    
    @torch.no_grad()
    def generate(self, idx, max_tokens, temperature=1.0, top_k=None):
        self.eval()
        
        context_length = self.position_embeddings.weight.shape[0]

        for _ in range(max_tokens):
            # crop to the context length
            idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
            
            # forward the model to get the logits for the index in the sequence
            # TODO: KV caching so that we don't recompute every 
            logits, _ = self(idx_cond)
            
            # scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # check for end of sequence token and stop generation.
            if idx_next.item() == self.eos_token_id:
                break
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
    
            
    def save_parameters(self, file_path: str):
        """
        Saves model parameters (state_dict) to a specified file path.

        Args:
            file_path (str): The path where the state_dict will be saved.
        """
        current_state_dict = self.state_dict()
        
        # Save the state dictionary to the specified file
        torch.save(current_state_dict, file_path)
        
        print(f"Model parameters saved to {file_path}")
        
    def load_parameters(self, file_path: str, mod_prefix='_orig_mod.', map_location=None):
        """
        Loads model parameters (state_dict) from a specified file path.

        Args:
            file_path (str): The path to the file containing the state_dict.
            remove_prefix (str): prefix to remove (often created by model.compile)
        """

        state_dict = torch.load(file_path, map_location=map_location)
        has_prefix = any(k.startswith(mod_prefix) for k in state_dict.keys())

        if has_prefix:
            state_dict = {
                k.removeprefix(mod_prefix): v
                for k, v in state_dict.items()
            }
        self.load_state_dict(state_dict)
        
        print(f"Model parameters loaded from {file_path}")

            
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1), persistent=False)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # the tensor is now shaped like num_heads and for each head we have (num_tokens, head_dim) 
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        
        # transpose flips (num_tokens, head_dim) to (head_dim, num_tokens) 
        # and the dot product calculates attention for each head
        # shape of attn_scores is (b, num_heads, head_dim, head_dim)
        attn_scores = queries @ keys.transpose(2, 3) 

        # mask future positions to be -inf
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # reshape values to be 'per head' as above
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.transpose(1, 2)

        # scale values by attention and transpose so that the per head
        # information is contiguous for each token
        context_vec = (attn_weights @ values).transpose(1, 2)

        # merge heads to create d_out
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # cross head attention: out_proj is the linear combination across heads
        context_vec = self.out_proj(context_vec)

        return context_vec, attn_weights
        
class TransformerBlock(nn.Module):
    """ 
    Implementation of Transformer Blocks without KV caching (not critical for training)

    src: https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L123C1-L130C26
    
    def block(x, scope, *, past, hparams):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
            x = x + a
            m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
            x = x + m
            return x, present
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim, d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout_rate,
            qkv_bias=cfg.qkv_bias)
        
        self.ff = FeedForward(cfg) # ff = mlp
        self.norm1 = LayerNorm(cfg.emb_dim) # norm(x, 'ln_1')
        self.norm2 = LayerNorm(cfg.emb_dim) # norm(x, 'ln_2')
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x):
        """
        Implements the transformer pre-norm forward pass:
    
            X_in -> Norm_1 -> Attention (A)     -> Shortcut (X_in + A)   -> X_a
            X_a  -> Norm_2 -> FeedForward (MLP) -> Shortcut (X_a + MLP) -> X_out

        """
        # x_in -> Norm1 -> Attn -> Residual -> x_a
        x_n = self.norm1(x) 
        a, _ = self.att(x_n) # ignore attn weights
        a = self.dropout(a)
        x_a = x + a            # (B, T, C) + (B, T, C) (add residual) -> (B, T, C)
        
        # feedforward
        x_na = self.norm2(x_a)
        x_mlp = self.ff(x_na)
        x_mlp = self.dropout(x_mlp)
        x_out = x_a + x_mlp # add residual (shortcut)
        return x_out
        
class LayerNorm(nn.Module):
    """
    Implements layer normalization
    
    src: https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L28
    
    def norm(x, scope, *, axis=-1, epsilon=1e-5):
        # normalize to mean = 0, std = 1, then do a diagonal affine transform.
        with tf.variable_scope(scope):
            n_state = x.shape[-1].value
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
            u = tf.reduce_mean(x, axis=axis, keepdims=True)
            s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
            x = (x - u) * tf.rsqrt(s + epsilon)
            x = x*g + b
            return x
    """
    
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-5
        self.gain = nn.Parameter(torch.ones(dim)) # scale
        self.bias = nn.Parameter(torch.zeros(dim)) # shift

    def forward(self, x):
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # n is large enough that biased vs unbiased shouldn't matter, but this is what GPT2 does
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # from the gpt2 source: "then do a diagonal affine transform."
        #
        # english translation:
        #     affine transfrom: y = Wx + b, "diagonal" means W is diagonal (other elements are 0)
        #     this effectively means that the transfrom is *per feature* and the gain + shift
        #     is learned for each feature without any cross-feature interactions.
        return self.gain * norm_x + self.bias


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward from AIAYN. The module expands input features, applies activation,
    and then projects them back to the original dimensions. The idea is to learn more expressive
    relationships between features.
    
    note: GPT2 replaces ReLU with GELU.

    from: https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L115
    
    def mlp(x, scope, n_state, *, hparams):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            h = gelu(conv1d(x, 'c_fc', n_state))
            h2 = conv1d(h, 'c_proj', nx)
            return h2

    def conv1d(x, scope, nf, *, w_init_stdev=0.02):
        with tf.variable_scope(scope):
            *start, nx = shape_list(x) # nx is the input feature dimension (in_channels)
            w = tf.get_variable('w', [1, nx, nf], # Kernel shape: [filter_width, in_channels, out_channels]
                                initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
            c = tf.nn.conv1d(x, w, stride=1, padding='VALID') + b # Uses tf.nn.conv1d
            return c
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        expansion_factor = 4

        # this was ... confusing. AIAYN uses a conv1d with a kernel size of 1 - meaning the convolution
        # operates on a single feature at a time. This is the same thing as nn.Linear which operates on the last
        # dimension of the input (in this case the embedding features).
        self.expansion = nn.Linear(cfg.emb_dim, expansion_factor * cfg.emb_dim) # conv1d expands (B, T, C) -> (B, T, 4C) 
        self.gelu = nn.GELU() 
        self.projection = nn.Linear(expansion_factor * cfg.emb_dim, cfg.emb_dim, bias=cfg.mlp_bias) # conv1d projects back to (B, T, C)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        
    def forward(self, x):
        # Asserts to make sure I'm thinking about this correctly
        # x initial shape: (B, T, C) where C is cfg.emb_dim
        assert x.ndim == 3, f"Input tensor expected to be 3D (B, T, C), got {x.ndim}."
        assert x.shape[-1] == self.expansion.in_features, \
            f"Input feature dimension mismatch. Expected {self.expansion.in_features}, but got {x.shape[-1]}."

        # The input and output dimensions should match, store them for the final assertion
        B, T, C = x.shape

        x = self.expansion(x)   # (B, T, C) -> (B, T, 4*C)
        x = self.gelu(x)        # (B, T, 4*C) -> (B, T, 4*C)
        x = self.dropout(x)
        x = self.projection(x)  # (B, T, 4*C) -> (B, T, C)

        assert x.shape == (B, T, C), f"Output shape mismatch. Expected {(B, T, C)}, but got {x.shape}."
        return x

class GELU(nn.Module):
    """
      GPT2 replace ReLU with Gaussian Error Linear Unit (GELU) as a smoother activation function.
      GELU based on math.erf (CDF of the standard normal distribution). For GELU, the integration
      of e^{-t^2} is approximated by a manually fit function of tanh. I believe GELU was used by BERT.
     
         gelu(x):
           return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
     
      ref: https://arxiv.org/pdf/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        )))
