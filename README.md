# LLM end-to-end project TODO list

**Project Goal:** Build an end-to-end pipeline to pre-train, fine-tune, and evaluate a language model that starts with a basic GPT2 
implementation and introduces features over time.

---

# TODO List:

- [ ] **Data Loading**: Get a very small dataset, imdb, and fineweb loading and set up for training / validation
- [ ] **Architecture**: Basic GPT2 model structure, probably in pytorch (later jax + flax) 
- [ ] **Pre-Training**: Basic pre-training including logging and tensorboard
- [ ] **Finetuning**: train for sentiment classification and instruction following
- [ ] **Evaluation**: Identify straightforward eval strategies for next token completion and instruction handling 
