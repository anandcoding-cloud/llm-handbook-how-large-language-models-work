# References and Sources

**Created by Anand** ([LinkedIn](https://www.linkedin.com/in/anand-balagopal) · [GitHub](https://github.com/anandcoding-cloud)). Drafted with AI assistance (ChatGPT, Claude) and reviewed by the author. Version 1.0, October 2026. Errors and suggestions are welcome as GitHub issues.

**Licence:** the text is released under [CC BY 4.0](LICENSE) and the code (`Appendix-A-tinygpt.js`) under the [MIT licence](LICENSE-MIT-code.txt).

---

This section lists where the handbook's material comes from: the learning material it was built on, the research papers it draws on, and the documentation, model cards and websites that were consulted. It is organized by topic, and each entry says which chapters use it.

## How to read this list

- **Papers** are cited by first author, year, title and arXiv identifier (links go to the arXiv abstract page). Every identifier was confirmed to resolve to the stated paper, either through arXiv's own catalogue or by opening its abstract page.
- **Documentation, model cards and websites** were opened and read in the course of writing, and are listed with their addresses. Documentation changes frequently, so details such as model names, defaults and prices may have moved by the time you read this. The date of the last check was **October 2026**.
- Entries marked **(search result only)** were seen as a headline or summary in a web search but not opened in full, so they are weaker evidence than the rest.
- A few explanations in the handbook come from general background knowledge and are not tied to one source. They are noted at the end, in "Notes on verification".

---

## 1. The learning backbone and the source conversation

The handbook grew out of a guided study session built around one lecture series. Two items sit underneath everything:

- **The YouTube lecture series *Building LLMs from scratch*, by the channel Vizuara**, which provided the order of topics for Chapters 2 to 11 (tokenization, embeddings, attention, the Transformer block, training). Many thanks to the Vizuara team for making it freely available. If you want to follow along with the original lectures, start here.
  - **First lecture:** *Lecture 1: Building LLMs from scratch: Series introduction*: <https://youtu.be/Xpr8D6LeAtw>
  - **Playlist:** *Building LLMs from scratch*: <https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu>
  - The handbook does not reproduce the lectures. It uses their sequence as a skeleton, and expands or reorders where more explanation was useful.
- **The source conversation:** a ChatGPT tutoring conversation titled *"LLM building overview"*, which walked through the lecture topics and went on to cover fine-tuning, quantization, serving, MoE, reasoning and multimodal models. The explanations, worked examples and the questions asked along the way come from it, reorganized and corrected where needed. Chapters 1 to 15 and 18 to 20 draw on it heavily, Chapters 22 and 23 only lightly (the conversation covered those topics briefly), and Chapters 16, 17, 21 and 24 were written mainly from the outside sources listed below. Address: <https://chatgpt.com/share/6a83f0ae-b5d8-83e8-aba9-4d63e88465db>
  - The web citations that appeared inside that conversation were not preserved in the exported copy, so they cannot be listed here. Where a claim needed support, it was re-checked against the sources below.
- **No other YouTube playlists or videos** were searched or used.

---

## 2. Foundations: the Transformer, tokenization, normalization, optimization (Chapters 2 to 11)

- Vaswani et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Chapters 4, 5, 7, 15.
- Sennrich et al. (2015). *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). Byte-pair encoding. Chapter 2.
- Ba et al. (2016). *Layer Normalization.* [arXiv:1607.06450](https://arxiv.org/abs/1607.06450). Chapter 7.
- He et al. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Residual connections. Chapter 7.
- Hendrycks et al. (2016). *Gaussian Error Linear Units (GELUs).* [arXiv:1606.08415](https://arxiv.org/abs/1606.08415). Chapter 6.
- Kingma et al. (2014). *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980). Chapter 11.
- Loshchilov et al. (2017). *Decoupled Weight Decay Regularization.* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). AdamW. Chapter 11 and Appendix A.
- Brown et al. (2020). *Language Models are Few-Shot Learners.* [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). GPT-3. Chapters 5, 7.

## 3. The Transformer family (Chapter 15)

- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- Raffel et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* [arXiv:1910.10683](https://arxiv.org/abs/1910.10683). T5.
- Lewis et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692).
- He et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654).
- Reimers et al. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084). Embedding models; also Chapter 24 (RAG).
- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* [arXiv:2212.04356](https://arxiv.org/abs/2212.04356). Whisper, an encoder–decoder model. Chapters 15, 19.

## 4. Inference: attention cost, the KV cache and long context (Chapters 12 and 17)

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Chapter 17.
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). Chapter 17.
- Shah et al. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* [arXiv:2407.08608](https://arxiv.org/abs/2407.08608). Chapter 17.
- Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172). Chapter 24 (context placement).

## 5. Fine-tuning and alignment (Chapter 13)

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). InstructGPT; the RLHF pipeline.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347). PPO.
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). DPO.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). Also Chapter 14.

## 6. Quantization (Chapter 14)

- Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339). Outliers in activations.
- Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
- Lin et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
- The GGUF file format and the llama.cpp quantization formats are covered in the documentation listed in Section 12.

## 7. Modern building blocks and attention variants (Chapters 16 and 17)

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). RoPE.
- Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.* [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
- Zhang et al. (2019). *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467). RMSNorm.
- Shazeer (2020). *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202). SwiGLU.
- Henry et al. (2020). *Query-Key Normalization for Transformers.* [arXiv:2010.04245](https://arxiv.org/abs/2010.04245). QK-Norm.
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). Multi-query attention.
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- Chowdhery et al. (2022). *PaLM: Scaling Language Modeling with Pathways.* [arXiv:2204.02311](https://arxiv.org/abs/2204.02311). A model using multi-query attention.
- DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). Multi-head latent attention (MLA).
- Jiang et al. (2023). *Mistral 7B.* [arXiv:2310.06825](https://arxiv.org/abs/2310.06825). Sliding-window attention.
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer.* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).
- Gu et al. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
- Yang et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464). Gated DeltaNet, used in the hybrid Qwen models.
- DeepSeek-AI (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* [arXiv:2512.02556](https://arxiv.org/abs/2512.02556). DeepSeek Sparse Attention. Also Chapter 21.

## 8. Mixture of Experts (Chapter 18)

- Lepikhin et al. (2020). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* [arXiv:2006.16668](https://arxiv.org/abs/2006.16668).
- Fedus et al. (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* [arXiv:2101.03961](https://arxiv.org/abs/2101.03961). Top-1 routing and the load-balancing loss.
- Jiang et al. (2024). *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- Dai et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* [arXiv:2401.06066](https://arxiv.org/abs/2401.06066). Fine-grained and shared experts.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). Also Chapters 17 and 21.

## 9. Multimodal models (Chapter 19)

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). Vision Transformer.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). CLIP.
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* [arXiv:2204.14198](https://arxiv.org/abs/2204.14198).
- Liu et al. (2023). *Visual Instruction Tuning.* [arXiv:2304.08485](https://arxiv.org/abs/2304.08485). LLaVA.
- Radford et al. (2022). Whisper, listed in Section 3, for audio encoders.

## 10. Reasoning models (Chapter 20)

- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* [arXiv:2201.11903](https://arxiv.org/abs/2201.11903).
- Kojima et al. (2022). *Large Language Models are Zero-Shot Reasoners.* [arXiv:2205.11916](https://arxiv.org/abs/2205.11916).
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
- Yao et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* [arXiv:2305.10601](https://arxiv.org/abs/2305.10601).
- Lightman et al. (2023). *Let's Verify Step by Step.* [arXiv:2305.20050](https://arxiv.org/abs/2305.20050). Outcome vs process supervision.
- Snell et al. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* [arXiv:2408.03314](https://arxiv.org/abs/2408.03314).
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). GRPO.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
- Muennighoff et al. (2025). *s1: Simple test-time scaling.* [arXiv:2501.19393](https://arxiv.org/abs/2501.19393). Budget forcing.
- Aggarwal et al. (2025). *L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning.* [arXiv:2503.04697](https://arxiv.org/abs/2503.04697). Length-controlled policy optimization.
- Sun et al. (2025). *BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens.* [arXiv:2508.17196](https://arxiv.org/abs/2508.17196). Seen in search results; abstract-level.
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629). Also Chapter 24.

## 11. Serving, speed and scale (Chapters 22 and 23)

- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). vLLM. The 20.4% to 38.2%, 800 KB-per-token and 2 to 4× figures come from this paper.
- Leviathan et al. (2022). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- Chen et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).
- Cai et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).
- Li et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* [arXiv:2401.15077](https://arxiv.org/abs/2401.15077).
- Shoeybi et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053). Tensor parallelism.
- Huang et al. (2018). *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* [arXiv:1811.06965](https://arxiv.org/abs/1811.06965). Pipeline parallelism.

## 12. Documentation, model cards and websites

**Serving software and file formats (Chapter 22)**

- GGUF file-format specification (ggml project): <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- llama.cpp server documentation (parallel slots, continuous batching, mmap, offload): <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- vLLM design documents: PagedAttention <https://docs.vllm.ai/en/latest/design/paged_attention.html> and automatic prefix caching <https://docs.vllm.ai/en/latest/design/prefix_caching.html>. The first is marked by its authors as a historical description of the original paper.

**Hardware (Chapter 23)**

- NVIDIA H100 product page (memory, bandwidth, NVLink, PCIe figures): <https://www.nvidia.com/en-us/data-center/h100/>
- Google, *Our eighth generation TPUs: two chips for the agentic era* (TPU 8t and TPU 8i): <https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/>
- StorageReview, *Google Announces TPU 8t Sunfish and TPU 8i Zebrafish* (memory and bandwidth details for the 8i): <https://www.storagereview.com/news/google-announces-tpu-8t-sunfish-and-tpu-8i-zebrafish>
- The Decoder, *OpenAI and Broadcom unveil "Jalapeño," a custom chip built for LLM inference* (June 2026): <https://the-decoder.com/openai-and-broadcom-unveil-jalapeno-a-custom-chip-built-for-llm-inference/>. Further coverage by TechPowerUp and others appeared in search results only.
- Both the TPU 8i and Jalapeño material are company announcements, and the Jalapeño performance claims were not independently verified.

**Reasoning effort, thinking budgets and tool use (Chapters 20 and 24)**

- Anthropic documentation: *Extended thinking* <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>; *Effort* <https://platform.claude.com/docs/en/build-with-claude/effort>; *Steering thinking* <https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost>; *Thinking* <https://platform.claude.com/docs/en/build-with-claude/thinking>; *Task budgets* <https://platform.claude.com/docs/en/build-with-claude/task-budgets>.
- OpenAI documentation: *Reasoning models* <https://developers.openai.com/api/docs/guides/reasoning>.
- Model Context Protocol, *What is MCP?*: <https://modelcontextprotocol.io/docs/getting-started/intro>

**Model cards and technical reports (Chapters 20 and 21)**

- Qwen Team (2025). *Qwen3 Technical Report.* [arXiv:2505.09388](https://arxiv.org/abs/2505.09388). Thinking mode, thinking budget, and the training stages.
- Qwen3-8B model card (thinking switch, sampling settings): <https://huggingface.co/Qwen/Qwen3-8B>
- Qwen3-Next-80B-A3B model card (hybrid Gated DeltaNet and attention layout): <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>
- Qwen3.5 and 3.6 families: described in Chapter 21 from their model cards on Hugging Face and secondary coverage in search results. These were **not individually re-verified**, so treat the details there as a snapshot.
- DeepSeek-V3.2 model card: <https://huggingface.co/deepseek-ai/DeepSeek-V3.2>
- DeepSeek-V4-Pro model card (hybrid compressed attention, mHC, Muon): <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- OpenAI gpt-oss-120b model card (reasoning level set in the system prompt): <https://huggingface.co/openai/gpt-oss-120b>
- Grattafiori et al. (2024). *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783). Model sizes used in several examples.
- Gemma Team (2025). *Gemma 3 Technical Report.* [arXiv:2503.19786](https://arxiv.org/abs/2503.19786).
- Kimi Team (2025). *Kimi K2: Open Agentic Intelligence.* [arXiv:2507.20534](https://arxiv.org/abs/2507.20534).

## 13. Production systems (Chapter 24)

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Liu et al. (2023), *Lost in the Middle*, and Yao et al. (2022), *ReAct*: listed in Sections 4 and 10.
- Model Context Protocol: listed in Section 12.

---

## Notes on verification

- **Papers.** Every arXiv identifier above was confirmed to match the stated paper. The abstracts were read for the papers from which the handbook quotes specific results: Mixtral (47B total, 13B active), Switch Transformer (speed-up), DeepSeekMoE, self-consistency, Tree of Thoughts, process supervision, test-time compute, chain of thought, s1, L1, DeepSeek-R1, DeepSeekMath, vLLM and speculative decoding (the 2 to 3× result), CLIP, ViT, Flamingo and LLaVA, and the Qwen3 report. For the remaining papers (for example RoPE, RMSNorm, SwiGLU, GQA, FlashAttention, LoRA, DPO, PPO, GPTQ, AWQ), the handbook relies on well-established descriptions of the methods and does not quote numbers from them.
- **Documentation and model cards.** Statements about reasoning-effort controls, thinking budgets, countdown markers, token billing and token limits come from the provider documentation listed in Section 12, as read in October 2026. Provider behaviour changes quickly.
- **Illustrative numbers.** Many numbers in the handbook are deliberately made up to show a principle: the stop-probability tables in Chapter 20, the cost and price examples in Chapters 22 and 24, the 2-D embedding vectors in Chapter 24, and the "quality" scores in Chapters 14 and 20. They are always labelled as illustrative in the text.
- **Appendix A** is different: every number in it was computed by a program (`Appendix-A-tinygpt.js`, included next to it), and its gradients were checked against numerical differentiation.
- **Hardware figures (Chapter 23)** such as H100 bandwidth, NVLink and PCIe speeds are as quoted by NVIDIA, and the NVLink and PCIe figures there are total both directions. The TPU 8i and Jalapeño details are company announcements.
- **Chapters 16, 17 and 21** (the modern-models chapters) were written earlier in the project and checked against the official papers and model cards for the models discussed. The papers listed above are the primary sources. Page-level addresses for individual claims in those chapters were not recorded, so check a specific figure against the cited paper or model card before relying on it.
- **Not used.** No book, course or website beyond those listed was consulted, and no other YouTube playlist or video was searched or sourced.
