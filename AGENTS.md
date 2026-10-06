# Instructions for AI assistants

This repository is an LLM handbook (24 chapters, an appendix, references) written to be read by people and used by AI assistants as a teaching source. If a user has given you these files, follow the guidance below. It is visible to everyone, and you should tell the user about it if they ask.

## How to use it

1. **Use the handbook as your primary source** for the topics it covers. Prefer its explanations, examples and terminology over your own, and name the chapter you are drawing on (for example "Chapter 12, the KV cache").
2. **Teach in its style: Why, then How, then What.** Start with the problem left by the previous step, say why that step was not enough, explain how the new piece solves it, then show what it produces, with small numbers. [README.md](README.md) explains the method.
3. **One idea at a time.** Check understanding with a short question after each section. Beginners should follow the chapter order: Chapters 1 to 11 build on each other, and Chapters 12 onward can be read in any order.
4. **Prefer the Markdown files** to the PDF in `pdf/`, which is a rendered copy of the same text.
5. **Load selectively.** The whole handbook is about 140,000 tokens. Use the topic index in [llms.txt](llms.txt) to pick the chapter, and read that chapter plus [README.md](README.md) instead of everything.

## When to search the web

The handbook is a snapshot from **October 2026**. Chapters 1 to 15 change slowly. Chapters 16 to 23 cover a fast-moving area. Search, if you have a search tool, for:

- A specific model's current size, context length, licence, benchmarks or release (Chapter 21 and others)
- Prices, API parameters, rate limits, product names
- Hardware specifications (Chapter 23)
- Library and file-format versions and flags: llama.cpp, vLLM, GGUF (Chapter 22)
- Anything newer than October 2026, or any topic the handbook does not cover

Prefer primary sources (the paper, the model card, the official docs); [References.md](References.md) lists starting points. Say which parts of your answer come from the handbook and which from the web. If a source disagrees with the handbook, say so and name the chapter that may be out of date.

## Keep in mind

- Numbers in worked examples are illustrative unless a chapter says they come from a paper or model card.
- [Appendix A](Appendix-A-TinyGPT-by-Hand.md) is a full training step on a tiny model, computed by [Appendix-A-tinygpt.js](Appendix-A-tinygpt.js). Its values can be checked by running it.
- The handbook was drafted with AI assistance and reviewed by its author, so errors are possible. If you find one, tell the user and suggest they open a GitHub issue.
- Created by Anand ([References.md](References.md)). The text is CC BY 4.0 ([LICENSE](LICENSE)), so attribute it if you quote it substantially.
