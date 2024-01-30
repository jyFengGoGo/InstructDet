In InstructDet, we leverage vicuna-13b-v1.3 as the language model to generate instructions based on code from https://github.com/lm-sys/FastChat.

# FastChat
| [**Demo**](https://chat.lmsys.org/) |

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:
- The weights, training code, and evaluation code for state-of-the-art models (e.g., Vicuna, FastChat-T5).
- A distributed multi-model serving system with web UI and OpenAI-compatible RESTful APIs.

For installation methods and model weights for FastChat, please refer to the initial readme of FastChat at ./README_ini.md.

## Main function
We mainly edit the gen_instruct() function in [fastchat/serve/inference.py](fastchat/serve/inference.py) to support the three tasks referring to LLM.
- Single-Modality Instruct Generation
- Multi-object Instruct Generation
- Instruct Grouping

## Scripts
- Instruct generation: test_instruct.sh
- Multi-object expression generation: test_multi.sh
- Instruct grouping: test_level.sh
