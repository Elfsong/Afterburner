# 🚀 Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2402.07844-b31b1b.svg)](https://arxiv.org/abs/2505.23387)
[![Monolith](https://img.shields.io/pypi/v/monolith-lib)](https://pypi.org/project/monolith-lib/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Venus-ffd21e.svg)](https://huggingface.co/datasets/Elfsong/Venus)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Monolith-ffd21e.svg)](https://huggingface.co/spaces/Elfsong/Monolith)


> **Abstract:** Large Language Models (LLMs) generate functionally correct solutions but often fall short in code efficiency, a critical bottleneck for real-world deployment. In this paper, we introduce a novel test-time iterative optimization framework to address this, employing a closed-loop system where LLMs iteratively refine code based on empirical performance feedback from an execution sandbox. We explore three training strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Experiments on our Venus dataset and the APPS benchmark show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO, using reinforcement learning (RL) with execution feedback, continuously optimizes code performance, significantly boosting both pass@1 (from 47% to 62%) and the likelihood of outperforming human submissions in efficiency (from 31% to 45%). Our work demonstrates effective test-time code efficiency improvement and critically reveals the power of RL in teaching LLMs to truly self-improve code efficiency.

While current LLMs excel at generating code that works, can we trust that generated code in real-world applications? Often, the answer is **NO**. 
Since current research has largely focused on **functional correctness**, leaving **code efficiency** as a significant bottleneck.

To tackle this, we introduce **Afterburner**, an iterative framework that leverages reinforcement learning (RL) to instruct LLMs how to generate code that is not only correct but also efficient. Afterburner creates a self-improving loop, continually refining code for better performance:

- While SFT & DPO methods plateau, our RL approach shows continuous improvement 🔮
- Pass@1 boosted from 47% to 62% 📈 
- Outperforms human code efficiency likelihood jumps from 31% to 45% 🏆

---
## Step 1. Data (Venus)
Link: https://github.com/Elfsong/Venus

## Step 2. Environment (Monolith)
Link: https://github.com/Elfsong/Monolith

## Step 3. Algorithm (Afterburner)
SFT & DPO: https://github.com/hiyouga/LLaMA-Factory
GRPO: https://github.com/volcengine/verl

## Step 4. Evaluation (Litmus)
Link: https://github.com/Elfsong/Litmus

## Citation
```
@article{du2025afterburner,
  title={Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization},
  author={Du, Mingzhe and Luu, Anh Tuan and Liu, Yue and Qing, Yuhao and Huang, Dong and He, Xinyi and Liu, Qian and Ma, Zejun and Ng, See-kiong},
  booktitle={https://arxiv.org/abs/2505.23387},
  year={2025}
}
```
