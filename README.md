# 🚀 Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2402.07844-b31b1b.svg)](https://arxiv.org/abs/2505.23387)
[![Monolith](https://img.shields.io/pypi/v/monolith-lib)](https://pypi.org/project/monolith-lib/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Venus-ffd21e.svg)](https://huggingface.co/datasets/Elfsong/Venus)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Monolith-ffd21e.svg)](https://huggingface.co/spaces/Elfsong/Monolith)


Large Language Models (LLMs) generate functionally correct solutions but often fall short in code efficiency, a critical bottleneck for real-world deployment. In this paper, we introduce a novel test-time iterative optimization framework to address this, employing a closed-loop system where LLMs iteratively refine code based on empirical performance feedback from an execution sandbox. We explore three training strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Experiments on our Venus dataset and the APPS benchmark show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO, using reinforcement learning (RL) with execution feedback, continuously optimizes code performance, significantly boosting both pass@1 (from 47% to 62%) and the likelihood of outperforming human submissions in efficiency (from 31% to 45%). Our work demonstrates effective test-time code efficiency improvement and critically reveals the power of RL in teaching LLMs to truly self-improve code efficiency.

🔮 The research explores three distinct training strategies for the Afterburner models:
- **Supervised Fine-Tuning (SFT):** Learns to mimic transformations from inefficient to efficient code examples.
- **Direct Preference Optimization (DPO):** Aligns the model with efficiency preferences by learning from ranked pairs of code solutions.
- **Group Relative Policy Optimization (GRPO):** A reinforcement learning (RL) approach that uses live execution feedback to continuously refine its optimization strategies.

💭 Experiments were conducted on a new, rigorously curated dataset named Venus and the existing APPS benchmark. The Venus dataset was created to facilitate robust efficiency assessment, providing a large number of human-written solutions for each problem to establish a reliable performance baseline.

📈 The results show that while SFT and DPO models provide initial efficiency gains, they quickly plateau. In contrast, the GRPO model, powered by reinforcement learning, demonstrates continuous self-improvement. This approach significantly boosted the pass rate (PASS@1) from 47% to 62% and increased the likelihood of generating code more efficient than human submissions from 31% to 45%. The study concludes that leveraging reinforcement learning with direct execution feedback is a highly effective strategy for teaching LLMs to generate genuinely high-performance code.

## Step 1. Data (Venus)

## Step 2. Environment (Monolith)

## Step 3. Algorithm (Afterburner)

## Citation
```
@article{du2025afterburner,
  title={Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization},
  author={Du, Mingzhe and Luu, Anh Tuan and Liu, Yue and Qing, Yuhao and Huang, Dong and He, Xinyi and Liu, Qian and Ma, Zejun and Ng, See-kiong},
  booktitle={https://arxiv.org/abs/2505.23387},
  year={2025}
}
```
