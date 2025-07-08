<img width="714" alt="image" src="https://github.com/user-attachments/assets/35dfa512-cbc4-42a7-8743-f8278f18847e" /># 🚀 Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2402.07844-b31b1b.svg)](https://arxiv.org/abs/2505.23387)
[![Monolith](https://img.shields.io/pypi/v/monolith-lib)](https://pypi.org/project/monolith-lib/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Venus-ffd21e.svg)](https://huggingface.co/datasets/Elfsong/Venus)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Elfsong/Monolith-ffd21e.svg)](https://huggingface.co/spaces/Elfsong/Monolith)

> **Abstract:** Large Language Models (LLMs) generate functionally correct solutions but often fall short in code efficiency, a critical bottleneck for real-world deployment. In this paper, we introduce a novel test-time iterative optimization framework to address this, employing a closed-loop system where LLMs iteratively refine code based on empirical performance feedback from an execution sandbox. We explore three training strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Experiments on our Venus dataset and the APPS benchmark show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO, using reinforcement learning (RL) with execution feedback, continuously optimizes code performance, significantly boosting both pass@1 (from 47% to 62%) and the likelihood of outperforming human submissions in efficiency (from 31% to 45%). Our work demonstrates effective test-time code efficiency improvement and critically reveals the power of RL in teaching LLMs to truly self-improve code efficiency.

While current LLMs excel at generating code that works, can we trust that generated code in real-world applications? Often, the answer is **NO**. 
Since current research has largely focused on **functional correctness**, leaving **code efficiency** as a significant bottleneck.

To tackle this, we introduce **Afterburner**, an iterative framework that leverages reinforcement learning (RL) to instruct LLMs how to generate code that is not only correct but also efficient. Afterburner creates a self-improving loop, continually refining code for better performance:

- 🔮 While SFT & DPO methods plateau, our RL approach shows continuous improvement.
- 📈 Pass@1 boosted from 47% to 62%.
- 🏆 Outperforms human code efficiency likelihood jumps from 31% to 45%.

## Overview
<p align="center">
  <img width="714" alt="image" src="https://github.com/user-attachments/assets/211a255f-00ab-428d-ac3c-bccb1926ace5" />
</p>

> In RL, there are three key components: **algorithm**, **environment**, and **priors**. For a long time, RL researchers focused mostly on the _**algorithm**_ (e.g. REINFORCE, DQN, TD-learning, actor-critic, PPO, TRPO…) – the intellectual core of how an agent learns – while treating the environment and priors as fixed or minimal. For example, Sutton and Barto’s classical textbook is all about algorithms and almost nothing about _**environments or priors**_. However, in the era of deep RL, it became clear that environments matter a lot empirically: an algorithm’s performance is often highly specific to the environment it was developed and tested in. -- [The Second Half](https://ysymyth.github.io/The-Second-Half/)

In this work, we introduce a novel iterative optimization framework (IOF) designed to enhance LLM-generated code efficiency through a closed-loop system of generation and evaluation, driven by **Monolith** and **Afterburner** trained on **Venus**.

## Step 1. Dataset (Venus)
**Venus** is the dataset used to train **Afterburner**. It is an extension of the original Mercury dataset and currently includes 6 languages: _Python3, C++, Javascript, Go, Rust, and Java_.

<p align="center">
  <img width="550" alt="image" src="https://github.com/user-attachments/assets/9b9c1471-703f-4b5c-b4bf-2ec70e5497cf" />
</p>

- **HF Dataset:** https://huggingface.co/datasets/Elfsong/Venus
- **Code:** https://github.com/Elfsong/Venus

## Step 2. Environment (Monolith)
**Monolith** is the code execution environment for **Afterburner**. It support parallel code execution for RL rollout (Isolated container with 100% CPU affinity) and high resolution performance measurement (10 kHz).
It measures three key metrics for each task from Venus: **1) Running Time**, **2) Memory Usage**, and **3) Integral Score** (The integral area of ​​running time versus memory usage).

<p align="center">
  <img width="397" alt="image" src="https://github.com/user-attachments/assets/3f7b518f-301c-4737-bee2-b1abffa27e0c" />
</p>

- **Demo:** https://monolith.cool/ (Too costly, [email me](mailto:mingzhe@nus.edu.sg) if you need it.)
- **Code:** https://github.com/Elfsong/Monolith (We recommend you to deploy your own Monolith.)

## Step 3. Algorithm (Afterburner)
We explore
<p align="center">
  <img width="533" alt="image" src="https://github.com/user-attachments/assets/eff9b2da-e0cd-4882-90d2-9246011a3bff" />
</p>

- **SFT & DPO:** https://github.com/hiyouga/LLaMA-Factory
- **GRPO:** https://github.com/volcengine/verl

## Step 4. Evaluation (Litmus)
Despite achieving high functional correctness (PASS@1), vanilla models generate code with strikingly inferior computational efficiency compared to human solutions

<p align="center">
  <img width="713" alt="image" src="https://github.com/user-attachments/assets/a3d01d4f-b446-4ff5-b10a-3c6fc3dbed83" />
</p>

- **Code:** https://github.com/Elfsong/Litmus

## Citation
```
@article{du2025afterburner,
  title={Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization},
  author={Du, Mingzhe and Luu, Anh Tuan and Liu, Yue and Qing, Yuhao and Huang, Dong and He, Xinyi and Liu, Qian and Ma, Zejun and Ng, See-kiong},
  booktitle={https://arxiv.org/abs/2505.23387},
  year={2025}
}
```
