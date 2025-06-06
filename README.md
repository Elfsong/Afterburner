# 🚀 Afterburner

Large Language Models (LLMs) generate functionally correct solutions but often fall short in code efficiency, a critical bottleneck for real-world deployment. In this paper, we introduce a novel test-time iterative optimization framework to address this, employing a closed-loop system where LLMs iteratively refine code based on empirical performance feedback from an execution sandbox. We explore three training strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Experiments on our Venus dataset and the APPS benchmark show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO, using reinforcement learning (RL) with execution feedback, continuously optimizes code performance, significantly boosting both pass@1 (from 47% to 62%) and the likelihood of outperforming human submissions in efficiency (from 31% to 45%). Our work demonstrates effective test-time code efficiency improvement and critically reveals the power of RL in teaching LLMs to truly self-improve code efficiency.

## Public Release
As you can see, this repo is for research purpose. You can check [this repo](https://github.com/Elfsong/Litmus/tree/main) out for practical evaluation and further development.

## Citation
```
@article{du2025afterburner,
  title={Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization},
  author={Du, Mingzhe and Luu, Anh Tuan and Liu, Yue and Qing, Yuhao and Huang, Dong and He, Xinyi and Liu, Qian and Ma, Zejun and Ng, See-kiong},
  booktitle={https://arxiv.org/abs/2505.23387},
  year={2025}
}
```
