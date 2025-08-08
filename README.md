# [AI4Math@ICML25] GTED

📝Official implementation for the paper:

[Generalized Tree Edit Distance (GTED): A Faithful Evaluation Metric for Statement Autoformalization](https://arxiv.org/abs/2507.07399)


## 1. Introduction
Statement autoformalization, the automated translation of statement from natural language into formal languages, has become a subject of extensive research, yet the development of robust automated evaluation metrics remains limited. Existing evaluation methods often lack semantic understanding, face challenges with high computational costs, and are constrained by the current progress of automated theorem proving. To address these issues, we propose GTED (Generalized Tree Edit Distance), a novel evaluation framework that first standardizes formal statements and converts them into operator trees, then determines the semantic similarity using the eponymous GTED metric. On the miniF2F and ProofNet benchmarks, GTED outperforms all baseline metrics by achieving the highest accuracy and Kappa scores, thus providing the community with a more faithful metric for automated evaluation.


## 2. Evaluation Results
**Result on miniF2F**
| **Metric** | **Precision** | **Recall** | **Accuracy** | **Kappa** |
| :--- | :---: | :---: | :---: | :---: |
| **Identity Match** | **100.00%** | 11.48% | 47.32% | 0.095 |
| **Typecheck** | 59.51% | **100.00%** | 59.51% | 0.000 |
| **BLEU** | 78.22% | 64.75% | 68.29% | 0.368 |
| **Majority Voting** | 88.00% | 54.10% | 68.29% | 0.397 |
| **Definitional Equality**| **100.00%** | 36.07% | 61.95% | 0.314 |
| **BEq** | 98.28% | 46.72% | 67.80% | 0.405 |
| **GTED (Ours)** | 88.75% | 58.20% | **70.73%** | **0.438** |

**Result on ProofNet**
| **Metric** | **Precision** | **Recall** | **Accuracy** | **Kappa** |
| :--- | :---: | :---: | :---: | :---: |
| **Identity Match** | 0.00% | 0.00% | 47.31% | 0.000 |
| **Typecheck** | 52.69% | **100.00%** | 52.69% | 0.000 |
| **BLEU** | 72.34% | 69.39% | **69.89%** | 0.398 |
| **Majority Voting** | 77.78% | 57.14% | 68.82% | 0.384 |
| **Definitional Equality**| 60.00% | 6.12% | 48.39% | 0.015 |
| **BEq** | **100.00%** | 16.33% | 55.91% | 0.156 |
| **GTED (Ours)** | 75.61% | 63.27% | **69.89%** | **0.402** |


## 3. Quick Start
1. **Install Lean4**
    Follow the instructions on the [Lean4 installation page](https://leanprover-community.github.io/get_started.html) to set up Lean4.

2. **Clone the repository**
    ```sh
    git clone https://github.com/XiaoyangLiu-sjtu/GTED.git
    cd GTED
    ```

3. **Build REPL**
    Follow the instructions on the [Lean REPL page](https://github.com/leanprover-community/repl.git) to set up Lean REPL and change the `DEFAULT_LEAN_WORKSPACE` parameter in `src/verifier.py` to your REPL path.
4. **Evaluation**
    - Evaluation on One Lean Code: function `test_one_lean_code` in `main.py`
    - Evaluation on Benchmark (miniF2F or ProofNet): function `evaluation` in `main.py`


## 4. Citation
```latex
@inproceedings{
liu2025generalized,
title={Generalized Tree Edit Distance ({GTED}): A Faithful Evaluation Metric for Statement Autoformalization},
author={Yuntian Liu and Tao Zhu and Xiaoyang Liu and Yu Chen and Liu ZhaoXuan and Guo qingfeng and Jiashuo Zhang and Kangjie Bao and Tao Luo},
booktitle={2nd AI for Math Workshop @ ICML 2025},
year={2025},
url={https://openreview.net/forum?id=824rq5iguB}
}
```


## 5. Contact
Feel free to discuss the paper/data/code with us through issues/emails!
- Xiaoyang Liu: xiaoyang.liu@sjtu.edu.cn