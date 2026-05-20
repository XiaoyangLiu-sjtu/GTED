# [AI4Math@ICML 2025] GTED

📝 Official implementation for the paper:

[Generalized Tree Edit Distance (GTED): A Faithful Evaluation Metric for Statement Autoformalization](https://arxiv.org/abs/2507.07399)

‼️ The final version of this paper, [ASSESS](https://arxiv.org/abs/2509.22246), has been published at ICLR 2026. Click to view the [GitHub repository](https://github.com/XiaoyangLiu-sjtu/ASSESS).


## Project Structure

The repository is organized as follows.

```text
GTED/
├── benchmark/                          # Input benchmark data used by `evaluation_benchmark`
│   ├── minif2f/
│   │   ├── label.jsonl                 # Ground-truth formal statements
│   │   └── predict.jsonl               # Model-generated formal statements
│   └── proofnet/
│       ├── label.jsonl
│       └── predict.jsonl
├── experiment/                         # Stored experiment artifacts and metric outputs
│   ├── minif2f/                        # Results on miniF2F
│   └── proofnet/                       # Results on ProofNet
│       ├── human_evaluation.json       # Human annotations used as evaluation reference
│       ├── identity_match/             
│       ├── typecheck/                  
│       ├── bleu/                       
│       ├── definitional_equality/      
│       ├── beq/                        
│       ├── majority_vote/              
│       └── gted/                       
├── src/                                # Core implementation of GTED pipeline
│   ├── parser.py                       # Lean theorem parsing and syntax reorganization
│   ├── hover.py                        # Lean hover info extraction and rewriting/processing
│   ├── opter.py                        # Operator-tree construction, visualization, TED similarity
│   ├── verifier.py                     # Lean verification process scheduler / worker pool
│   └── evaluator.py                    # Baseline evaluators and metric summarization
├── main.py                             # Public entry points: tree_lean_codes / ted_lean_codes / evaluation_benchmark
├── utils.py                            # Shared utilities: I/O, syntax standardization, OPT build orchestration
└── README.md
```


## Quick Start
1. **Install Lean4.** Follow the official [Lean4 installation guide](https://leanprover-community.github.io/get_started.html).
2. **Clone the repository.** Clone this repository and enter the project directory.
3. **Build the project.** Follow the instructions on the [Lean REPL page](https://github.com/leanprover-community/repl.git) to set up Lean REPL and change the `DEFAULT_LEAN_WORKSPACE` & `MATHLIB_PATH` parameters in `src/verifier.py` & `utils.py` to your REPL & Mathlib paths.
4. **Evaluation.** There are three functions `tree_lean_codes`, `ted_lean_codes` and `evaluation_benchmark` in `main.py` for evaluation.
    - `tree_lean_codes`: Input header and formal statement to build the corresponding operator tree.
        ```shell
        # Function1: tree_lean_codes
        start_time = time.time()
        header_list = ["import Mathlib\n"] * 100
        formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 100
        tree_lean_codes(header_list, formal_statement_list)
        end_time = time.time()
        print(f"Time taken for one lean code: {end_time - start_time:.2f} seconds")
        ```
    
    - `ted_lean_codes`: Input a pair of header and formal statements, build the corresponding operator tree and calculate the TED similarity.
        ```shell
        # Function2: ted_lean_codes
        label_header_list = ["import Mathlib"] * 3
        label_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 3
        predict_header_list = ["import Mathlib"] * 3
        predict_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 3
        print(ted_lean_codes(label_header_list, label_formal_statement_list, predict_header_list, predict_formal_statement_list))
        ```

    - `evaluation_benchmark`: Just pass in miniF2F or ProofNet. Please note that you may need to change the format of your own json file. This function is currently adapted to the json file format in `experiment/{benchmark}/human_evaluation.json`.
        ```shell
        # Function3: evaluation_benchmark
        evaluation_benchmark("minif2f")
        evaluation_benchmark("proofnet")
        ```


## 4. Citation
```bibtex
@inproceedings{liu2025generalized,
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
