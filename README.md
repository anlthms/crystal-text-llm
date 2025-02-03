# Fine-Tuned Language Models Generate Stable Inorganic Materials as Text

This repository contains the code for the paper
[_Fine-Tuned Language Models Generate Stable Inorganic Materials as Text_](https://arxiv.org/abs/2402.04379)
by Nate Gruver, Anuroop Sriram, Andrea Madotto, Andrew Gordon Wilson, C. Lawrence Zitnick, and Zachary Ward Ulissi (ICLR 2024).

<figure>
  <img src="./assets/crystal_llm_graphic.png" alt="Image">
  <figcaption> We show that finetuned LLMs can be used to generate stable materials using string encodings. These finetuned LLMs can match or exceed the performance of a domain specific diffusion model (CDVAE). LLMs can also be used to mutate existing materials or to sample crystal structures conditioned on text descriptions. </figcaption>
</figure>

\
⚠️ Our method resembles but is not the same as CrystaLLM (https://arxiv.org/abs/2307.04340), which explores training language models from scratch on CIF-formatted crystals. You can find the code associated with that project at the following link: https://github.com/lantunes/CrystaLLM. ⚠️

## 🛠 Installation
Run the following command to install all dependencies. 
```
pip install -r requirements.txt
```

## 🚀 Training and Sampling Models
Run training with
```
python llama_finetune.py --run-name 7b-test-run --model 7b
```
and sample from a PEFT model with
```
python llama_sample.py --model_name 7b --model_path=exp/7b-test-run/checkpoint-500 --out_path=llm_samples.csv
```

If you are using Llama-3.2 use the scripts inside the [scripts](scripts) subdirectory.

## E above hull evaluation

To construct the convex hull, you must download structure entries from the following link:
https://figshare.com/articles/dataset/Matbench_Discovery_v1_0_0/22715158

Then energy values can be computed with
```
python e_above_hull.py --structures_fn [CSV FILE WITH STRUCTURES] --entries_fn [PATH TO STRUCTURE ENTRIES .json.gz FILE]
```

## License

The majority of crystall-llm is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/materialsproject/pymatgen is licensed under the MIT license, https://github.com/huggingface/transformers is licensed under Apache 2.0, and https://gitlab.com/ase/ase/-/ is licensed under GNU Lesser General License

## Citation
Please cite our work as:
```bibtex
@inproceedings{gruver2023llmtime,
    title={Fine-Tuned Language Models Generate Stable Inorganic Materials as Text},
    author={Nate Gruver, Anuroop Sriram, Andrea Madotto, Andrew Gordon Wilson, C. Lawrence Zitnick, and Zachary Ward Ulissi},
    booktitle={International Conference on Learning Representations 2024},
    year={2024}
}
```
