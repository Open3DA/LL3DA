<div align= "center">
    <h1> Official repo for LL3DA <img src="./assets/icon.png" width="35px"></h1>

</div>

<div align="center">
    <h2> <a href="https://ll3da.github.io/">LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning</a></h2>

<p align="center">
  <a href="https://ll3da.github.io/">💻Project Page</a> •
  <a href="https://arxiv.org/abs/2311.18651">📄Arxiv Paper</a> •
  <a href="https://www.youtube.com/watch?v=224JzkdHjfg">🎞YouTube</a> •
  🤗HuggingFace Demo (WIP) •
  <a href="#-citation">Citation
</p>

</div>

![teaser.gif](assets/teaser-simutaneous.gif)


<!-- <div align="center">
|                                                   Teaser Video                                                   |                                                    Demo Video                                                    |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| <video src="https://github.com/OpenMotionLab/MotionGPT/assets/120085716/a741e162-b2f4-4f65-af8e-aa19c4115a9e" /> | <video src="https://github.com/OpenMotionLab/MotionGPT/assets/120085716/ae966d17-6326-43e6-8d5b-8562cf3ffd52" /> |
</div> -->

<!-- ### [MotionGPT: Human Motion as a Foreign Language](https://motion-gpt.github.io/) -->
<!-- ### [Project Page](https://motion-gpt.github.io/) | [Arxiv Paper](https://arxiv.org/abs/2306.14795) | [HuggingFace Demo](xxx) -->

## 🏃 Intro LL3DA

LL3DA is a Large Language 3D Assistant that could respond to both visual and textual interactions with **complex 3D environments**.
<!-- 
<details>
    <summary><b>Technical details</b></summary> -->

Recent advances in Large Multimodal Models (LMM) have made it possible for various applications in human-machine interactions. However, developing LMMs that can comprehend, reason, and plan in complex and diverse 3D environments remains a challenging topic, especially considering the demand for understanding permutation-invariant point cloud 3D representations of the 3D scene. Existing works seek help from multi-view images, and project 2D features to 3D space as 3D scene representations. This, however, leads to huge computational overhead and performance degradation. In this paper, we present LL3DA, a Large Language 3D Assistant that takes point cloud as direct input and respond to both textual-instructions and visual-prompts. This help LMMs better comprehend human interactions and further help to remove the ambiguities in cluttered 3D scenes. Experiments show that LL3DA achieves remarkable results, and surpasses various 3D vision-language models on both 3D Dense Captioning and 3D Question Answering.

![pipeline.png](assets/pipeline.png)

<!-- <img width="1194" alt="pipeline" src="assets/pipeline.png">
</details> -->

## 🚩 News

- [2023/11/30] Upload paper and init project

## ⚡ Quick Start

<details>
  <summary><b>Setup and download</b></summary>

### 1. Environment

Our code is tested with CUDA 11.6 and Python 3.8.16. To run the codes, you should install the following packages:

```
h5py
scipy
cython
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
'torch=1.13.1+cu116'
'transformers=4.34.1'
'trimesh=2.35.39'
```

### 2. Data

**Step 1. Prepare ScanNet 3D Data**

1. Follow the instructions [here](https://github.com/ch3cook-fdu/Vote2Cap-DETR/tree/master/data/scannet) and download the ScanNetV2 dataset. 
2. Change the SCANNET_DIR to the scans folder in [data/scannet/batch_load_scannet_data.py](https://github.com/ch3cook-fdu/Vote2Cap-DETR/blob/master/data/scannet/batch_load_scannet_data.py#L16), and run the following commands.
```{bash}
cd data/scannet/
python batch_load_scannet_data.py
```

**Step 2. Prepare Language Annotations**

TODO


</details>

## 💻 Train your own models

<details>
  <summary><b>Training</b></summary>
</details>

<details>
  <summary><b>Evaluation</b></summary>
</details>


## 📖 Citation

If you find our code or paper helps, please consider starring ⭐ us and citing:

```{bibtex}
@misc{chen2023ll3da,
    title={LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning}, 
    author={Sijin Chen and Xin Chen and Chi Zhang and Mingsheng Li and Gang Yu and Hao Fei and Hongyuan Zhu and Jiayuan Fan and Tao Chen},
    year={2023},
    eprint={2311.18651},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgments

Thanks to [Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR), [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM), [Scan2Cap](https://github.com/daveredrum/Scan2Cap), and [3DETR](https://github.com/facebookresearch/3detr). We borrow some of their codes and data.


## License

This code is distributed under an [MIT LICENSE](LICENSE). If there are any problem regarding our project, please open an issue.
