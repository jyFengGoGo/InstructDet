<p align="center">
  <h4 align="center"><font color="#966661">InstructDet</font>: Diversifying Referring Object Detection with Generalized Instructions</h4>
  <p align="center"><img src="./assets/teaser.png" alt="teaser" width="600px" /></p>
  <p align="center">
    <a href='https://github.com/jyFengGoGo/InstructDet'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/pdf/2310.05136.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  </p>
</p>

***
<font color="#966661">**InstructDET**</font>, a data-centric method for **referring object detection**(ROD) that localizes target objects based on user instructions.

Our ROD aims to execute diversified user detection instructions compared to visual grounding. For images with object bbxs, we use foundation models to produce human-like object detection instructions. By training a conventional ROD model with incorporating tremendous instructions, we largely push ROD towards practical usage from a data-centric perspective.

## Demo Video
[![Instruct demo video](./assets/cover.png)](https://www.youtube.com/watch?v=huRehKBSCDQ "Demo of InstructDet")

## Milestones

- [ ] Release code and online demo.
- [ ] Release the InDET dataset.
- [ ] Release paper.

## Examples
Our diversified referring object detection(DROD) model, visual comparison with [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT) and [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO).
<img src="./assets/DROD_comparison_1.png" alt="DROD_1" style="zoom: 25%;" />
<img src="./assets/DROD_comparison_2.png" alt="DROD_2" style="zoom: 25%;" />

## Cite

```bibtex
@article{dang2023instructdet,
  title={InstructDET: Diversifying Referring Object Detection with Generalized Instructions.},
  author={Dang, Ronghao and Feng, Jiangyan and Zhang, Haodong and Ge, Chongjian and Song, Lin and Gong, Lijun and Liu, Chengju and Chen, Qijun and Zhu, Feng and Zhao, Rui and Song, Yibin},
  journal={arXiv preprint arXiv:2310.05136},
  year={2023}
}
```

## Acknowledgement

This repo benefits from [LLaVA](https://github.com/haotian-liu/LLaVA), [Vicuna](https://github.com/lm-sys/FastChat), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.
