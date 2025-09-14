This is folder contains the github repo for [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/a485cbb49d692b5f88e38b5ae2b68331f277b9ea) introduced in:

@article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}}

We added some files that allow us to train and evalute H-UQNN and Evidential Deep Learning (EDL).
IN particular, we added: 
- [uqnn trainer](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/trainers/uqnn_trainer.py)
- [edl trainer](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/trainers/edl_trainer.py)
- [uqnn evaluator](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/evaluators/uqnn_evaluator.py)
- [edl evaluator](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/evaluators/edl_evaluator.py)
- [uqnn net](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/networks/uqnn_net.py)
- [edl net](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/networks/edl_net.py)
- [uqnn postprocessor](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/postprocessors/UQNNPostprocessor.py)
- [edl postprocessor](https://github.com/jwiggerthale/H-UQNN/blob/main/Classification/openood/postprocessor/edl_postprocessor.py)




Note that We did not use the trainer for H-UQNN for the tests conducted in our paper. 
