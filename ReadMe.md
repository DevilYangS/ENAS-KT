
## Introduction

This project is the source code of the paper entitled [*Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing*](https://arxiv.org/abs/2310.01180)


## Usage

```
Step 1 is necessary, 
Step 2~3 are too time-consuming, 
you could directly go to step 4 after step 1, since we uploaded the pretrained model weights, you can download the pretrained weights of supetnet from *Release*.

---------------------1. prepare dataset-------------------------
(1) download EdNet and RAIEdNet2020 from the website;
(2) unzip the downloaded file;
(3) run the 'process_data/Pre_process_EdNet.py' or 'process_data/Pre_process_RAIEd.py'  for generating 'interaction.csv' for EdNet and RAIEdNet2020 


---------------------2. train supernet-------------------------
python  training_script.py --data_dir datapath --dataset EdNet --evalmodel weight-sharing --epochs 60

---------------------3. Evolutionary Search-------------------------
python EvoTransformer.py --data_dir datapath --dataset EdNet  --pre_train_path path2pth

---------------------4. Fine-tune the best architecture-------------------------
 python  training_script.py --data_dir datapath --dataset EdNet --evalmodel single --epochs 30 --pre_train_path Super_pth/120/cross_True_fold_t_epoch_best.pth
--NAS  [[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                       [0, 0, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1], [1, 0, 1, 2, 1, 4, 2, 2, 3, 0, 2, 1],
                       [[0, 0, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1], [0, 1, 0, 0]]                      ]
```








## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.
```
@inproceedings{yang2023evolutionary,
  title={Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing},
  author={Yang, Shangshang and Yu, Xiaoshan and Tian, Ye and Yan, Xueming and Ma, Haiping and Zhang, Xingyi},
  journal={Proceedings of 37-th Conference on Neural Information Processing Systems},
  year={2023}
}
```




