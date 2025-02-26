# flatreward-RRL

This is the official PyTorch implementation of our paper "Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning" 

> **Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning**  
> Hyun Kyu Lee, Sung Whan Yoon  
> **Accepted by: ICLR 2025**
>
> [[ICLR 2025](https://openreview.net/forum?id=4OaO3GjP7k)]

## Prerequisites

Install the required dependencies as below:

- `gym == 0.21.0`
- `mujoco-py == 2.1.2.14`
- `torch == 1.12.1`


## Experiments

### Train Policy
 - `python train_sam_ppo.py --config configs/Walker.yaml --use_sam`
### Evaluate Policy 
 - `python eval_action_sam_ppo.py --config configs/Walker.yaml`

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{leeflat,
  title={Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning},
  author={Lee, Hyun Kyu and Yoon, Sung Whan},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

