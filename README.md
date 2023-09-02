# N-Shot-Benchmarking-of-Whisper
N-Shot Benchmarking of Whisper

## To run
1. Install dependencies in your environment:  `pip install -r requirementx.txt`
2. Run train.py as: `python train.py --training_args` OR
3. Run train.sh as: `bash train.sh` OR
4. Submit a  slurm job for training: `sbatch --job_args train.sh --training_args`
5. You can use multiple launcher including `deepspeed`, `torchrun`, and `python`.
6. Remove `--deepspeed "ds_config2.json"` and change launcher from `deepspeed` to `torchrun` or `python` if you don't want to use deepspeed. 
   

### Citation
```
@article{talafha2023n,
  title={N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition},
  author={Talafha, Bashar and Waheed, Abdul and Abdul-Mageed, Muhammad},
  journal={arXiv preprint arXiv:2306.02902},
  year={2023}
}
```
