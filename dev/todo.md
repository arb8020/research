[~] igsm rl experiment - replicate Physics of LMs 2.2 + test RL
  [x] create pretrain configs
  [x] create LoRA SFT config
  [x] create GRPO RL configs
  [ ] run pretrain on clean data - 100k steps, ~2-3hrs on H100
  [ ] run experiments from checkpoint:
    [ ] full SFT on retry data - should work, paper baseline
    [ ] LoRA SFT on retry data - should fail, paper result
    [ ] GRPO RL full params
    [ ] GRPO RL + LoRA
  [ ] add eval harness for held-out hard problems
  [ ] analyze results
