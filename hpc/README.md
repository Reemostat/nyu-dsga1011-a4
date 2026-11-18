# Running A4 on NYU HPC (Torch burst nodes)

The code is already complete. On HPC you only set up the environment once, then submit jobs.
Greene is retired, so the GPUs are the **Torch burst nodes**, which you reach through **Open OnDemand (OOD)**.

## Step 0: Put the code on GitHub (from your laptop)

You have to submit a GitHub link anyway, so use GitHub to move the code to HPC.

```bash
cd "~/Desktop/nlp fall /nyu-dsga1011-a4-main"
git init && git add . && git commit -m "A4 implementation"
# create an EMPTY repo on github.com (e.g. nyu-dsga1011-a4), then:
git remote add origin https://github.com/<you>/nyu-dsga1011-a4.git
git branch -M main && git push -u origin main
```

## Step 1: Open a terminal on the burst side

1. Connect to the **NYU VPN** if you are off campus.
2. Go to https://ood-burst-001.hpc.nyu.edu/ and log in with NYU SSO. If the page misbehaves, try Safari or clear cookies.
3. Start an interactive app (Jupyter is fine) on a **CPU** partition such as `n2c48m24`.
   Select your course account, set the time to about 2h, and launch it.
   Once it is running: **Connect → File → New → Terminal**.

The burst nodes have their own `/scratch`, separate from Torch's. Do everything below from this terminal.

## Step 2: Clone and configure (once)

```bash
cd /scratch/$USER
git clone https://github.com/<you>/nyu-dsga1011-a4.git nyu-dsga1011-a4
#   private repo: use a GitHub personal access token as the password
cd nyu-dsga1011-a4/hpc

sacctmgr show assoc user=$USER format=account%50   # find your DS-GA 1011 account name
nano config.sh                                      # set SLURM_ACCOUNT_NAME on the first line
```

If you cloned somewhere other than `/scratch/$USER/nyu-dsga1011-a4`, also fix `A4_ROOT` in `config.sh`.

## Step 3: Build the environment (once, about 15–25 min)

```bash
bash setup_env.sh
```

This copies the CUDA Singularity image and a 25 GB overlay into `/scratch/$USER`.
It installs miniconda plus two envs, `a4p1` (Part 1 pins) and `a4p2` (Part 2 pins), and downloads the NLTK data.
At the end it prints `cuda: False`, which is expected on a CPU node. The jobs will see the GPU.

## Step 4: Submit jobs

All commands are run from the `hpc/` directory. Logs go to `hpc/logs/`.

| Step | Command | GPU time | Produces |
|---|---|---|---|
| Q1 sanity | `bash submit.sh part1_debug.sbatch` | ~10 min | log shows acc > 88% and 5 transformed examples |
| Q1 + Q2 | `bash submit.sh part1_q1_q2.sbatch` | ~50 min | `part1/out_original.txt`, `part1/out_transformed.txt` |
| Q3 | `bash submit.sh part1_q3.sbatch` (**after Q1+Q2 finishes**) | ~65 min | `part1/out_augmented_{original,transformed}.txt` |
| Part 2 | `bash submit.sh part2_t5.sbatch` | ~30–60 min | `part2/results/t5_ft_experiment_test.sql`, `part2/records/t5_ft_experiment_test.pkl` |

- Part 2 does not depend on Part 1, so you can submit it right away, in parallel with Part 1.
- Q3 has to wait for Q1+Q2: both jobs write to the same `part1/` folder and the Q3 model reuses that code.

Monitoring:

```bash
squeue -u $USER                     # PD = pending, R = running
tail -f logs/a4_p1_q1q2_<jobid>.out # live log
scancel <jobid>                     # kill a job
```

Burst nodes are **preemptible**. If a job dies partway, `--requeue` restarts it.
If it doesn't restart, resubmit it.

## Step 5: Check the numbers

- **Q1:** the log ends with `Score: {'accuracy': ...}` and you need **> 0.91**.
- **Q2:** look at the second `Score:` line. You need original minus transformed **> 4 points**.
  If the drop is smaller, raise `SYNONYM_PROB` (e.g. 0.35) and/or `TYPO_PROB` (e.g. 0.15) at the top of `part1/utils.py`.
  Then run `bash submit.sh part1_q2_only.sbatch` (eval only, no retraining).
  Do this **before** running Q3, since Q3 must use the final transformation.
- **Part 2:** the log prints `Record F1` for every epoch. The final model is the epoch with the best dev F1.
  You want the dev F1 to be at least 65–70 to be safe for the ≥ 65 **test** target.
  To run an ablation for Table 4: `bash submit.sh part2_t5.sbatch lr1e-3 --learning_rate 1e-3`
  or `bash submit.sh part2_t5.sbatch frzemb --freeze embeddings`.
  The final submission must use `--experiment_name experiment`, which is the default with no arguments.

## Step 6: Get the outputs back

Download these files with the OOD **Files** browser (Files → /scratch/$USER/nyu-dsga1011-a4):

- `part1/out_original.txt`, `out_transformed.txt`, `out_augmented_original.txt`, `out_augmented_transformed.txt`
- `part2/results/t5_ft_experiment_test.sql`, `part2/records/t5_ft_experiment_test.pkl`
- the logs in `hpc/logs/`, which hold the numbers for the report
- `part2/results/t5_ft_experiment_dev.sql`, used for the error analysis in Q6
- the Q7 checkpoint folder `part2/checkpoints/ft_experiments/experiment/best/` (~240 MB).
  Zip it first (`cd part2/checkpoints/ft_experiments/experiment && zip -r best.zip best`),
  then download it and upload it to a Google Drive folder for the report link.
