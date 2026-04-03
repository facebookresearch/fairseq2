for f in recipes/lm/train/scripts/scaling_experiment/run_qwen35_ablation_*.sh;

do sbatch "$f";

done
