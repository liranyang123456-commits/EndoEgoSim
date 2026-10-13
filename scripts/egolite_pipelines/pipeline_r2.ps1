$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
"[$(Get-Date -Format HH:mm)] training r_v2 (fine-tune r_v1, old:abs:r2s=2:1:1)" | Out-File results\egolite\pipeline_r2.log -Append
python scripts/train_egolite.py --out results/egolite/r_v2 --iters 2200 --batch 3 --clip 8 --lr 5e-5 `
    --val-every 1100 --workers 6 --abs-depth --init results/egolite/r_v1/last.pth `
    --flow-net large --head-extra 6 --ctx --train-flow `
    --data "sim_data/train:2,sim_data_abs/train:1,sim_data_r2s/train:1" --val-abs sim_data_r2s/val `
    *> results/egolite/r_v2.log
"[$(Get-Date -Format HH:mm)] evaluating r_v2" | Out-File results\egolite\pipeline_r2.log -Append
$ck = "results/egolite/r_v2/last.pth"
$evals = @(
    @("lists/sim_confirmatory265.txt", "egolite_r_v2_core265", 64),
    @("lists/scared_official.txt", "egolite_r_v2_scared", 64),
    @("lists/c3vd_full.txt", "egolite_r_v2_c3vd", 64),
    @("lists/simtest92.txt", "egolite_r_v2_simtest", 64),
    @("lists/sim_confirmatory_extension93.txt", "egolite_r_v2_ext93", 64),
    @("lists/stereomis.txt", "egolite_r_v2_stereomis", 64),
    @("lists/egomo_gtraj.txt", "egolite_r_v2_egomo_gtraj", 64),
    @("lists/stereomis.txt", "egolite_r_v2_stereomis_full", 0),
    @("lists/scared_official.txt", "egolite_r_v2_scared_full", 0),
    @("lists/c3vd_full.txt", "egolite_r_v2_c3vd_full", 0)
)
foreach ($e in $evals) {
    python scripts/eval_egolite.py --list $e[0] --ckpt $ck --tag $e[1] --max-frames $e[2] *> "results/egolite/eval_$($e[1]).log"
}
python scripts/eval_egolite.py --list lists/stereomis.txt --ckpt results/egolite/s_big_ft/last.pth --tag egolite_s_big_ft_stereomis_full --max-frames 0 *> results/egolite/eval_egolite_s_big_ft_stereomis_full.log
"[$(Get-Date -Format HH:mm)] done" | Out-File results\egolite\pipeline_r2.log -Append
