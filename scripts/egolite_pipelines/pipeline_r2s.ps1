$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
function Wait-Procs($pattern) {
    while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }) {
        Start-Sleep 60
    }
}
"[$(Get-Date -Format HH:mm)] waiting for R2S generation" | Out-File results\egolite\pipeline_r2s.log -Append
Wait-Procs 'r2s_generator'
Get-ChildItem sim_data_r2s\val -Directory | ForEach-Object { $_.FullName } | Set-Content lists\r2s_val.txt
Get-ChildItem sim_data_r2s\test -Directory | ForEach-Object { $_.FullName } | Set-Content lists\r2s_test.txt
"[$(Get-Date -Format HH:mm)] generation done; waiting for feed-forward baselines" | Out-File results\egolite\pipeline_r2s.log -Append
Wait-Procs 'eval_slam.py --method (pi3|reloc3r)'
"[$(Get-Date -Format HH:mm)] training r_v1" | Out-File results\egolite\pipeline_r2s.log -Append
python scripts/train_egolite.py --out results/egolite/r_v1 --iters 6600 --batch 3 --clip 8 --lr 1e-4 `
    --val-every 1100 --workers 6 --abs-depth --init results/egolite/s_big_ft/last.pth `
    --flow-net large --head-extra 6 --ctx --train-flow `
    --data "sim_data/train:1,sim_data_abs/train:1,sim_data_r2s/train:2" --val-abs sim_data_r2s/val `
    *> results/egolite/r_v1.log
"[$(Get-Date -Format HH:mm)] evaluating r_v1" | Out-File results\egolite\pipeline_r2s.log -Append
$ck = "results/egolite/r_v1/last.pth"
$evals = @(
    @("lists/simtest92.txt", "egolite_r_v1_simtest", 64),
    @("lists/stereomis.txt", "egolite_r_v1_stereomis", 64),
    @("lists/scared_official.txt", "egolite_r_v1_scared", 64),
    @("lists/c3vd_full.txt", "egolite_r_v1_c3vd", 64),
    @("lists/egomo_gtraj.txt", "egolite_r_v1_egomo_gtraj", 64),
    @("lists/abs_test94.txt", "egolite_r_v1_abstest", 64),
    @("lists/r2s_test.txt", "egolite_r_v1_r2stest", 64),
    @("lists/sim_confirmatory_extension93.txt", "egolite_r_v1_ext93", 64),
    @("lists/sim_confirmatory265.txt", "egolite_r_v1_core265", 64),
    @("lists/scared_official.txt", "egolite_r_v1_scared_full", 0),
    @("lists/c3vd_full.txt", "egolite_r_v1_c3vd_full", 0)
)
foreach ($e in $evals) {
    python scripts/eval_egolite.py --list $e[0] --ckpt $ck --tag $e[1] --max-frames $e[2] *> "results/egolite/eval_$($e[1]).log"
}
python scripts/eval_egolite.py --list lists/r2s_test.txt --ckpt results/egolite/s_big_ft/last.pth --tag egolite_s_big_ft_r2stest *> results/egolite/eval_s_big_ft_r2stest.log
"[$(Get-Date -Format HH:mm)] done" | Out-File results\egolite\pipeline_r2s.log -Append
