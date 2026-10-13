$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
function Wait-File($f, $pat) { while (-not ((Test-Path $f) -and (Select-String -Path $f -Pattern $pat -Quiet))) { Start-Sleep 60 } }
Wait-File results\egolite\pipeline_r2.log "done"
Wait-File results\egolite\ablate_infer.log "done"
Wait-File results\egolite\decouple_all.log "done"
$common = @("--batch", "3", "--clip", "8", "--workers", "6", "--abs-depth", "--flow-net", "large",
            "--head-extra", "6", "--ctx", "--val-abs", "sim_data_r2s/val")
$runs = @(
    @{name = "r_v3_gap"; args = @("--init", "results/egolite/r_v2/last.pth", "--iters", "2200", "--lr", "5e-5",
        "--val-every", "1100", "--train-flow", "--clip", "6",
        "--stride-p", "0.35,0.15,0.1,0.08,0.06,0.06,0.05,0.05,0.04,0.03,0.02,0.01",
        "--data", "sim_data/train:2,sim_data_abs/train:1,sim_data_r2s/train:2")},
    @{name = "abl_noph"; args = @("--init", "results/egolite/r_v1/last.pth", "--iters", "2200", "--lr", "5e-5",
        "--val-every", "1100", "--train-flow", "--no-ph", "--data", "sim_data/train:2,sim_data_abs/train:1,sim_data_r2s/train:1")},
    @{name = "abl_frozenflow"; args = @("--init", "results/egolite/r_v1/last.pth", "--iters", "2200", "--lr", "5e-5",
        "--val-every", "1100", "--data", "sim_data/train:2,sim_data_abs/train:1,sim_data_r2s/train:1")},
    @{name = "abl_nor2s"; args = @("--init", "results/egolite/s_big_ft/last.pth", "--iters", "6600", "--lr", "1e-4",
        "--val-every", "1100", "--train-flow", "--data", "sim_data/train:1,sim_data_abs/train:1")}
)
foreach ($r in $runs) {
    "[$(Get-Date -Format HH:mm)] train $($r.name)" | Out-File results\egolite\ablate_train.log -Append
    python scripts/train_egolite.py --out "results/egolite/$($r.name)" @common @($r.args) *> "results/egolite/$($r.name).log"
    $ck = "results/egolite/$($r.name)/last.pth"
    foreach ($e in @(@("lists/simtest92.txt", "simtest"), @("lists/sim_confirmatory265.txt", "core265"),
                     @("lists/scared_official.txt", "scared"), @("lists/stereomis.txt", "stereomis"),
                     @("lists/c3vd_full.txt", "c3vd"), @("lists/r2s_test.txt", "r2stest"),
                     @("lists/egomo_gtraj.txt", "egomo_gtraj"))) {
        $tag = "egolite_$($r.name)_$($e[1])"
        python scripts/eval_egolite.py --list $e[0] --ckpt $ck --tag $tag *> "results/egolite/eval_$tag.log"
    }
}
"[$(Get-Date -Format HH:mm)] done" | Out-File results\egolite\ablate_train.log -Append
