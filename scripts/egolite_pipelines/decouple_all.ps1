Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
while (-not (Select-String -Path results\egolite\pipeline_r2.log -Pattern "done" -Quiet)) { Start-Sleep 60 }
foreach ($ck in @(@("results/egolite/r_v2/last.pth", "r_v2"), @("results/egolite/s_big_ft/last.pth", "s_big_ft"),
                  @("results/egolite/v3abs/best.pth", "v3abs"))) {
    foreach ($l in @(@("lists/r2s_test.txt", "r2stest"), @("lists/abs_test94.txt", "abstest"))) {
        python scripts/eval_decoupling.py --ckpt $ck[0] --list $l[0] --out "results/egolite/decouple_$($ck[1])_$($l[1]).json" *> "results/egolite/decouple_$($ck[1])_$($l[1]).log"
    }
}
"done" | Out-File results\egolite\decouple_all.log
