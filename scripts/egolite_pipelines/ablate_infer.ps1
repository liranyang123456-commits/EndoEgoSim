Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
while (-not (Select-String -Path results\egolite\pipeline_r2.log -Pattern "done" -Quiet)) { Start-Sleep 60 }
$ck = "results/egolite/r_v2/last.pth"
foreach ($ab in @("nogn", "uniform", "nolayer", "novar", "staticonly")) {
    foreach ($e in @(@("lists/simtest92.txt", "simtest"), @("lists/scared_official.txt", "scared"),
                     @("lists/r2s_test.txt", "r2stest"), @("lists/abs_test94.txt", "abstest"))) {
        $tag = "egolite_r_v2_abl_$($ab)_$($e[1])"
        python scripts/eval_egolite.py --list $e[0] --ckpt $ck --tag $tag --ablate $ab *> "results/egolite/eval_$tag.log"
    }
}
foreach ($e in @(@("lists/r2s_test.txt", "r2stest"), @("lists/abs_test94.txt", "abstest"))) {
    python scripts/eval_egolite.py --list $e[0] --ckpt $ck --tag "egolite_r_v2_$($e[1])" *> "results/egolite/eval_egolite_r_v2_$($e[1]).log"
}
"done" | Out-File results\egolite\ablate_infer.log
