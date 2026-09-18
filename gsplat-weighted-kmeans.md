# My paper's fix didn't help gsplat. Measuring the loss found one that did.

*Two negative results and one measured win from trying to improve the PNG codec in [gsplat](https://github.com/nerfstudio-project/gsplat), the Gaussian Splatting library from the Nerfstudio project. The win is now an open pull request, [#1063](https://github.com/nerfstudio-project/gsplat/pull/1063).*

One finding in my ICCA paper is a failure. Quantize a 3D Gaussian Splatting model to 8 bits with one min/max per attribute column, and quality collapses: a few outliers stretch the range and crush everything else into a handful of integer codes. Give small groups of splats their own ranges and it comes back. ([The full story is here](int8-collapse-morton-fix.html).) Then I read gsplat's `PngCompression`, the codec behind its compression benchmark, and found one min/max per channel. It looked like the same problem.

**The short version: the fix from my paper didn't transfer to a real codec, and measuring where the loss actually sat found a better one, worth +0.111 dB at the same file size.** The two dead ends on the way are the part I trust most.

## What the codec does

`PngCompression` sorts the splats into a 2D grid and stores each parameter as an image: the log-transformed means as 16-bit PNGs, everything else as 8-bit PNGs, each channel against one global min/max. The higher-order spherical-harmonic coefficients (shN) go through K-means instead: 65,536 centroids and a codebook quantized to 6 bits with one scalar min/max. All experiments use gsplat's own benchmark scripts (MCMC, 1M Gaussians) on Kaggle T4s.

## Hypothesis 1: the ranges are too wide

The direct port of my fix is tile-wise ranges: give each tile of the grid its own min/max per channel. The collapse does exist here. With the means forced to 8 bits, one range per channel gives 14.20 / 14.53 dB on garden / bicycle, and tile ranges bring that back to 21.70 / 20.50 dB. But the codec stores the means with 16 bits, so its default never enters that regime.

At the real bit depths, tiles help a little and cost a lot: tile 8 gains +0.027 / +0.054 dB and makes the output 27.1% / 29.5% bigger. The fair competitor is the cheapest one: keep the ranges you have and use fewer bits. I traced that curve, and **all 22 tile and smooth-range configs whose size fell inside it landed below it at matched size**, the best by 0.068 / 0.087 dB. A second, independent training agreed. Against the current codec, tiles look like a small win. Against "just use fewer bits", they lose.

## Stop guessing: split the loss

My next idea was already lined up. This time I measured first, swapping one stage at a time for its uncompressed input: **U** is the uncompressed checkpoint, **P** leaves the PNG parameters as raw floats, **S** leaves shN raw, and **F** clusters shN but keeps float centroids.

| Loss term | garden | bicycle |
|---|---|---|
| Total: U minus compressed | 0.434 dB | 0.237 dB |
| shN compression: U minus P | 0.403 dB | 0.176 dB |
| PNG parameters: U minus S | 0.034 dB | 0.063 dB |
| **Clustering: S minus F** | **0.374 dB** | **0.163 dB** |
| 6-bit codebook: F minus compressed | 0.026 dB | 0.011 dB |

The split is approximate, since PSNR losses don't add exactly, but the shape is clear. Everything hypothesis 1 was about costs 0.034 / 0.063 dB. The clustering costs 0.374 / 0.163 dB: a million splats share 65,536 centroids.

**My second hypothesis died on this table.** I had been eyeing the codebook, one scalar range over the whole thing, the one-ruler pattern from my paper again. The table capped any fix there at 0.026 / 0.011 dB.

## Hypothesis 2: a better ruler for the codebook

I ran it anyway, as a test of the table. One min/max per codebook dimension at 8 bits gained **+0.026 / +0.011 dB**, the predicted ceiling to the third decimal, for 5.2% / 4.2% more bytes. A small lever, correctly measured, and not a pull request. It also meant I could trust the table's other row.

## Hypothesis 3: which centroids, not how they're stored

The library clusters with TorchPQ, using Manhattan distance, and every splat counts the same. They don't count the same in the image: a large, opaque splat covers far more of the frame than a small, nearly transparent one.

Run 3 changed only the clustering: same checkpoints, sort, format and decoder. I wrote my own Lloyd's algorithm with per-splat weights, checked it with every weight set to one (it has to match TorchPQ's Euclidean mode), then weighted by opacity, and by opacity × area, where area is the splat's largest cross-section. I also gave TorchPQ 300 iterations instead of 100. At K-means seed 0, relative to the library:

| Clustering | garden PSNR | bicycle PSNR | garden size | bicycle size |
|---|---|---|---|---|
| Euclidean (TorchPQ) | −0.038 dB | −0.022 dB | +0.18% | +0.05% |
| Lloyd, weight 1 | −0.036 dB | −0.023 dB | +0.18% | +0.05% |
| Lloyd, opacity | +0.032 dB | +0.016 dB | −0.19% | −0.08% |
| **Lloyd, opacity × area** | **+0.082 dB** | **+0.027 dB** | **−0.25%** | **−0.14%** |
| Manhattan, 300 iterations | −0.0003 dB | +0.0022 dB | −0.0004% | +0.006% |

The surprise is the first row: **plain Euclidean clustering is worse than the Manhattan distance the library already uses.** Had I only tested the distance, I would have stopped there. The weighting is what matters: opacity turns the loss into a gain, and area adds more. The last row rules out convergence as the lever. Over three K-means seeds, opacity × area gained +0.082 / +0.104 / +0.103 dB on garden and +0.027 / +0.031 / +0.033 dB on bicycle, a mean 5.1× / 4.6× the baseline's own seed-to-seed spread.

## Nine scenes, then a dataset it had never seen

Run 4 took all 9 MipNeRF360 scenes from gsplat's benchmark, under a rule written down before the run. Run 5 swapped in the library code from the PR branch, which gave identical metrics and file sizes on all 9, and then ran it on Tanks & Temples, which played no part in choosing the method.

| Scene | Loss of the current codec | PSNR gain | Size change |
|---|---|---|---|
| garden | 0.434 dB | +0.082 dB | −0.25% |
| bicycle | 0.237 dB | +0.027 dB | −0.14% |
| stump | 0.290 dB | +0.048 dB | +0.06% |
| bonsai | 0.859 dB | +0.272 dB | −0.30% |
| counter | 0.546 dB | +0.189 dB | +0.25% |
| kitchen | 0.824 dB | +0.247 dB | −0.47% |
| room | 0.386 dB | +0.085 dB | −0.21% |
| treehill | 0.101 dB | +0.005 dB | −0.15% |
| flowers | 0.219 dB | +0.044 dB | +0.27% |
| T&T train | 0.171 dB | +0.061 dB | +0.001% |
| T&T truck | 0.188 dB | +0.044 dB | +0.153% |

*Loss = uncompressed PSNR minus current-codec PSNR, same checkpoint.*

- **MipNeRF360:** +0.111 dB mean PSNR at −0.10% mean size. **Tanks & Temples:** +0.052 dB at +0.08%.
- **All 11 scenes gain.** LPIPS is lower on all 11 and SSIM higher on 10 (garden: −0.00016).
- **Same format.** File layout, codebook and decoder are untouched.

The gain tracks the loss: the weighting recovers 21.1% of the codec's loss on MipNeRF360 on average, most on the indoor scenes, where the loss is largest.

## What it costs

K-means takes 1.26× the TorchPQ time on MipNeRF360 on average (less on Tanks & Temples), and peak GPU memory during compression rises by 2.11 to 2.35 GB; a `kmeans_chunk_size` option shrinks the largest buffer without changing the result. The [PR](https://github.com/nerfstudio-project/gsplat/pull/1063) has the full cost table.

## How I kept myself honest

**Write the rule down before the run.** Every run had a pass/fail rule fixed in advance, never changed after its results came in. Run 4's: mean PSNR gain above zero, a gain on all scenes but at most one, none worse than −0.02 dB, mean LPIPS no worse, mean SSIM change no lower than −0.0002, no output more than 0.3% larger. Run 5 reused it.

**Compare in pairs.** Every delta is a candidate minus the baseline from the same session, on the same checkpoint. My baseline sits 0.20 dB above the repo's published MipNeRF360 row; measuring against that row would have credited my method with 0.20 dB it didn't earn.

**Measure the noise floor, then respect it.** Re-running the unchanged codec with three K-means seeds moves PSNR by 0.019 / 0.007 dB and SSIM by 0.000158 / 0.000051. Run 3's rule ignored that: it demanded a match or better in every cell, with zero tolerance. Opacity × area gained PSNR and LPIPS in all six cells and failed two garden SSIM cells, −0.00016 and −0.00009, both inside the baseline's own spread. So the recorded run-3 verdict for the method that went on to win all 11 scenes is `pr_worthy = false`. I left it, because a rule relaxed after seeing the answer isn't a rule, and wrote run 4's rule with a small tolerance on mean SSIM. A rule that looks rigorous but is measuring seed noise says no to a real win.

**Correct the record where the claim was made.** After run 3 I wrote that TorchPQ's K-means is deterministic per seed across sessions. Run 5 re-ran it on all nine scenes: identical on 7, off by +0.0024 dB and +0.00006 dB on the other two. The result survives (paired against that re-run, the gain is +0.1106 dB), but the sentence was wrong, so the findings file corrects it where it was written.

## Takeaways

1. **A failure you've proven is only a hypothesis in someone else's code.** The collapse reproduced in gsplat, at a bit depth the codec doesn't use.
2. **Beat the cheapest alternative, not the status quo.** Tile ranges beat the current codec and lost to "use fewer bits".
3. **Decompose the loss before picking a fix.** One table killed a hypothesis, predicted its ceiling, and pointed at the term worth 0.374 / 0.163 dB.
4. **Fix the rule before the run, and size it to the noise.** Too strict, and it rejects real wins. Changed afterwards, and it rejects nothing.

## The pull request

The change is open as [gsplat#1063](https://github.com/nerfstudio-project/gsplat/pull/1063): the new backend with the default output unchanged, a memory knob, and a separate, droppable commit that makes it the default. Every number here comes from [FINDINGS.md](https://github.com/Daceyyreal/gsplat/blob/bench/tilequant/kaggle/FINDINGS.md), which records all five runs, their rules, their verdicts and the corrections.

---

*Built on [gsplat](https://github.com/nerfstudio-project/gsplat) and its compression benchmark; evaluated on MipNeRF360 and Tanks & Temples on Kaggle T4 GPUs. Comments and corrections welcome.*
