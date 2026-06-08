# Why naive INT8 destroys a Gaussian Splatting model — and how local ranges fix it

*A debugging story from building [splat-slim](https://github.com/Daceyyreal/splat-slim), a training-free compressor for 3D Gaussian Splatting.*

I had a bug that looked like a catastrophe. I quantized a trained 3D Gaussian Splatting model to 8 bits, expected a small quality hit, and instead watched reconstruction quality fall off a cliff — to about **15 dB PSNR**, which is the visual equivalent of static. The strange part wasn't that it broke. It was *how* it broke: every scene collapsed to roughly the same number.

That detail turned out to be the whole story.

## What's actually inside a splat file

A trained 3D Gaussian Splatting (3DGS) scene is just a list of Gaussians. Each one carries a position, an opacity, a 3D scale, a rotation, and color stored as spherical-harmonic (SH) coefficients — a direct-current term plus higher-order bands for view-dependent shading. For a real scene that's millions of Gaussians, and every attribute is a 32-bit float. The exported `.ply` ends up at hundreds of megabytes: the Garden scene I work with is 390 MB, Bicycle is 657 MB.

For something you might want to stream to a browser or ship to a phone, that's far too heavy. So you compress. And the most obvious lever is precision: a 32-bit float carries far more resolution than the eye needs, so quantizing to 8 bits is an easy 4× on the fields you apply it to.

## The obvious way to do it (and why it explodes)

INT8 quantization maps a float range onto 256 integer levels. You pick a `min` and a `max`, then linearly map every value in `[min, max]` onto `0..255`. Decompression reverses the map. The entire quality of the result hinges on one decision: **what range do those 256 levels have to cover?**

The naive choice is one range per attribute column. Take every Gaussian's `scale_x`, find the global minimum and maximum across the whole model, and quantize that column against it. Simple, fast, one range per field. Here is what it did, at SH degree 3:

| Scene (SH3) | FP16 (reference) | Per-column INT8 |
| ----------- | ---------------- | --------------- |
| Garden      | 26.24 dB         | 14.92 dB        |
| Bicycle     | 22.77 dB         | 14.69 dB        |
| Vase        | 23.92 dB         | 14.85 dB        |

*Per-column INT8 lands at ~15 dB on every scene. The full FP16 / mixed / per-column / sub-group sweep across SH levels is in `fig3_quantization_compare.png`.*

Fifteen decibels is not "slightly worse." It's gone.

## The clue: it failed the same way everywhere

Three very different scenes — a detailed outdoor garden, a bike on gravel, a self-captured indoor object — all landed at the same PSNR. That scene-independence is the tell. If the problem were about scene difficulty or content, the scenes would degrade by different amounts. When the number is flat across wildly different inputs, the failure isn't in the content. It's in the *machinery* — something structural that doesn't care what the scene is.

The structural thing here is the range. Gaussian attribute distributions have heavy tails: most values cluster tightly, but a handful of outliers sit far out. A single global `[min, max]` is stretched by those outliers to cover an enormous span. The 256 levels then get spread thinly across that span, so the densely-packed bulk of real values — the part that actually matters — gets crushed into a tiny handful of integer codes. Almost every Gaussian rounds to nearly the same value. Geometry, which has the widest dynamic range, suffers worst.

And critically: **more bits would not have saved this.** The error came from where the range boundaries sat, not from how finely the range was subdivided. That is the lesson hiding inside the bug.

## The fix: stop making everyone share one ruler

If one global range per column is the problem, the answer is more rulers — many local ranges instead of one global one. Partition the Gaussians into groups, and give each group its own per-column `min`/`max`. Now each group's 256 levels only have to cover that group's much tighter local spread, and the crushing disappears.

But the partition matters. You want each group to contain Gaussians whose attribute values are *already* similar, so the local range is genuinely small. Spatial proximity is a good proxy: Gaussians that sit near each other in 3D tend to have similar scales and colors.

So I sort the Gaussians along a **Morton (Z-order) curve**. A Morton curve is a space-filling ordering that interleaves the bits of the x, y, z coordinates, turning a 3D position into a single number while largely preserving locality — points that are close in 3D stay close in the 1D order. Sort by that key, then chop the sorted list into a fixed number of contiguous groups (around a thousand for these scenes). Each group is now a spatially-coherent cluster with a tight local range, and gets its own quantization bounds.

The result, at SH degree 3:

| Scene (SH3) | FP16 (reference) | Sub-group INT8 | Gap     |
| ----------- | ---------------- | -------------- | ------- |
| Garden      | 26.24 dB         | 25.82 dB       | 0.42 dB |
| Bicycle     | 22.77 dB         | 22.11 dB       | 0.66 dB |
| Vase        | 23.92 dB         | 23.84 dB       | 0.08 dB |

From "static" back to within **0.08–0.66 dB of FP16** — without adding a single bit. Same 8 bits per value; the only thing that changed was that the ranges became local. And because INT8 packs tighter than FP16, this is also the *smallest* variant on disk: on Garden it's 85.7 MB (a 78% cut from baseline) versus 101 MB for the mixed-precision configuration, at almost the same quality.

## Making it self-decodable

A subtlety that's easy to get wrong: once each group has its own range, the file has to *carry* those ranges, or it can't decode itself. My first cut leaned on global statistics and quietly assumed they'd be recomputable later — which is exactly how you ship a compressed file that nobody else can open.

So the per-group ranges travel with the model, in a sidecar `.meta.npz` next to the quantized integers. Decompression needs nothing but the file itself: read the group's stored range, invert the map, done. No recomputation, no external state, no surprises.

## Three things I took away

1. **Quantization error is about range granularity, not just bit-width.** When a single range is dominated by outliers, adding bits is the wrong fix; partitioning into locally-coherent groups is the right one. This generalizes well beyond 3DGS — it's the same reason neural-network quantization moved from per-tensor to per-channel and per-group scales.
2. **Scene-independent failure is a diagnostic, not just a symptom.** The flat 15 dB across three different scenes was the single most useful piece of information I had. When something breaks *identically* regardless of input, look at the pipeline, not the data.
3. **A compressed artifact has to be self-contained.** If decoding silently depends on stats you happened to have lying around, you haven't compressed the model — you've compressed half of it.

## See it for yourself

The collapse is more convincing when you can drag a slider and watch it happen. [splat-lab](https://daceyyreal.github.io/splat-lab/) runs the whole pipeline live in the browser — flip the quantization control to INT8 and the model falls apart in front of you; the size readout is computed honestly from the field layout, so the numbers are the real numbers.

A practical note on the tooling: [splat-slim](https://github.com/Daceyyreal/splat-slim) ships **mixed precision** (FP16 geometry + INT8 appearance) as its recommended, fully-supported default — it sidesteps the collapse entirely and already gives ~74% size reduction. The Morton-ordered sub-group INT8 described here is the research method from my undergraduate work; folding it into the CLI as a first-class quantization mode is on the roadmap.

---

*Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [gsplat](https://github.com/nerfstudio-project/gsplat); evaluated on MipNeRF-360 plus a self-captured scene. Comments and corrections welcome.*
