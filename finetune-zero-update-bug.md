# The fine-tuning run that changed nothing

*Why a perfectly healthy-looking training loop produced zero parameter updates after pruning — and why I think the null result is worth publishing. From building [splat-slim](https://github.com/Daceyyreal/splat-slim), a training-free compressor for 3D Gaussian Splatting.*

The run finished without a single error. The loss curve looked plausible. The checkpoints saved. And the model was, byte for byte, exactly the same as when I started.

This is the story of a bug that doesn't crash — it just quietly wastes your time and, worse, quietly corrupts your conclusions. I'm writing it up because the setup that triggers it is one of the most common moves in model compression, and because I think we under-report this kind of failure.

## The "obvious" recovery step

3D Gaussian Splatting represents a scene as millions of Gaussians. A simple, effective way to shrink a trained model is to **prune**: drop the Gaussians that contribute least — the near-transparent ones below some opacity threshold. Pruning is cheap and removes a large fraction of the primitives, but it costs a little quality, because even low-opacity Gaussians were doing *some* work.

So the textbook next step is to **fine-tune**: let the survivors train for a short while to absorb the work the pruned ones used to do. Prune, then fine-tune to recover. Compression papers do it constantly. I expected a small, free quality bump.

## The symptom: nothing moved

I pruned, kept training, and watched the metrics. PSNR didn't improve. That alone isn't alarming — maybe the gains were just small. What *was* alarming: PSNR didn't change at all. Not up, not down, not jittering the way a live optimization does. It sat flat across the entire fine-tune, identical to three decimal places.

A flat-line metric across thousands of iterations is not "the model converged." It's "the model isn't moving." Those are very different diagnoses, and the second one means something in the loop is broken.

## Ruling things out

I went down the usual list. Learning rate non-zero? Yes. Loss actually computed on the right tensors? Yes. Gradients flowing — were they `None` or all zeros? They were populated and nonzero. The data pipeline was feeding real frames. By every local check, the loop looked like it was training.

So I did the one test that actually settles it: I snapshotted a parameter tensor before the fine-tune and compared it after. **Identical.** The gradients existed, the optimizer's `step()` was being called, and the parameters the renderer actually used never changed by a single bit.

## The cause: the optimizer was tracking ghosts

Here's what was happening. Pruning rebuilds the parameter tensors at the new, smaller Gaussian count — the means, scales, rotations, opacities, and spherical-harmonic coefficients all go from N rows to fewer. But an Adam-style optimizer doesn't just hold a learning rate; it holds **per-parameter state**: a first- and second-moment buffer shaped exactly like each parameter, plus references to the specific parameter objects it's responsible for.

When I pruned the model outside the framework's own densification machinery, those two things fell out of sync. The optimizer was still pointing at the *pre-prune* parameters, and its moment buffers were still shaped for the *old* Gaussian count. The forward and backward passes used the new, pruned tensors and dutifully filled their gradients — but `optimizer.step()` was operating on a set of tensors the renderer no longer used. The updates landed nowhere visible. The dimension mismatch between the stale optimizer state and the pruned parameters was the smoking gun.

In the splatting frameworks, this is a particularly easy trap, because densification and pruning *during* normal training are handled by a dedicated strategy that carefully edits the optimizer state in lockstep — slicing the moment buffers with the same mask it uses on the parameters. Do your pruning through that path and everything stays aligned. Do it as a standalone post-processing step, as is natural when you're compressing an already-trained model, and you silently bypass all of that bookkeeping.

## Why this one is so dangerous

Most bugs announce themselves. This one passes every surface check:

- It doesn't raise — no shape error, because the mismatch is between optimizer state and parameters, not inside the forward pass.
- The loop runs to completion and saves checkpoints.
- Gradients are real, so a `grad is not None` check passes.
- If you only watch a loss that's dominated by other terms, or you don't run long enough to expect visible movement, the flat line looks like "already converged."

The only reliable detector is the blunt one: **assert that your parameters actually changed.** Snapshot a tensor, train, diff it. If you've done any structural surgery on a model — pruning, growing, re-initializing, splitting — also check that your optimizer's parameter groups reference the *same tensor objects* the model now uses, and that its state shapes match the current parameters. Those two assertions would have saved me a day.

## Why publish a failure

It would have been easy to fix this quietly, get my recovery numbers, and never mention it. I didn't, for two reasons.

First, **reproducibility**. Prune-then-fine-tune is everywhere in compression work, and this failure mode is silent. A run that no-ops doesn't look like a broken run — it looks like "fine-tuning didn't help much." If someone reports that fine-tuning gives little recovery, but their fine-tune was secretly a no-op, the conclusion is contaminated and nobody can tell from the outside. Documenting the trap and how to detect it is the kind of thing that lets other people trust — or correctly distrust — a result.

Second, it sharpened a design decision. splat-slim is deliberately **training-free**: it operates on the exported `.ply` as a pure post-processing pass, with no optimizer in the loop at all. Part of the case for that design is exactly this — once a model is trained and you're compressing it, re-entering the training machinery is fragile in ways that have nothing to do with your compression idea. A pass that never touches the optimizer can't desynchronize it.

Negative and null results are real findings. A failure mode that silently produces zero updates, in a step half the field runs, is worth a paragraph in the record — not a deletion from the notebook.

## Takeaways

1. **A flat metric is a diagnosis, not a non-event.** "Didn't improve" and "didn't move at all" are different failures; the second usually means the update isn't landing.
2. **After any structural change to a model, the optimizer is suspect.** Its state is shaped for the old model until you rebuild or reindex it. Assert that parameter-group identities and state shapes match the live parameters.
3. **The cheapest correctness test is a before/after diff of the weights.** If they're identical after "training," nothing else you measured matters.
4. **Report the no-ops.** Silent failures in common pipeline steps are exactly the results that protect everyone else's conclusions.

---

*Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [gsplat](https://github.com/nerfstudio-project/gsplat). If you've hit the same wall from a different angle, I'd like to hear about it.*
