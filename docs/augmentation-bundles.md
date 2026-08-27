# Plan named augmentation bundles

Bundles let a customer choose a business-risk question without assembling raw augmentation IDs.
Each bundle is a versioned operator-selection and budget policy. It contains no generation prompt.

Discover the built-in policies:

```bash
ul augmentations bundles list
ul augmentations bundles show everyday-customers
```

Preview independent probes for opaque source case IDs:

```bash
ul augmentations bundles plan everyday-customers \
  --case checkout-17 \
  --case checkout-18 \
  --source-feature "production interaction"
```

When a UL project is configured, omit `--case` to use the configured dataset and case limit. The
preview makes no model, target, or network calls. Human output and `--json` both show:

- applicability plus an explicit blocked or skip reason;
- the exact controlled write surfaces and expected relation;
- required response, state, evaluator, and human-review evidence;
- maximum model calls, target calls, duration, authorized cost, mutation risk, and reset needs;
- a stable canonical probe identity.

Every planned probe starts from its named source case. Equivalent selections are deduplicated.
Chained composition is deliberately unavailable. Planning fails closed when a case, fan-out, call,
duration, authorized-cost, or mutation limit would exceed the bundle's hard budget.

The cost value is the bundle policy's maximum authorized amount, not a provider-price estimate.
Inspect the selected version before relying on it, and use a trusted price-aware execution ledger to
enforce actual spend.
