+++
name = "augmentation.input.intent.self_correction"
description = "Adds one explicit inline correction for a specified argument."
author = "UL"
+++
Add one natural inline false start for the specified argument: first write one plausible temporary value, then immediately and explicitly correct it to the exact original value. Write like a real person typing normally, not a polished benchmark template. For example only, 'transfer 120$ to alice' could become 'transfer 100$, sorry 120$ to alice'. Never copy the example. The original value must be the final active value. Preserve every other request, fact, value, constraint, identifier, relationship, and request order. Add no other context or correction.
