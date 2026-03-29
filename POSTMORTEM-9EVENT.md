# Post-Mortem: 9-Event Model Failure

## What Happened
Attempted to upgrade from 7-event (1B, 2B, 3B, HR, BB, HBP, OUT) to 9-event 
(split OUT into K, BIP_OUT, PROD_OUT). Model produced 500+ runs per team.

## Root Cause
**Hybrid contamination.** The 9-event code was introduced incrementally into a 
7-event pipeline. Multiple functions in the adjustment chain (`apply_park_factors`, 
`apply_platoon_adjustment`, `apply_form_adjustment`, `apply_weather_adjustment`, 
`apply_workload_adjustment`, `apply_umpire_adjustment`, `apply_bullpen_availability`) 
all hardcode `p_out` as the out key. When 9-event vectors (with `p_k`, `p_bip_out`, 
`p_prod_out` but NO `p_out`) passed through these functions, they either:

1. Created a spurious `p_out` key alongside the 9-event keys → sum > 1.0
2. Stripped the 9-event keys entirely → only hit outcomes remained → no outs recorded
3. Both, depending on the function

The bullpen transition was the trigger: starter matchups were 7-event (worked), 
but bullpen matchups were 9-event (broke). Half-innings in bullpen innings never 
recorded 3 outs → infinite PA loops → 500+ runs.

## Lesson
**Never mix event models in the same pipeline.** The upgrade must be atomic:
- ALL probability vectors must be 9-event OR 7-event
- ALL adjustment functions must handle the same format
- ALL normalization must use the same key set
- The switchover must happen in one commit, not incrementally

## Correct Implementation Plan
1. Create a single `EventModel` abstraction that ALL functions use
2. Convert ALL data sources to produce 9-event vectors at ingestion time
3. Convert ALL adjustment functions to work with the abstraction
4. Convert the simulator to consume 9-event
5. Test the FULL pipeline end-to-end before deploying
6. Ship as one atomic branch merge
