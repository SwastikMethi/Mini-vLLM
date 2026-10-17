# UI projection contract

Documentation version 1.1. Engine event envelope version 1 is unchanged.

The three operational views read one normalized state, built from the ordered journal. They must share a run, selected request, and observation cursor. The included traces are illustrative fixtures, not measured inference results.

## Request state

| Field | Source and meaning |
| --- | --- |
| `status` | Accepted, tokenizing, queued, prefill, decoding, completed, cancelled, or failed |
| `accepted_ns` | Timestamp of `request.accepted` |
| `queued_ns` | First `request.queued`; never reset by queue-position updates |
| `admitted_ns` | Timestamp of `request.admitted`; null before admission |
| `wait_reason`, `queue_position` | Latest `request.queued` payload while queued; clear on admission or terminal |
| `executing_batch_id` | Current started batch containing the request; null between forwards |
| `prompt_tokens` | `request.tokenized.prompt_tokens` |
| `processed_tokens` | Whole-model committed input positions; historical after cleanup |
| `cached_tokens` | Committed K/V retained now; zero after release |
| `generated_tokens` | Number of sampled outputs; sampling does not itself fill a KV slot |
| `decode_step` | Last committed decode step; display a started step separately until commit |
| `block_table` | Ordered logical block index to physical block ID mapping |
| `reservation_blocks` | Admission credits, independent of owned physical block count |
| `sampled_ns` | Ordered sampled-token timestamps, indexed by output index |
| `terminal_ns`, `stop_reason` | Terminal timestamp and reason, null while active |

Queue updates reuse `request.queued` only while status is queued. Emit an update when reason or position changes; do not re-emit unchanged state each loop. UI queue age uses the first queue entry. Server queue latency remains queued-to-first-prefill-started as specified in the TRD.

The lifecycle `decoding` state does not mean a forward is currently executing. During B prefill, A can remain decoding with `executing_batch_id = null`. Do not show A reading memory at that moment.

## Memory projection

Use `engine.ready.block_size` and `pool_blocks`. A physical block ID refers to the same token positions across all model layers. The map describes the preallocated KV pool; it is not a map of all device allocations.

For logical block `j` of a request:

```python
physical_id = request.block_table[j]
first_position = j * block_size
valid_slots = min(block_size, max(0, request.cached_tokens - first_position))

# Address of a logical token position p:
logical_block, offset = divmod(p, block_size)
physical_id = request.block_table[logical_block]
```

`cache.allocated` appends mappings and changes ownership. It does not mark slots committed. `request.prefill.completed` and `request.decode.completed` advance committed inputs; `token.sampled` changes only output. `cache.released` clears ownership, valid slots, and admission credits for that request. Keep processed counts as history.

In R0, display `owned_blocks`, `free_blocks`, `unused_owned_slots`, and `reserved_capacity_blocks` separately. The pool stays allocated when a request releases ownership. At a committed state a partially filled final block can have up to `block_size - 1` unused slots; a just-allocated block can temporarily have zero committed slots. R1 shared blocks require reference counts and different accounting before this projection can be reused.

## Batch projection

| Field | Source and meaning |
| --- | --- |
| `batch_id`, `phase`, `members` | `batch.scheduled`; each member has request ID, input count, and starting logical position |
| `scheduled_ns` | `batch.scheduled.monotonic_ns` |
| `started_ns`, `start_seq` | `batch.started` time and event sequence; null before start |
| `completed_ns`, `end_seq` | `batch.completed` or `batch.failed`; null while ongoing |
| `status` | Scheduled, active, completed, or failed |
| `host_duration_ms` | Completed event measurement; no estimate for missing data |
| `gpu_duration_ms` | Device-event measurement or null |

For each member, highlight positions from `start_position` through `start_position + input_tokens - 1`. During execution show these as planned writes; after that request's completion event show them as committed. A shared batch is one engine interval even if it is drawn on several request lanes.

R0 engine forward intervals must not overlap. A new prefill may occupy the gap between two decode calls of an older request. It does not interrupt a running kernel and does not bypass FIFO admission among waiting requests. The timeline does not promise simultaneous prefill/decode. R2 mixed batches require a versioned event contract with explicit per-member phase and chunk ranges.

Use timestamps relative to the run's first event, not browser wall time. An ongoing segment extends to the last observed cursor with an ongoing label; its final endpoint is unknown. Time-axis zoom preserves duration ratios. Missing history is explicitly unavailable.

## Snapshot and replay

`GET /api/runs/{run_id}/snapshot` must contain the request and batch fields above, physical ownership, the model/run manifest, and `high_watermark_seq`. Its current OpenAPI schema permits these projection fields. Snapshot history must retain batch intervals and members; durations alone cannot reconstruct a timeline.

Subscribe after the watermark. Ignore duplicate sequence numbers; resynchronize on gaps. The global stream is ordered before filtering by request. All three panels apply each accepted event in the same reducer transaction. Rendering can be batched to frames without discarding events.

Clicking a historical batch switches every panel to a shared replay cursor. Never combine current memory ownership with old timeline state. A block ID can be reused, so historic selections are identified by `(run_id, seq, physical_block)` rather than a physical ID alone. A **Return to live** action restores the current reducer state. Replay changes only playback speed, not recorded timings.

## Illustrative fixtures

- [Single request trace](event_examples.jsonl) and [terminal snapshot](snapshot_example.json): 15 prompt tokens, block-boundary growth, and cleanup.
- [Continuous batching trace](continuous_batching_example.jsonl): A starts first, B prefills between A decode steps, both share a later batch, and C waits for growth credits.
- [69 ms checkpoint](continuous_batching_checkpoint.json): the state represented by [the legacy contract diagram](legacy_contract_wireframe.png), not the approved visual mockup.
- [Continuous batching terminal snapshot](continuous_batching_snapshot.json): all three requests completed and all blocks released.

At the checkpoint, A has 17 cached inputs in P3 and P1; B has 5 in P0. P2 is free. Three of four admission credits are reserved; C requires two, so C is queued for capacity. These values are intentionally small to expose the scheduling behavior. The default application can use a larger pool.

## Acceptance checks

- Token sampling alone never increments committed slots.
- A newcomer can be prefilling while an older request remains in its decode lifecycle phase; the old request's execution lane is idle during that prefill.
- Batch membership, input positions, and all block mappings agree with the same event cursor.
- A queue update retains the first queued timestamp.
- Completed batches never overlap in the single engine lane.
- Reconnecting and replaying reproduce the checkpoint and terminal snapshots.
- A physical block reused by another request does not change its historical owner in replay.
