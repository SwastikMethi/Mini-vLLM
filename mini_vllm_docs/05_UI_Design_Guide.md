# Mini vLLM Approved UI Design Guide

Version 1.2 | Approved 17 October 2026

Implement the approved amber-on-graphite interface in [the reference image](UI_Request_Workspace.png). It uses a full-width composer, request streams on the left, a physical KV block grid and logical lookup table on the right, and a continuous batching timeline below. This guide supersedes the earlier green theme and sidebar layout.

The image establishes appearance and composition. The [UI specification](03_UI_Specification.md), [TRD](02_TRD.md), and [projection contract](contracts/ui_projection_contract.md) determine runtime behavior. The screenshot is a design concept, not a consistent event fixture. Never infer state transitions, real timing, token counts, or capacity settings from its sample values.

## Visual system

| Role | Token | Value |
| --- | --- | --- |
| Canvas | `--color-bg` | `#131416` |
| Raised surface | `--color-surface` | `#1A1B1E` |
| Divider | `--color-border` | `#303237` |
| Main text | `--color-text` | `#E6E8EB` |
| Secondary text | `--color-text-muted` | `#A3A7AF` |
| Primary accent | `--color-accent` | `#DDB46E` |
| Accent text | `--color-accent-text` | `#E6C48A` |
| Selected surface | `--color-selection-bg` | `#241F16` |
| Other allocated blocks | `--color-allocation` | `#858B95` |
| Error text | `--color-error` | `#E28F8F` |

Use [assets/theme.css](assets/theme.css) as the shared token source. Amber identifies the selected request, linked memory blocks, active navigation, primary actions and streamed terminal text. Other requests remain neutral and identifiable by their IDs. Status labels, hatching and explicit occupancy counts make the UI understandable without color. Error color is reserved for actual failures.

Use flat surfaces, 1 px dividers and 0 to 3 px corner radii. Avoid gradients, glow, large rounded buttons, floating metric cards, decorative charts and a permanent navigation sidebar. Keep four spacing steps: 4, 8, 12 and 16 px; use 24 px only between major regions.

Use a neutral sans-serif for section titles and readable prompts, with a monospace font for IDs, output, commands, counters and timestamps. A suitable locally hosted pairing is Inter and IBM Plex Mono; the CSS includes system fallbacks. Body text is 14 px, technical rows 13 px, secondary labels at least 12 px, and section titles 18 px. Use tabular numerals for counts. Essential text must meet at least 4.5:1 contrast; interactive indicators and focus rings must remain clearly visible.

## Desktop composition

| Region | Suggested size | Behavior |
| --- | --- | --- |
| Application bar | 48 to 54 px high | Product name, Live tab, functional secondary navigation, model identity |
| Composer | 72 to 96 px high | Prompt field, output cap and compact Send button; expands for multiline input |
| Status strip | 40 to 44 px high | Active and waiting counts, viewing mode, Pause view and Step |
| Request streams | About 54% of main width | Dense rows with separators; selected row has a thin amber edge |
| KV cache | About 46% of main width | Physical-ID grid above the selected request's logical block table |
| Timeline | 180 to 240 px high | Common time cursor, engine lane and request lanes; locally resizable |
| Footer | 24 to 28 px high | Connection state, configured block size and keyboard hints |

Treat dimensions as responsive targets, not a fixed screenshot canvas. The reference uses a 24-block sample pool; the grid must render the configured backend pool. Use six columns only when cells remain legible. Reduce columns as needed and virtualize large pools. The main panes can resize, with minimum widths around 440 px for streams and 400 px for memory.

Live is the default route. Expose Traces when replay/import is implemented and Benchmarks when it has functional content. Do not ship empty navigation merely to match the image.

## Request streams

Each row contains a stable request ID, explicit phase, prompt excerpt, streaming output excerpt, cached input count and owned physical page IDs. Keep numeric columns aligned. Truncate long excerpts visually and reveal complete content in the inspector; never discard the underlying output. Page chips wrap or collapse into a labelled overflow control.

Selecting a row sets `selected_request_id` for every view. Show a subtle amber background and 2 px left edge. Expand the selected lifecycle inline, including accepted/tokenized status when relevant. The larger output, raw events, timings, stop reason and Cancel action appear on demand in a detail drawer. Do not duplicate a permanent output panel below every stream.

Finished rows become quieter and explicitly say that pages were released. Preserve history until the user filters or clears the view. Waiting rows show their actual reason and own no blocks unless the backend explicitly reports otherwise. A decode lifecycle phase can remain active while execution waits between engine steps.

## Cache inspection

Keep physical blocks in physical-ID order. Each tile shows its ID, owner ID or free state, and committed occupancy. Amber outlines identify blocks owned by the selected request; other allocated blocks use gray. A hatched owned-but-empty state differs from a free block. Normalized occupancy bars in compact mode summarize utilization and must not imply that each segment equals one token.

On selection, show the exact `committed / block_size` count. Expanded slot mode renders the configured number of slots and distinguishes valid, reserved-but-uncommitted, and free states. A token slot represents cached K/V for one processed input position across model layers. Slots commit on forward completion; sampling a new output does not fill a new slot immediately.

The lookup table contains logical block, logical token range, physical block and committed slots. Clicking a row highlights its physical tile and exposes the exact mapping. Changing request selection updates this table; clicking a free block does not invent an owner. Keep capacity credits separate from owned physical blocks in the capacity detail view.

## Timeline and presentation controls

Use real elapsed time by default, with an engine lane and aligned request lanes. Prefill, decode and waiting have text labels and distinguishable shapes. Actual batch membership and duration come from journal events. Short intervals can use a visible marker and tooltip without claiming inflated duration.

An optional scheduler-step view can use equal-width ordinal cells, clearly labelled as steps rather than elapsed time. The screenshot uses this simplified presentation. It must not be treated as evidence of simultaneous prefill and decode. In R0, a new prefill enters between serial engine calls while older requests retain their decode lifecycle state and cache.

Pause view freezes the displayed cursor while backend work continues. Mark the view as paused/historical and show the distance behind live. Step advances one recorded event at that cursor; disable it at the newest available event. Resume or Return to live updates every pane to one coherent state. Recorded playback speed changes presentation only. Cancellation is a separate explicit backend action.

Every phase label, occupancy counter, output excerpt and timeline cursor must agree at the selected sequence number. If a request has already completed prefill and decoding at that cursor, it cannot still display Prefill as its current phase. A detail selection alone must never submit work or pause the engine.

## Responsive and accessible behavior

- At 1280 px and wider, show both main panes and the lower timeline.
- At 960 to 1279 px, rebalance pane widths, reduce grid columns and collapse secondary controls into a menu.
- Below 960 px, stack panels or expose Requests, Cache and Timeline tabs while keeping selection and cursor shared. Allow multiline composer controls and local horizontal timeline scrolling.
- Provide Ctrl/Cmd+Enter to submit, arrow keys to move between request rows, Enter to inspect and Escape to close details. Space may pause the view only when focus is outside editable fields and interactive controls that already use Space.
- Use semantic buttons, labels and tables, visible amber focus rings, sensible tab order and touch targets of at least 44 px on touch layouts. Announce lifecycle changes without announcing every token.
- Honor reduced motion. Fast requests must remain inspectable through replay even if intermediate states never appear during live painting.

## Required nonhappy states

| State | Required presentation |
| --- | --- |
| No requests | Short prompt invitation; empty cache or actual backend pool; no fabricated activity |
| Model loading | Loading status and disabled Send until the engine is ready |
| Connection lost | Stale status, last synchronized cursor and reconnect action; do not imply fresh data |
| Capacity wait | Explicit queue reason; inspectable owned blocks and reserved growth credits |
| Cancellation pending | Keep request visible until engine cleanup and terminal acknowledgement |
| Request failure | Preserve partial output, error details, stop reason and cleanup outcome |
| Unsupported measurement | Show unavailable rather than an estimated GPU duration |
| Historical selection | Persistent replay/historical badge and Return to live action |

## Implementation and acceptance

The existing T09 task covers the ordered reducer and shared selection. T10 should compose `AppShell`, `RequestComposer`, `RequestStreams`, `RequestRow`, `KVBlockGrid`, `BlockTable`, `BatchTimeline`, `PresentationControls` and `DetailDrawer` around that reducer. Keep appearance in shared tokens; do not copy live state into independent panel stores. This is a UI revision and does not add engine features or change the existing release milestones.

1. Compare the desktop result to the approved image for structure, hierarchy, amber selection and control density.
2. Select any request and verify that page IDs, owner labels, lookup rows and timeline selection agree.
3. Replay the existing multi-request fixture and confirm truthful prefill pauses, shared decode batches, slot commits and cleanup.
4. Pause presentation while generation continues, then step and resume without losing or fabricating events.
5. Verify long prompts, many blocks, empty state, reconnect, failure, keyboard operation and narrow layouts.
6. Check that no old green palette, large sidebar composer or stacked dashboard-card layout remains as the default.

## Design references

These sources informed the visual direction; they are not runtime dependencies.

- [Oxide design system](https://github.com/oxidecomputer/design-system/blob/master/design.md): restrained technical styling, neutral surfaces, compact controls and semantic color.
- [Linear UI redesign](https://linear.app/now/how-we-redesigned-the-linear-ui): alignment, density and visual hierarchy.
- [Perfetto UI](https://perfetto.dev/docs/visualization/perfetto-ui): selectable event lanes and details tied to a shared timeline.

The final amber palette is the user's approved variation. It replaces the earlier green concept.
