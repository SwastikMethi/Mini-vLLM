# Mini-vLLM implementation package

Package version 1.2 · 17 October 2026

Build a Python inference server with real paged KV storage and continuous batching. The approved UI uses amber on graphite, request streams beside the physical cache map, and a shared batching timeline below. Selection links lifecycle, logical-to-physical mapping and batch membership.

## Read in this order

1. [Product requirements](01_PRD.md) — goals, scope, user journeys, and release gates.
2. [Technical requirements](02_TRD.md) — Python architecture, cache and scheduler invariants, model integration, API contracts, and validation.
3. [UI specification](03_UI_Specification.md) — live request flow, memory and batch views, replay, visual direction, and accessibility.
4. [Implementation plan](04_Implementation_Plan.md) — staged tasks, dependencies, estimates, and the release test matrix.

The four core documents are available in Markdown, Word, and plain text. The UI specification is version 1.2; the PRD, TRD and implementation plan remain version 1.1 because their engine scope is unchanged. The UI specification and design guide govern final appearance. The Markdown files preserve the document content with formatted headings, lists, and tables.

## Supporting files

- [Approved amber UI](UI_Request_Workspace.png) — final visual direction; illustrative data, not a live screenshot.
- [UI design guide](05_UI_Design_Guide.md) — layout, components, interaction, responsiveness and visual acceptance.
- [Theme tokens](assets/theme.css) — amber and graphite colors plus shared geometry.
- [Changes in this package](CHANGELOG.md) — version 1.2 update details.
- [Event schema](contracts/event_schema.json) — typed event payload contract.
- [Example event trace](contracts/event_examples.jsonl) — illustrative ordered request trace.
- [Expected snapshot](contracts/snapshot_example.json) — terminal state corresponding to the example trace.
- [API contract](contracts/api_contract.yaml) — proposed HTTP interface.

## Updated UI contracts and examples

- [UI projection contract](contracts/ui_projection_contract.md) — fields and event rules for the three views, queue updates, selection, and snapshot recovery.
- [Continuous batching trace](contracts/continuous_batching_example.jsonl) — three requests, 76 illustrative engine events, newcomer admission, a shared decode batch, capacity waiting, and cleanup.
- [Contract checkpoint](contracts/continuous_batching_checkpoint.json) — a 69 ms event fixture.
- [Legacy contract diagram](contracts/legacy_contract_wireframe.png) — visualizes that fixture only; superseded as a UI design.
- [Continuous batching terminal snapshot](contracts/continuous_batching_snapshot.json) — all three requests finished and all blocks released.

The package is version 1.2; the event envelope remains version 1. Slot counts advance on committed forwards. A sampled output is cached on its next forward. New prefills enter between decode calls; the first engine implementation executes these calls serially.

## Implementation scope

The two-day core milestone is narrower than the complete release, estimated at five to seven focused developer days. Prefix reuse and preemption are later extensions.

This package contains plans and specifications; it does not contain an implemented inference server. Fixtures demonstrate contracts and are not benchmark results. The live UI must derive state from actual engine events.
