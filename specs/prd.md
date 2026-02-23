# PRD — Lightweight Frame Annotation App

## 1. Overview
Build a minimal local Python desktop app for fast frame-by-frame artifact annotation, replacing Label Studio for day-to-day labeling.

The app must prioritize speed, reliability, and low cognitive load.

---

## 2. Goals
- Fast sequential image annotation.
- Zero-friction workflow (no explicit submit step).
- Simple rectangle-only annotations.
- Manual model iteration loop with explicit **Train** and **Predict** controls.

---

## 3. Users
- Primary: single annotator working locally on ordered frame sequences.

---

## 4. Core Workflow
1. Open image sequence folder.
2. Current frame is displayed.
3. Draw/delete rectangles as needed.
4. Navigate with arrow keys.
5. Changes are saved automatically.
6. Use **Train** when ready.
7. Use **Predict** to prefill boxes, then correct.

No skip flow is needed.

---

## 5. Functional Requirements

### FR-1: Sequence loading
- Load images from a selected directory.
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`.
- Sort in natural filename order.

### FR-2: Viewer
- Display current frame with rectangle overlays.
- Show frame progress and filename.

### FR-3: Navigation
- `Right Arrow`: next frame.
- `Left Arrow`: previous frame.
- Free movement in both directions.
- Navigation implicitly saves current frame data.

### FR-4: Create rectangle
- Click + drag + release creates a rectangle on current frame.
- Store as pixel coordinates: `x`, `y`, `w`, `h`.

### FR-5: Delete rectangle
- Clicking inside a rectangle deletes it.
- If overlap exists, delete deterministic target (topmost / latest added).

### FR-6: Clear frame rectangles
- `Ctrl+Backspace` deletes all rectangles on current frame.

### FR-7: Implicit submit / autosave
- No explicit submit button.
- Data is saved automatically:
  - on add/delete/clear
  - on frame navigation
  - on app close

### FR-8: Persistence
- Save to `annotations.json` in working/project folder.
- Structure:
```json
{
  "frame_000001.jpg": [{"x":120,"y":80,"w":64,"h":92}],
  "frame_000002.jpg": []
}
```

### FR-9: Resume
- On startup, load existing `annotations.json` automatically when present.

### FR-10: Train button
- UI contains a **Train** button.
- Clicking Train:
  1. saves current annotations,
  2. triggers backend training manually,
  3. shows status (`Training...`, success, error).

### FR-11: Predict button
- UI contains a **Predict** button.
- MVP behavior: predict **current frame only**.
- Predicted boxes are rendered immediately and can be edited/deleted.

---

## 6. Non-Functional Requirements
- Local-first, offline-capable for annotation.
- Responsive navigation (target near-instant frame switching).
- Stable autosave with crash-safe writes.
- Python 3.11+ on Linux (NixOS target).

---

## 7. UI Requirements

### Main window
- Image canvas (primary area).
- Top controls: **Train**, **Predict**.
- Status bar includes:
  - frame index / total,
  - filename,
  - rectangle count,
  - backend status message.

### Shortcut hints (visible in UI)
- `←/→`: previous/next frame
- `Drag`: add rectangle
- `Click inside rectangle`: delete rectangle
- `Ctrl+Backspace`: clear all rectangles on current frame

---

## 8. Out of Scope (MVP)
- Multi-class taxonomy.
- Polygon/segmentation tools.
- Collaboration/users/roles.
- Review queues.
- Cloud storage.
- Complex batch prediction management UI.

---

## 9. Backend Integration (MVP)
- Reuse existing local ML backend endpoint.
- Train action: manual trigger endpoint.
- Predict action: per-frame predict endpoint.
- App remains usable for manual annotation even if backend is unavailable.

---

## 10. Acceptance Criteria
1. User can annotate a folder with only keyboard + mouse in one window.
2. Left/right navigation works reliably and never blocks on submit semantics.
3. Rectangle create/delete/clear interactions work exactly as specified.
4. Autosave persists all edits and restores after restart.
5. Train button triggers backend training and shows clear status.
6. Predict button fills current-frame predictions and allows immediate correction.

---

## 11. Milestones
- M1: Viewer + sequence loading + navigation
- M2: Rectangle editing interactions
- M3: Autosave + resume
- M4: Train/Predict backend wiring
- M5: UX polish + packaging/run script
