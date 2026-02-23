## Review
- What's correct
  - The implementation aligns with the annotation-app plan: folder loading, frame navigation, rectangle create/delete/clear, autosave persistence, resume support, and Train/Predict controls are present.
  - `label_studio/artifact_detection_config.xml` keeps a single primary rectangle label (`body_distortion`) while preserving optional compatibility choices.
  - `ml_backend/grounding_dino_backend/model.py` adds a coherent sequence-learning train/predict flow with environment toggles and bounded interpolation.

- Fixed: Issue and resolution
  - Fixed: Non-thread-safe Qt UI update in `artifact_detector/ui/main_window.py` backend health check.
    - Resolution: Added a Qt signal (`backend_status_changed`) and emitted from the background thread; UI label updates now happen on the Qt main thread.
  - Fixed: Frame sequence ordering did not satisfy natural sort requirement (`frame_2` vs `frame_10`) in `artifact_detector/core/loader.py`.
    - Resolution: Added `_natural_sort_key()` and switched folder ordering to natural sorting.
  - Fixed: Mouse-to-image coordinate cache could be stale after geometry changes in `artifact_detector/ui/frame_viewer.py`.
    - Resolution: Recompute layout inside `_widget_to_image()` before coordinate conversion.
  - Fixed: Label Studio event hook robustness in `ml_backend/grounding_dino_backend/model.py`.
    - Resolution: Made `additional_params` optional in `process_event()`, broadened project-id extraction to handle dict/int payloads, and made task-fetch parsing tolerate both list and paginated dict API responses.

- Note: Observations
  - `model.py` now mixes zero-shot inference and sequence-trained interpolation cleanly, but the file has become large; splitting sequence-learning and Label Studio API glue into helpers/modules would improve maintainability.
  - The desktop app currently focuses on folder-based image sequences (matching MVP workflow); video-loading paths have intentionally been removed from the main UI flow.
