# Implementation Plan: Anime Character Clips

## 1. Goal

Given anime episodes and reference images for a named character, find frames showing that character's head and export short, watchable video clips containing those appearances at the source video's dimensions, with audio when present. The head is the only required visible body part. Reference embedding and hair color determine identity; eye color is not used. Every frame included in a clip must pass the character-presence check. Cut the clip when the character is no longer detected; start a new clip if they return.

Proposed folder layout:

```text
anime/
  hunterxhunter/
    episodes/
      episode-01.mkv
      episode-02.mp4
    reference/
      killua/
        killua1.png
        killua2.png
    results/
      killua/
        episode-01/
          clip-001.mp4
          matches.csv
```

Adding another character means adding another folder under the series' `reference/`. Adding another anime means adding another folder under `anime/`. Episodes are shared among characters of the same anime.

The `hunterxhunter` folders, both Killua reference images, and one MP4 episode are present. Clip filenames above illustrate future output.

## 2. Non-Goals

- Body-only recognition when the head is absent.
- A web upload interface or automatic downloading of episodes.
- Artificial upscaling beyond the episode's source resolution.
- Intro/outro padding that includes frames where the character is absent.

## 3. Constraints

- Keep implementation simple; reuse existing anime detector, encoder, and frame-reading/batching code.
- Avoid unnecessary duplication, files, dependencies, and abstractions.
- Preserve existing `video_to_faces` behavior and security; do not modify unrelated code.
- Do not commit or push.

## 4. Current Understanding

- `video_to_faces` already detects anime faces and classifies saved crops against reference images. It does not retain timestamps for clip creation.
- The existing duplicate filters remove repeated appearances, so clip detection must use matches before deduplication. Its default one-second sampling is insufficient for frame-accurate clip boundaries.
- Reference folders can contain multiple images, but the current classification path encodes only the first one. The new workflow should consider every reference image without changing existing classification behavior.
- `killua1.png` is a close head portrait; `killua2.png` includes the upper body, a turned head, and one closed eye. Both have dark backgrounds. Reference processing must isolate the head and keep the hair rather than encode the background or clothing as identity cues.
- The references show light silver/white hair. Hair color is derived from each reference, not hard-coded for other characters. Eye color is excluded from matching by user direction.
- Episodes may zoom in and out. Head size will vary; matching must use size-normalized head regions and not rely on a fixed pixel size.
- The existing CLI has no clip-export mode. `ffmpeg` is available in the current environment, but should be checked when the feature runs.
- User confirmed the output should be short clips containing the character.

Unknown items:

- Exact match threshold that works best across anime styles; expose it as an option and verify with sample footage.
- How reliably the existing anime face detector finds a head when little or none of the face is visible. Validate this with real episode frames before deciding whether a small additional head-candidate method is needed; never label an unverified head as Killua solely from hair color.
- How best to extract hair color from varied drawings without including backgrounds, highlights, or clothing. Test the smallest approach using existing image tools before adding a dependency.

## 5. Areas to Inspect

- `src/videotofaces/detection.py`: frame sampling, filtering, and face crops.
- `src/videotofaces/grouping.py`: anime encoder and cosine-distance matching.
- `src/videotofaces/prep.py`, `main.py`, and `__main__.py`: input validation and CLI conventions.
- `tests/` and `README.md`: test and usage patterns.

## 6. Simplest Implementation Strategy

Add one focused character-clips workflow callable from the CLI and Python. Read episodes from `anime/<series>/episodes/` and references from `anime/<series>/reference/<character>/`. Crop references and candidates around the head, including hair, then classify every source frame. Confirm identity from reference-derived hair color and the existing anime embedding, with hair weighted more strongly; do not assess eyes. Use head-relative regions so zoom changes do not change the matching rule; if the head or hair cannot be verified at a frame's scale/angle, classify it as uncertain rather than assuming presence. Consecutive confirmed frames form one appearance; the first absent or uncertain frame ends it, and a later match starts another. Use `ffmpeg` to cut full-frame clips at those frame boundaries into `anime/<series>/results/<character>/<episode>/`. Keep dimensions unchanged and use a quality setting suitable for viewing. Emit a per-frame match report, including hair and embedding scores and uncertainty, so clips can be audited. This gives frame-level enforcement, though model errors mean absolute visual certainty still requires human review.

Reuse opportunities:

- Existing anime face detector, image encoder, box filtering, and frame-reading path.
- Existing image-extension and device-selection conventions.

Avoid creating:

- A second embedding implementation, a database, a web service, or a new video editing library. Add a head-specific candidate method only if the existing detector cannot satisfy tested head-only cases.

## 7. Security Considerations

- Validate series and character names as single folder components and keep output paths inside that series' `results/`; reject path traversal and unsafe symlinks.
- Invoke `ffmpeg` with an argument list, not a shell command; treat episode and character names as data.
- Accept only readable video/image files and give clear errors for invalid media; never delete input episodes or references.

## 8. Implementation Phases

### Phase 1: Find Character Appearances

**Status:** Review Failed — crop and matching criteria under revision; model-backed verification pending

**Depends on:** None

#### Goal

Produce per-episode match timestamps and scores for a named character from the proposed `anime/` layout.

#### Scope

- Add the focused workflow entry point and CLI option(s) for series folder and character name.
- Validate folder structure and reference images; support multiple references for one character, including portraits and wider poses.
- Isolate head regions from references and frames; match with reference-derived hair color as the priority cue and existing embeddings as supporting evidence. Do not extract or compare eye color.
- Reuse anime detection and encoding to record match/non-match status and episode/frame timestamps for every source frame, before duplicate filtering.
- Write a per-episode `matches.csv` under the character's results folder.

#### Do Not Change

- Existing face-image extraction, clustering, classification, or live-action behavior.
- Video encoding or clip creation; that belongs to Phase 2.

#### Reuse First

- Inspect `get_detector_model`, `get_encoder_model`, box filtering/cropping, and video-reading code before adding any new path.

#### Steps

1. Confirm the smallest way to expose head boxes/crops and timestamps from the existing detection path without altering its outputs; assess partial/profile head coverage.
2. Validate the input layout; crop `killua1.png` and `killua2.png` to head regions that retain hair while excluding dark backgrounds and clothing, then encode all valid references.
3. Derive hair color from the references with existing image tools; combine it with embedding similarity so hair has priority without using color alone as proof of identity.
4. Match each episode frame across head sizes and angles, marking absent/uncertain frames conservatively, and record timestamps, cue scores, and decisions.
5. Add focused tests for matching, zoom changes, eye appearance having no effect, multiple references, empty/invalid inputs, and safe paths.
6. Run relevant tests and the security check.

#### Acceptance Criteria

- A run for `hunterxhunter/killua` produces a timestamped per-frame report per processed episode, including non-matches, and clearly reports episodes with no matches.
- Multiple reference images can contribute to a match.
- The wider `killua2.png` is evaluated from its head, not from its clothing or black background; eye appearance does not affect matching.
- Hair color influences the match more than the generic embedding, and the report exposes both scores without an eye-color field. Head size changes due to zoom do not change the comparison regions.
- Frames with no verifiable character head are marked absent or uncertain and cannot enter a clip.
- Existing `video_to_faces` outputs and CLI remain functional.
- Invalid character names cannot escape the intended folders.

#### Verification

- Run focused unit tests with mocked detector/encoder and a tiny generated video; verify consecutive frame indexes, timestamps, and no-match behavior.
- Test both supplied Killua images for head cropping and hair-color extraction; include resized variants to cover zoom in/out and eye-appearance independence.
- Run existing affected tests where locally feasible; model-weight downloads are not required for unit tests.

#### Security Check

- Inspect all input/output path handling and confirm no untrusted string enters a shell.

#### Recovery Subphases

Complete and review these in order. Phase 1 can pass review only after both subphases pass; Phase 2 still depends on Phase 1.

##### Phase 1.1: Correct Crop and Remove Eye Cue

**Status:** Done — Passed Review
**Depends on:** None

###### Goal

Keep the corrected encoder crops and remove unreliable eye-color matching from the existing batched workflow.

###### Scope

- Feed the existing anime encoder a face-bounded crop from each reference and candidate while retaining the expanded head crop for the hair cue. Check the supplied images to ensure the encoder crop excludes clothing and substantially reduces background; use a small local crop adjustment only if the face box is insufficient.
- Remove eye-region extraction, eye thresholds, eye comparison, and the eye report column. Confirm only when both reference embedding and hair color meet their thresholds, with hair weighted more heavily in candidate ranking.
- Keep one detector pass and at most one encoder pass per batch. Add no model, dependency, second video pass, or image-segmentation pipeline.

###### Do Not Change

- Existing `video_to_faces` behavior, input layout, unrelated CLI options and report columns, or Phase 2 clip logic.

###### Reuse First

- Reuse the face and expanded-head boxes already returned by `_face_heads`, `_appearance`, and the existing anime encoder.

###### Steps

1. Change only the encoder input crops in reference loading and frame reporting; retain head-relative hair sampling.
2. Remove eye-color computation and its CLI/report fields while retaining hair and embedding thresholds.
3. Add focused regression tests showing that eye appearance cannot change a match when hair and embedding agree; cover both encoder crop paths and zoomed images. Keep tests runnable without local reference images or model weights.
4. Run focused tests, syntax checks, `git diff --check`, and the security check.

###### Acceptance Criteria

- A frame with matching hair and reference embedding can be confirmed regardless of whether eyes are open, closed, hidden, or drawn in another color.
- Both reference and frame embeddings receive the smaller face-bounded crop; hair still comes from the expanded head region.
- On both supplied references, the encoder crop excludes clothing and has less background than the current expanded-head crop.
- Batch call counts are unchanged; `--eye-threshold` and `eye_distance` are removed, and no additional model or media pass is introduced.

###### Verification

- Run `python3 -m unittest tests/test_character.py`, syntax compilation, and `git diff --check`.
- Inspect crop content on both supplied references when available; synthetic fixtures must cover the same code path in a clean checkout.

###### Security Check

- Confirm the crop and cue changes add no file access, shell execution, or untrusted model input path.

###### Earlier Execution Notes

- The encoder now receives the same face-bounded crop for references and episode frames, trimmed at the lower edge to avoid neck/clothing; expanded head crops still supply the hair cue. On the supplied references, the encoder crop's near-black fraction is 8.2% and 4.6%, down from 35.6% and 35.9% in the expanded crops.
- Eye pixels must differ in chroma from nearby face skin; a closed or uncheckable eye supplies no eye cue. Nine focused tests, syntax compilation, and `git diff --check` pass. Batch call counts and the report schema are unchanged. No new file, shell, or model path was added.
- Actual detector and encoder output on the episode remains for Phase 1.2; the current environment lacks OpenCV, Torch, and model weights.

###### Earlier Review Notes

- Nine focused tests, syntax compilation, and `git diff --check` pass, and the reference/frame crop paths and batch call counts match the subphase scope. No new security issue was found.
- A synthetic closed eye with a violet eyelid line (BGR `[40, 35, 85]`) yields an eye cue even though no iris is visible. Against an otherwise matching reference, this produces eye distance `0.316` and status `uncertain`. The chroma-versus-skin filter alone cannot distinguish an iris from colored eyelid/shadow pixels, so the unavailable-eye criterion remains unmet. Add a focused regression case and require stronger eye evidence before Phase 1.1 can pass.

###### Current Execution Notes

- Following the revised user requirement, removed eye extraction, `--eye-threshold`, and `eye_distance` from the workflow. Matching now requires reference embedding and hair color, with hair weighted 70% when ranking candidates. Existing face-bounded encoder crops and expanded-head hair sampling remain in place.
- Nine focused tests pass, including closed, colored, and differently colored eye appearances yielding the same hair/reference decision; syntax compilation and `git diff --check` pass. No extra model or media pass, file access, or shell execution was added.
- Model-backed episode validation remains in Phase 1.2.

###### Current Review Notes

- Passed: all nine focused tests, syntax compilation, CLI help, and `git diff --check` pass. The report and CLI contain no eye-color field or threshold, and matching requires both hair and embedding thresholds. The supplied reference crops reduce near-black pixels from 35.6% to 8.2% and from 35.9% to 4.6%; the focused tests confirm the same crop path for reference and frame encoding with unchanged batch call counts.
- No new security issue or unnecessary model, dependency, or media pass was found. Real-model episode accuracy remains outside this subphase and belongs to Phase 1.2.

##### Phase 1.2: Validate Episode Matching

**Status:** Blocked — anime ViT-B16 checkpoint unavailable from its configured Google Drive link
**Depends on:** Phase 1.1

###### Goal

Verify that the corrected matcher produces credible per-frame reports with the actual anime models and episode.

###### Scope

- Use the project's existing OpenCV, Torch, torchvision, detector weights, and anime encoder weights; do not add a dependency or model.
- Run the CLI against the supplied references and episode, inspect reference detection and selected report frames across zoomed, profile, absent, and return cases where present; eye appearance must not affect scores.
- Adjust existing thresholds only if the observed report supports it. Record any head angles the face detector cannot verify as uncertain or absent.

###### Do Not Change

- Phase 2 clip export, original extraction/classification behavior, or model architecture.

###### Reuse First

- Reuse the Phase 1 report, existing model loaders, and current CLI options.

###### Steps

1. Check the installed runtime dependencies and existing model weights; use the project's normal setup if they are missing.
2. Run a small model-backed sample first, then the supplied episode; verify frame indexes, timestamps, cue scores, and no-match reporting.
3. Review selected frames against the report, make only evidence-backed threshold changes within Phase 1, and rerun affected checks.
4. Confirm the existing CLI still works and run the security check.

###### Acceptance Criteria

- Both supplied reference images are detected and encoded from their intended subject regions.
- The supplied episode produces a per-frame `matches.csv` with consecutive frame indexes, timestamps, cue scores, and confirmed/uncertain/absent decisions; episodes with no matches are reported clearly.
- Spot checks show eye appearance has no effect, hair and embedding remain discriminating, and frames without a verifiable head are not confirmed.
- The original CLI remains functional and no additional model or video pass was added.

###### Verification

- Run the Phase 1 focused tests and a model-backed CLI sample, then inspect the episode report against selected source frames.
- If the existing runtime dependencies or weights cannot be obtained, record the exact blocker rather than marking Phase 1 complete.

###### Security Check

- Recheck input/output path containment and confirm no shell command is built from series, character, or episode names.

###### Execution Notes and Blocker

- Removed the unused JIT-loading branch and made the active shared loader pass `weights_only=True` explicitly. It fails closed when that option is unsupported. A mocked loader check confirmed the safe argument, state-dict mapping, and fail-closed behavior. The real anime detector checkpoint also loaded successfully through this path.
- Installed OpenCV, Torch, and torchvision in a temporary runtime outside the repository. Both supplied Killua references produced high-confidence detections with the actual detector. Nine focused character tests, the original CLI help, and the whitespace check pass.
- The configured anime ViT-B16 checkpoint ID `1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC` is unavailable: the normal downloader cannot obtain it, and the Google Drive file page and direct download endpoint return HTTP 404. No encoder checkpoint was written. Model-backed encoding and the episode report therefore cannot run. Resume Phase 1.2 when the same compatible checkpoint is available from a trusted source; do not substitute a different model or weaken restricted loading.

#### Earlier Execution Notes

- Added `src/videotofaces/character.py` with a Python entry point and `python -m videotofaces.character` CLI. It writes one per-frame `matches.csv` for each episode and reports when an episode has no confirmed matches.
- Reused the project's anime ViT encoder and narrowed the eye sample to the visible iris area. Eight focused tests pass, including generated reference loading that works without the ignored local images, optional checks of both supplied references, resized heads, color-priority decisions, safe paths, invalid input, and a leave/return frame sequence. Syntax compilation and `git diff --check` pass.
- The current Python environment lacks OpenCV and Torch, and no model weights are present. CLI execution and real-model recognition on the episode remain unverified; partial/back-of-head coverage and thresholds may require adjustment after a model-backed run.

#### Earlier Review Notes

- The five focused tests pass locally, but `tests/test_character.py` reads `killua1.png` and `killua2.png`. Both are ignored by `.gitignore` and absent from `git ls-files`, so the test suite is not reproducible from a clean checkout.
- `PYTHONPATH=src python3 -m videotofaces.character --help` fails because OpenCV is missing; Torch and model weights are also absent. Reference detection and a full per-frame report for the supplied episode were not validated.
- No security issue was found in the Phase 1 changes. Recheck the actual head/face detector's coverage of the supplied references and profile/back-of-head frames before another review.
- Latest review: eight focused tests and `git diff --check` pass. A synthetic closed-eye frame with matching hair and embedding yields a skin-colored eye cue and is marked `uncertain`, contrary to the unavailable-eye requirement. The tested head rectangles for both supplied references contain about 36% near-black pixels, which are passed into the encoder, contrary to the background-exclusion requirement. A real-model episode report is still unverified because OpenCV, Torch, and model weights are unavailable here. No new security issue was found.

### Phase 2: Export Clips and Document Use

**Status:** Not Started

**Depends on:** Phase 1

#### Goal

Turn match timestamps into short playable clips, organized by character and episode.

#### Scope

- Split appearances at every absent or uncertain frame, with no padding or missed-frame tolerance, and clamp ranges to episode duration.
- Export full-frame clips with source dimensions and audio when present; use frame-exact video boundaries, including for variable-frame-rate input.
- Document the folder layout, command/Python usage, output naming, quality settings, and face-visibility limitation.

#### Do Not Change

- Detection/classification models or the original face-image workflow.

#### Reuse First

- Use Phase 1 match reports and the installed `ffmpeg` executable; avoid a new media dependency.

#### Steps

1. Define deterministic clip ranges from per-frame confirmed/absent/uncertain status; cover departures and returns, zoom changes, boundaries, and no matches.
2. Call `ffmpeg` via `subprocess` argument lists, select exactly the matching source frames, preserve source dimensions, include audio if present, and use a configurable encoding quality.
3. Give distinct output names per episode and clip; avoid overwriting prior output silently.
4. Add focused tests for range merging and media export; verify a tiny generated video's dimensions, duration, and audio.
5. Update README and run the security check.

#### Acceptance Criteria

- Each matching episode yields playable clips under `anime/<series>/results/<character>/<episode>/`; separate appearances produce separate clips with no frames marked absent or uncertain by the head-presence check.
- Clips keep source width and height; audio is present when the source has audio.
- Episodes without matches produce no clips; reruns do not silently destroy existing output.
- The README shows how to add another character without changing code and explains that model false positives/negatives still require visual review for strict guarantees.

#### Verification

- Unit tests for clip-range logic (including match → absence → return) and `ffmpeg` command construction.
- Verify decoded frames in generated test clips align with the expected first/last matching source frames.
- Integration test using a tiny generated video, checking clip duration, dimensions, and audio stream.
- Manual sample-video review for match quality when footage is available.

#### Security Check

- Confirm bounded output paths, no shell execution, no input deletion, and safe handling of filenames containing spaces or punctuation.
