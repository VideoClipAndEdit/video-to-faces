# Implementation Plan: Anime Character Clips

## 1. Goal

Given anime episodes and reference images for a named character, find frames showing that character's head and export short, watchable video clips containing those appearances at the source video's dimensions, with audio when present. The head is the only required visible body part. Hair color and, when an eye is visible, eye color take priority in identity matching. Every frame included in a clip must pass the character-presence check. Cut the clip when the character is no longer detected; start a new clip if they return.

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
- The references show light silver/white hair and blue-violet eyes. These are appearance cues, not hard-coded rules for other characters. A closed, hidden, or too-small eye is unavailable evidence rather than a negative match.
- Episodes may zoom in and out. Head size will vary; matching must use size-normalized head regions and not rely on a fixed pixel size.
- The existing CLI has no clip-export mode. `ffmpeg` is available in the current environment, but should be checked when the feature runs.
- User confirmed the output should be short clips containing the character.

Unknown items:

- Exact match threshold that works best across anime styles; expose it as an option and verify with sample footage.
- How reliably the existing anime face detector finds a head when little or none of the face is visible. Validate this with real episode frames before deciding whether a small additional head-candidate method is needed; never label an unverified head as Killua solely from hair color.
- How best to extract hair/visible-iris colors from varied drawings without including backgrounds, highlights, or clothing. Test the smallest approach using existing OpenCV capabilities before adding a dependency.

## 5. Areas to Inspect

- `src/videotofaces/detection.py`: frame sampling, filtering, and face crops.
- `src/videotofaces/grouping.py`: anime encoder and cosine-distance matching.
- `src/videotofaces/prep.py`, `main.py`, and `__main__.py`: input validation and CLI conventions.
- `tests/` and `README.md`: test and usage patterns.

## 6. Simplest Implementation Strategy

Add one focused character-clips workflow callable from the CLI and Python. Read episodes from `anime/<series>/episodes/` and references from `anime/<series>/reference/<character>/`. Crop references and candidates around the head, including hair, then classify every source frame. Compare reference-derived hair and visible-eye colors as priority cues, supported by the existing anime embedding for identity. Use head-relative regions so zoom changes do not change the matching rule; if the head or cues cannot be verified at a frame's scale/angle, classify it as uncertain rather than assuming presence. Consecutive confirmed frames form one appearance; the first absent or uncertain frame ends it, and a later match starts another. Use `ffmpeg` to cut full-frame clips at those frame boundaries into `anime/<series>/results/<character>/<episode>/`. Keep dimensions unchanged and use a quality setting suitable for viewing. Emit a per-frame match report, including cue scores and uncertainty, so clips can be audited. This gives frame-level enforcement, though model errors mean absolute visual certainty still requires human review.

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

**Status:** In Progress

**Depends on:** None

#### Goal

Produce per-episode match timestamps and scores for a named character from the proposed `anime/` layout.

#### Scope

- Add the focused workflow entry point and CLI option(s) for series folder and character name.
- Validate folder structure and reference images; support multiple references for one character, including portraits and wider poses.
- Isolate head regions from references and frames; match with reference-derived hair/visible-eye colors as priority cues and existing embeddings as supporting evidence. Treat uncheckable eyes as unavailable evidence.
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
3. Derive hair and visible-iris color cues from the references with existing image tools; combine them with embedding similarity so color has priority without using color alone as proof of identity.
4. Match each episode frame across head sizes and angles, marking absent/uncertain frames conservatively, and record timestamps, cue scores, and decisions.
5. Add focused tests for matching, zoom changes, an eye closed or too small to assess, multiple references, empty/invalid inputs, and safe paths.
6. Run relevant tests and the security check.

#### Acceptance Criteria

- A run for `hunterxhunter/killua` produces a timestamped per-frame report per processed episode, including non-matches, and clearly reports episodes with no matches.
- Multiple reference images can contribute to a match.
- The wider `killua2.png` is evaluated from its head, not from its clothing or black background; a closed eye does not automatically reject a match.
- Hair and visible-eye color influence the match more than the generic embedding, and the report exposes these cue scores. Head size changes due to zoom do not change the comparison regions.
- Frames with no verifiable character head are marked absent or uncertain and cannot enter a clip.
- Existing `video_to_faces` outputs and CLI remain functional.
- Invalid character names cannot escape the intended folders.

#### Verification

- Run focused unit tests with mocked detector/encoder and a tiny generated video; verify consecutive frame indexes, timestamps, and no-match behavior.
- Test both supplied Killua images for head cropping and color-cue extraction; include resized variants to cover zoom in/out and a closed-eye case.
- Run existing affected tests where locally feasible; model-weight downloads are not required for unit tests.

#### Security Check

- Inspect all input/output path handling and confirm no untrusted string enters a shell.

#### Execution Notes

- Added `src/videotofaces/character.py` with a Python entry point and `python -m videotofaces.character` CLI. It writes one per-frame `matches.csv` for each episode and reports when an episode has no confirmed matches.
- Five focused tests pass, including both supplied references, resized heads, color-priority decisions, safe paths, and a leave/return frame sequence. Syntax compilation, CLI help, layout validation, and `git diff --check` pass.
- The current Python environment lacks OpenCV and Torch, and no model weights are present. Real-model recognition on the episode and partial/back-of-head coverage remain unverified; those results may require threshold or candidate-detection adjustments during review.

#### Review Notes

- The five focused tests pass locally, but `tests/test_character.py` reads `killua1.png` and `killua2.png`. Both are ignored by `.gitignore` and absent from `git ls-files`, so the test suite is not reproducible from a clean checkout.
- `PYTHONPATH=src python3 -m videotofaces.character --help` fails because OpenCV is missing; Torch and model weights are also absent. Reference detection and a full per-frame report for the supplied episode were not validated.
- No security issue was found in the Phase 1 changes. Recheck the actual head/face detector's coverage of the supplied references and profile/back-of-head frames before another review.

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
