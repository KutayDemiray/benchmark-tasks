# Recorded evidence and limitations

| Run | Score | Ordinary actions | Simulated time | Outcome |
| --- | --- | --- | --- | --- |
| Scripted demonstration | 1.0 | 905 | 45.25 s | Three correct snaps retained after withdrawal |
| Fresh-process scripted reproduction | 1.0 | 895 | 44.75 s | Same success conditions |
| Genuine Dream-1 attempt | 0.0 | 80 | 4.0 s | Provider session error |
| Exact-action open-loop replay | 0.0 | 905 | 45.25 s | Valid physics, no retained target pairs |

All four use seed 100 and 20 Hz ordinary 14D targets. Demonstration and model
share the same native scene/reset, physical configuration, instruction, scoring,
recording and 6000-action budget. Scripted feedback is used only for feasibility;
external policy mode rejects script commands. The independent audits verify
recorded state, action provenance, camera inputs, actual score and termination.

Dream returned one real 48-target chunk in 2.1413 wall seconds. Together with
32 standard preroll holds, those 80 actions form the complete uncut recording.
The second request failed. The exact provider error was:

```text
inference session e26dba35-3460-48b9-b8f5-3f1cad2935f0 was replaced by 0c8ef758-70e1-4734-a96f-0a7957ac9277 (expired): the episode restarted server-side
```

There were two started requests, one completed request and no episode retry.
No physics invalidity occurred. The native process shut down cleanly and the
scheduler confirmed all descendants completed. The service interruption is
preserved with score zero; it is not a full-budget model performance result.
The cause of session expiry is unverified. Slow native execution is a hypothesis.

The final local audit passed nine evidence checks and the evaluated source
passed 170 CPU tests. The selected 64-iteration, 5 mm mesh configuration passed
two 10-second reset screens and a 300-second horizon. The shortcut suite passed
13 cases, including wrong pairs, proximity-only credit, hidden attachments,
robot support, damage, rim penetration and legitimate partial scores. These
checks do not establish a population success rate. Same-process reset failures,
failed coarser candidates and failed hanger experiments remain documented.

Both videos are anonymous, uncut H.264/yuv420p MP4s with overhead and two wrist
views plus a factual outcome card. The model video has 80 frames at 20 fps and
221336 bytes; the full-score demo has 905 frames and 2034091 bytes. The setup PNG
has 37357 bytes and comes from the first overhead frame at 0.05 simulated seconds.
Source streams and exports decode completely, and actual start/pressing/release
or model-motion frames were inspected. The metadata scores match these videos.
The upload's task-specific media ID belongs in `task.json`.

Hashes, exact package versions and compact audit results are recorded in
[submission-evidence.json](submission-evidence.json). Executed model source:
`f44ad36cd58f4dd7dceb0f5486f84c4f58369d409ad1ad4faffeec4f0055a233`.
Executed 20 Hz demo source:
`75a94d8696044f6e2d08180f3fe0f28a9aae91cc16325a04c1e459df2367286c`.
Raw actions, JPEG inputs, physics-rate states and inference logs are retained
outside Git. The repository contains source and compact provenance, not large
recordings or credentials.

Submission packaging changes only metadata, documentation and path configuration.
Every physical/shared native/controller/policy/scorer file remains byte-identical
to the evaluated source. The GPU helper's functions/classes are unchanged; only
its scheduler path constant moves into `task_paths.py`. The exact new helper hash
is accepted alongside the exact historical one in the existing provenance guard;
unknown helpers and changed physics remain rejected. This cleanup required no
new simulation or paid model call.

The cleaned submission passed **171 CPU tests**, including a new regression
check for rejecting any further unreviewed GPU-helper edit. The 29 protected
physical/shared-runtime/controller/policy/scorer files match the evaluated source.
No additional simulator or hosted model run was launched for packaging.
