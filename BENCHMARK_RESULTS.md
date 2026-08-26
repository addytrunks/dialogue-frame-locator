# Benchmark Results 

ASR provider: `openrouter` 

*yet to test for local*

## Per-case results

| id | kind | expected | actual | exit | onset err (ms) | frame err | WER | confidence | notes |
|---|---|---|---|---|---|---|---|---|---|
| clean_speech | synthetic | FOUND | FOUND | 0 | 36.0 | 0 | 0.000 | 0.94 | |
| phrase_twice | synthetic | AMBIGUOUS | AMBIGUOUS | 2 | 8.0 | 1 | 0.000 | 0.94 | |
| phrase_absent | synthetic | NOT_FOUND | NOT_FOUND | 3 | n/a | n/a | n/a | 0.00 | |
| no_audio | synthetic | NOT_FOUND | NOT_FOUND | 3 | n/a | n/a | n/a | 0.00 | |
| chunk_boundary | synthetic | FOUND | FOUND | 0 | 44.0 | 0 | 0.000 | 0.94 | |
| background_music | synthetic | FOUND | FOUND | 0 | 48.0 | 0 | 0.000 | 0.94 | |
| low_bitrate_resolution | synthetic | FOUND | FOUND | 0 | 48.0 | 0 | 0.200 | 0.79 | |
| vfr_clip | synthetic | FOUND | FOUND | 0 | 22.0 | n/a | 0.000 | 0.94 | |
| accent_variation | synthetic | FOUND | FOUND | 0 | 48.0 | 1 | 0.000 | 0.94 | |
| ok_ru_example | real | FOUND | - | - | - | - | - | - | SKIPPED: kind=real, pass --include-real to run |

## Aggregate metrics (DESIGN.md §15.1)

- Cases: 10 total, 9 scored, 1 skipped, 0 harness errors
- Status exact-match accuracy: 100.0%
- Confusion (positive = FOUND/AMBIGUOUS): TP=7 FP=0 FN=0 TN=2
- Precision: 100.0%  |  Recall: 100.0%
- False-positive rate: 0.0%  |  False-negative rate: 0.0%
- Onset error (n=7): median=44.0ms, P90=48.0ms
- Tolerance-band accuracy: ±100ms=100.0%, ±500ms=100.0%
- Tolerance-band accuracy (frame, n=6): ±1frame=100.0%, ±5frame=100.0%
- WER (n=7): median=0.000
- Mean wall time per case: 6.4s
