
Submissions are evaluated using **AURC (Area Under the Risk-Coverage Curve)** [1], which measures both:

1. **Normalized Edit Distance**: How close your transcriptions are to the ground truth
2. **Calibration**: How well your confidence scores reflect actual prediction quality

### Why AURC?

Unlike traditional accuracy metrics, AURC rewards models that "know what they don't know." A well-calibrated model should:
- Assign high confidence to accurate predictions
- Assign low confidence to uncertain or difficult cases

This is crucial for real-world deployment where conservators need to know which transcriptions require manual verification.

**Lower AURC scores are better.** The metric ranges from 0 (perfect) to 1 (worst).



The final score is the average of:
- AURC for verbatimDate predictions
- AURC for verbatimLocality predictions

## Submission Format

Your submission must include predictions and confidence scores for each test image:

```csv
image_file,verbatimDate,verbatimDate_confidence,verbatimLocality,verbatimLocality_confidence
1234567.jpeg,"27.IV,2022",0.95,"DENMARK: Helsingør",0.87
1234568.jpeg,"3/10 95",0.5,"Kb",0.9
1234569.jpeg,"MISSING",0.5,"MISSING",0.9
...
```

**Required columns:**
- `image_file`: Image filename
- `verbatimDate`: Your predicted date transcription
- `verbatimDate_confidence`: Confidence score for date (0.0 to 1.0, where 1.0 is most confident)
- `verbatimLocality`: Your predicted locality transcription
- `verbatimLocality_confidence`: Confidence score for locality (0.0 to 1.0, where 1.0 is most confident)


## Acknowledgements
Data for this competition has been painstakingly annotated by digitizers at the Natural History Museum of Denmark, funded through the DaSSco initiative. Thank you Kim Steenstrup Pedersen for providing access to the data, and Aslak Kappel Hansen for being project PI behind the dung beetle project and the many digitizers for their expert efforts to transcribe the information. We would also like to thank ECCV 2026 for hosting the first CVNH Workshop which includes this competition.

### References
[1] Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 4885–4894.
