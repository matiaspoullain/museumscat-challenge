## Overview

This dataset contains 3,500 images of digitized pinned specimens from Natural History Museum of Denmark's Danish Dung Beetle collection. Collected from the late 1800s to present. 

## Files

- **train.csv**: 200 training examples with ground truth annotations
- **test.csv**: Test examples (image filenames only)
- **images.zip**: All competition images

## Data Fields

### Training Data (train.csv)

- `image_file`: Filename of the label image (e.g., "1709398.jpeg")
- `verbatimDate`: The collection date exactly as written on the label
- `verbatimDate_confidence`: 1 since all are 'ground truth'. Added for consistency with submission format
- `verbatimLocality`: The collection location exactly as written on the label
- `verbatimLocality_confidence`: 1 since all are 'ground truth'. Added for consistency with submission format

### Test Data (test.csv)

- `image_file`: Filename of the label image

## Data Characteristics

### verbatimDate Formats

Dates appear in many formats reflecting different time periods and conventions:

- **Standard formats**: "27.5.2022", "18-7 1965"
- **Roman numerals**: "27.IV.2022" (April), "12.VI.2000" (June)
- **Abbreviated**: "4/5 70" (May 4, 1870)
- **Partial dates**: "IV.91" (April 1891)
- **Missing dates**: "NULL". Note: kaggle does not accept empty values so please denote missing dates and localities as "NULL"

**Important:** The metric normalizes some date punctuation! All these are treated as equivalent:
- "27.5.2022"
- "27,5,2022"
- "27-5-2022"  
- "27·5·2022"  
- "27 5 2022"

This allows flexible matching since it is often difficult to tell what the original author wrote, and it is unimportant for the transcription.

### verbatimLocality Formats

Localities range from simple place names to detailed coordinates:

- **Simple**: "Dyrehaven", "Bekkes"
- **Hierarchical**: "V. Sjælland Røsnæs Kongstrup K1"
- **With coordinates**: "DENMARK: NEZ: Helsingør, Grønne Vestkile, 56.0375N, 12.5598E"

**"Dania" is not a place**
Note: "Dania" is both the name of the danish collection and sometimes used as a location (Denmark). Since this is ambiguous for the purposes of this dataset we do not consider "Dania" as a location (so "Dania" does not appear in the ground truth verbatimLocality).

**"i kogøding" (In cow dung) is also not a place**
Sometimes the label will say something like 'i kogjorning', which means the specimen was found "in cow dung". This is not considered part of the locality information eg,

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5490045%2Ff9ad75dc0caea8a7059ef7cec76084d1%2FScreenshot%20from%202026-05-11%2011-16-57.png?generation=1778491034472989&alt=media)

Here the correct locality transcription is: ```verbatimLocality: "Kb | Dyrehaven"```. Mus Løv refers to the previous museum collection. Kb is short for København, the capital of Denmark. Dyrehaven is a park near København with many deer, and is a popular insect collection spot.


### Multi-Card Specimens (The Pipe Quirk!)

Some specimens have information split across multiple cards/labels. In the ground truth, these are represented with **pipe separators (|)**:

Example 1: ```verbatimLocality: "Ti | Kulhuset Jægerspris", verbatimDate: "6/1862"```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5490045%2F617f1142462f4234249534e6d7e42aed%2FScreenshot%20from%202026-05-11%2011-21-33.png?generation=1778491355588287&alt=media)

Example 2: ```verbatimLocality: "Rotholme Jyll. | Rotholme Jyll.", verbatimDate: "10.6.1951 | 10.6.1951"```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5490045%2F638588ad2c2a3017fe3c421e0031b040%2FScreenshot%20from%202026-05-11%2011-18-20.png?generation=1778491135478355&alt=media)

Here two specimens with two sets of labels are on the same pin, and both dates and both localities should be transcribed, even if those dates and localities are the same (often they are not).


**How scoring works:**
- The metric tries ALL possible orderings of the cards, in the example 1 above if you submit "Ti | Kulhuset Jægerspris", then both "Ti Kulhuset Jægerspris", and "Kulhuset Jægerspris | Ti" will be checked against the solutions
- Your prediction is scored against the best matching order
- The pipes ("|") themselves are removed from the predictions, so you can use them generously when there is ambiguity
- All cards must be transcribed for a perfect score

This accommodates specimens where:
- Information is split across multiple physical labels
- The reading order might not be obvious from the image
- Labels are arranged vertically, horizontally, or irregularly

### Missing Values

Both dates and localities may be missing from the label. Predicting an empty string when appropriate can improve your score! Note that kaggle does not accept submissions with empty values, **therefore please denote these as "MISSING" in your submission**, as they are in the train csv, then we will handle them appropriately in the matching. 

## Confounding Information and Dates
Often there are other labels which we are not interested in at the moment, ie who collected the specimen, the determined species, and when the species determination was made. eg,

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5490045%2F8cf2161e91158e617a97b8eb8c7c432a%2FScreenshot%20from%202026-05-11%2011-29-14.png?generation=1778491785009862&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5490045%2F2920d354a49c460da9ac40c8a4c3cd72%2FScreenshot%20from%202026-05-11%2011-31-09.png?generation=1778491890555810&alt=media)

Sometimes, as above, the collector name is prefixed with "Coll." or similar, and often if a date is related to a species determination or a change in collector, this will be indicated by it being on the same label as the species name, and/or prefixed with 'det.' or 'Tilg.' as above.

Here the correct transcription is:
```verbatimDate: "2-4-49"
verbatimLocality: "Rude Skov"
```
## Evaluation Details

### Normalized Edit Distance (NED)

The metric uses NED to measure transcription accuracy:

- **NED = 0.0**: Perfect match
- **NED = 1.0**: Completely wrong
- **NED = 0.5**: Half the characters need editing

NED is calculated as: `edit_distance / max_length`

**Key features:**
- Case-insensitive matching
- Date punctuation normalization (for dates only, see above)
- Handles pipe separators in ground truth

### AURC Calculation

For automatic transcription it is crucial that the model can give an accurate confidence score so easy predictions can be completed automatically and uncertain ones can be reviewed by experts. Therefore we use the AURC metric which includes ranking of model confidence.

The Area Under Risk-Coverage curve evaluates your predictions in order of confidence. :

1. Predictions are sorted by confidence (highest first)
2. For each coverage level (top 10%, top 20%, etc.), we calculate the average normalized edit distance (NED)
3. AURC is the area under the risk-coverage curve

**What this means:**
- A perfect model would be confident about correct predictions and uncertain about errors
- High confidence on wrong answers hurts your score significantly
- Low confidence on correct answers hurts less (but still matters)
- The metric rewards knowing what you don't know

## Data Quality Notes

- All images have been manually reviewed and annotated by experts
- Ground truth represents the text exactly as written, including spelling variations and abbreviations
- Some labels are quite challenging, even for experts (faded, damaged, or unclear handwriting)
- Historical spelling differs from modern conventions

## Questions?

Post in the competition discussion board for clarification on:
- Data ambiguities
- Metric calculation details
- Submission format issues etc.