## The Task
In this competition, you'll build models to extract two key pieces of information from specimen label images:

1. **verbatimDate**: The collection date as written on the label (e.g., "27.IV.2022", "18-7 1965", "10/5 14")
2. **verbatimLocality**: The collection location as written on the label (e.g., "DENMARK: NEZ: Helsingør, Grønne Vestkile, 56.0375N, 12.5598E", or simple "Kb" which represents København (the Capital of Denmark))

### What Makes This Challenging?

- **Historical spelling and handwriting**: Labels span over a century with varying handwriting styles, often using cursive writing. One specific to this dataset is danish writing used to use umlauts, now place names with these often are changed as follows: ö -> ø, ÿ -> y). In the solutions everything has been transcribed to the danish alphabet (abcdefghijklmnopqrstuvwxyzæøå) 
- **Multiple non-standard date formats**: Dates may use Roman numerals, various punctuation or abbreviated months in sometimes non standard formats (eg Septmbr)
- **Label condition**: Some labels are faded, damaged, or partially obscured
- **Mixture of handwritten and machine written text**: Often labels, even dates, will be partially machine written and partially hand written. 
- **Multi-card specimens**: Some specimens have information split across multiple labels - See the dataset page for more details
- **Missing and confounding labels**: Some specimens may not have a date or locality label, but may have other labels - see the dataset page for more information

## ECCV CVNH Workshop
This Competition is part of the Computer Vision for Natural Heritage Workshop (CVNH) ([link](https://computer-vision-for-natural-heritage.github.io/)) which will take place at the ECCV 2026 Conference in Malmø, Sweden in September 2026. The results of the competition will be evaluated by a review panel and a few participants will be invited to present their results at the workshop, virtually or at your own expense. Attending the workshop is not required to participate in the competition.   

## Timeline

- **May 19, 2026**: Competition Starts
- **August 31, 2026**: Competition Ends
- **September 8 or 9** (exact date TBD): CVNH Workshop at ECCV 2026

