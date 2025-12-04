# Election 2020 Twitter Dataset (IEEE DataPort)

## Download Instructions

1. Go to: https://ieee-dataport.org/open-access/usa-nov2020-election-20-mil-tweets-sentiment-and-party-name-labels-dataset

2. Click "Download Dataset" (may require free IEEE account)

3. Download the CSV file (~3.48 GB)

4. Place in this directory as `election2020_tweets.csv`

## Dataset Details

- **Size**: 24,201,654 tweets
- **Period**: July 1, 2020 - November 11, 2020
- **Format**: CSV with 11 columns

### Columns

| Column | Description |
|--------|-------------|
| Id | Unique tweet ID |
| Created-At | Timestamp |
| From-User-Id | User who posted |
| To-User-Id | Reply target (if any) |
| Language | Tweet language |
| Score | VADER sentiment score (-1 to +1) |
| Scoring String | Words contributing to score |
| Negativity | Sum of negative components |
| Positivity | Sum of positive components |
| Party | Detected party affiliation |
| Text | Full tweet text |

## For Hamiltonian Analysis

This dataset allows testing:
1. **Oscillation detection**: Do sentiment scores oscillate after news events?
2. **Network effects**: Do users influence each other's sentiment trajectories?
3. **Social mass**: Does network position predict response speed?
