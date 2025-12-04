# Twitter Sentiment Geographical Index (TSGI)

## Download Instructions

1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3IL00Q

2. Click "Access Dataset" → "Download"

3. Download the CSV files for the years/regions you want

4. Place files in this directory

## Dataset Details

- **Source**: Harvard Dataverse / MIT Sustainable Urbanization Lab
- **Coverage**: 164 countries, 2019-present
- **Granularity**: Daily sentiment at country/state/county level
- **Size**: 4.3 billion geotagged tweets

### Data Structure

Each CSV contains:
| Column | Description |
|--------|-------------|
| date | Date (YYYY-MM-DD) |
| admin0 | Country |
| admin1 | State/Province |
| admin2 | County/City |
| sentiment_mean | Average sentiment (-1 to +1) |
| sentiment_std | Standard deviation |
| tweet_count | Number of tweets |

## For Hamiltonian Analysis

This dataset allows testing:
1. **Long-term oscillations**: Multi-day/week cycles in sentiment
2. **Cross-regional dynamics**: How sentiment propagates geographically
3. **Event responses**: National/global shocks (elections, crises)

## Citation

Chai, Y., Kakkar, D., Palacios, J. & Zheng, S. (2022). Twitter Sentiment
Geographical Index. Harvard Dataverse. https://doi.org/10.7910/DVN/3IL00Q
