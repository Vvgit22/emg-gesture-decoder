# Participant-level results

Windows within a trial are correlated; the participant is the unit.

## cnn / both / cross_subject
- n=43 participants; mean 79.60%, median 81.30%, SD 7.43 pp, range 59.38–90.07%, p5 67.46%
- below 70%: 5; below 80%: 16
- trial-level (majority vote): 20.60%
- rest→active 2.96%, active→rest 0.88%

## lda / both / cross_subject
- n=43 participants; mean 67.98%, median 68.58%, SD 8.30 pp, range 40.56–81.79%, p5 55.29%
- below 70%: 24; below 80%: 42
- trial-level (majority vote): 8.31%
- rest→active 13.22%, active→rest 1.81%

### paired CNN−LDA (cross_subject)
mean 11.61 pp, SD 4.75 pp, 95% CI [10.24, 13.03] (participant bootstrap, 10k)

## cnn / both / cross_day
- n=43 participants; mean 88.79%, median 89.39%, SD 5.73 pp, range 76.35–98.39%, p5 79.23%
- below 70%: 0; below 80%: 4
- trial-level (majority vote): 14.62%
- rest→active 5.79%, active→rest 1.12%

## lda / both / cross_day
- n=43 participants; mean 87.02%, median 89.36%, SD 7.25 pp, range 66.99–98.10%, p5 73.93%
- below 70%: 1; below 80%: 8
- trial-level (majority vote): 12.62%
- rest→active 8.04%, active→rest 1.38%

### paired CNN−LDA (cross_day)
mean 1.78 pp, SD 4.96 pp, 95% CI [0.36, 3.26] (participant bootstrap, 10k)
