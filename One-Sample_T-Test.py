from scipy import stats

scores = [72, 88, 64, 74, 67, 79, 85, 75, 89, 77]
hypothesized_mean = 70

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(scores, hypothesized_mean)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print("Reject Null Hypothesis: Mean is significantly different from 70.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference from 70.")
