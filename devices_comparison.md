### True values, measurements from device 1, measurements from device 2
1. Fit linear regression for each device vs true concentration
2. Separate t-tests on difference(measurements, true) == 0
3. If each device measured the same samples multiple times, get the SD to assess precision and repeatability
4. Bland-Altman plot for Device 1 vs Device 2

Note about Bland-Altman: 
In Bland–Altman terms, the limits of agreement are NOT the acceptance criteria.
Defining “acceptable limits” in Bland–Altman or any method comparison study is the most important—and also the most non-statistical—part of the whole process. Statistics can estimate variability and bias, but it cannot decide what error is tolerable. That decision is fundamentally domain- and risk-driven.
Acceptable limits should be defined before analysis, jointly by domain experts and QA/regulatory stakeholders, based on decision impact, safety thresholds, or regulatory requirements—not derived from the data or from Bland–Altman results.
Examples:

If sulphate > 250 mg/L is problematic,
then ±5 mg/L error might be negligible.
If you are tracking small changes (e.g. 10–20 mg/L range),
then ±5 mg/L might be unacceptable.