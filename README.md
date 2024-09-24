# Gravitational Search Algorithm for Feature Selection

---
Sample commands to run the code:

Minimal command:
```bash
python optimizer.py --dataset ../Iris.csv
```

Full command:
```bash
python optimizer.py --classifier RF --metric acc --dataset ../Iris.csv --test_size=0.3 --threshold 0.5 --runs 1 --pop_size 5 --iterations 10 --export True --export_name experiment.csv
```
___
### Hyperparameter Explanation

#### 1. **Population Size**:
- The number of agents (candidate solutions) in the optimization process. A larger population size increases the diversity of solutions and allows the algorithm to explore more areas of the search space.
- **Increase**: More exploration of the search space, higher chance of finding the optimal solution, but longer computation time.
- **Decrease**: Faster execution, less exploration, may result in suboptimal solutions due to limited diversity.

#### 2. **Iterations**:
- How long the algorithm runs and how much optimization is performed. Each iteration allows agents to update their positions in the search space based on gravitational interactions.
- **Increase**: Allows more fine-tuning, better chance to converge to a global optimum, but longer runtime.
- **Decrease**: Faster, but might stop before convergence, risking suboptimal solutions.

#### 3. **Runs**:
- How many independent times the algorithm is executed with different random initializations. Multiple runs allow you to gather statistically significant results and reduce the impact of randomness.
- **Increase**: More robust results by averaging multiple outcomes, reduces impact of randomness, but increases total computation time.
- **Decrease**: Faster, but results may be less reliable due to fewer independent attempts.

#### 4. **Threshold**:
- Use to select features based on their importance or value within an agent’s position. In feature selection, this threshold determines whether a feature is included or excluded from the final feature subset
- **Increase**: Fewer features are selected, leading to simpler models (less overfitting), but may exclude relevant features (risk of underfitting).
- **Decrease**: More features are selected, which may capture more information, but risks overfitting by including irrelevant features.

___
### Notes

- The first column is considered as the index and it is not considered (so column 0 is the second one)

- The last column is the target 

___
### My experiments

Args = (classifier=X, metric='f1', dataset='churn_data.csv', ignore_first=False, test_size=0.3, threshold=0.5, runs=1, pop_size=10, iterations=100, export=True, export_name='experiment.csv')

- LR: [ 2  3  4  6  7  9 11 15 16 17 20 21 23 27 29]
- RF:


___
### Attributes

1. total_events,
2. ui_events,
3. ui_open_events,
4. ui_close_events,
5. ui_click_events,
6. ui_onboard_events,
7. config_events,
8. config_button_map_events,
9. app_events,
10. config_point_and_scroll_events,
11. voice_use_events,
12. mouse_button_click_events,
13. connect_events,
14. mx_cat_count,
15. ergo_cat_count,
16. lifes_cat_count,
17. mains_cat_count,
18. other_cat_count,
19. mouse_type_count,
20. keyboard_type_count,
21. other_type_count,
22. device_number,
23. eu,
24. cont_AF,
25. cont_AS,
26. cont_EU,
27. cont_NA,
28. cont_SA,
29. cont_OC,
