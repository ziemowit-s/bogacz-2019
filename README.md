Implementation of the paper: Möller, M., & Bogacz, R. (2019). Learning the payoffs and costs of actions. 
In PLoS Computational Biology (Vol. 15, Issue 2). https://doi.org/10.1371/journal.pcbi.1006285

## Run

### make_fig6.py
Recreate fig. 6 from the paper

|                            |  Pellet | Chow |
|----------------------------|:-------:|------|
| **Control**                |         |      |
| replication                | 1.05    | 0.45 |
| fig 6                      | 1.0     | 0.4  |
| **D2R Depleted**           |         |      |
| replication                | -0.5704 | 0.45 |
| fig 6                      | -0.6    | 0.4  |

### make_fig8.py
Recreate fig. 8 from the paper

Replication of fig. 8:
* B) Theory chart
* D) Theory chart

![fig8](figures/fig8.png)

Original fig. 8 from the paper: 
* B) Theory chart
* D) Theory chart

![fig8publication](figures/fig8_publication_b_d.png)

Values of GO/NO-GO populations for a single run (single rat):
* left column: no cost for obtaining the pellet
* right column: with cost for obtaining the pellet
* first row: go for pellet action
* second row: go for chow action
![go/no-go](figures/go-nogo-values.png)