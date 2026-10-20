# Résultats

Données BCE du 05/01/2010 au 19/10/2026 (4 300 jours ouvrés). Portefeuille : USD 4,0 M€, GBP 2,5 M€, JPY 2,0 M€, CHF 1,5 M€.

Kurtosis en excès des rendements réduits par l'EWMA : 4,30 (0 pour une loi normale). Degrés de liberté de la Student estimés à la dernière date : ν = 4,3.

## Risque à la dernière date (horizon 1 jour)

| niveau   | modèle                        | VaR (€)   | ES (€)    |
|:---------|:------------------------------|:----------|:----------|
| 99 %     | Simulation historique (500 j) | 75 503 €  | 112 807 € |
| 99 %     | Variance-covariance (500 j)   | 61 674 €  | 70 657 €  |
| 99 %     | Monte Carlo normal + EWMA     | 55 481 €  | 63 454 €  |
| 99 %     | Monte Carlo Student + EWMA    | 62 730 €  | 85 657 €  |
| 97,5 %   | Simulation historique (500 j) | 51 128 €  | 80 654 €  |
| 97,5 %   | Variance-covariance (500 j)   | 51 960 €  | 61 977 €  |
| 97,5 %   | Monte Carlo normal + EWMA     | 46 709 €  | 55 591 €  |
| 97,5 %   | Monte Carlo Student + EWMA    | 46 873 €  | 65 874 €  |

## Backtesting VaR 99 % (12/12/2011 – 19/10/2026)

| modèle                        |   jours |   exceptions |   attendues | taux   |   p Kupiec |   p Christoffersen |   exceptions 250 j | zone Bâle   |
|:------------------------------|--------:|-------------:|------------:|:-------|-----------:|-------------------:|-------------------:|:------------|
| Simulation historique (500 j) |    3800 |           45 |          38 | 1,18 % |      0,267 |              0,563 |                  0 | vert        |
| Variance-covariance (500 j)   |    3800 |           41 |          38 | 1,08 % |      0,629 |              0,466 |                  0 | vert        |
| Monte Carlo normal + EWMA     |    3800 |           55 |          38 | 1,45 % |      0,009 |              0,204 |                  3 | vert        |
| Monte Carlo Student + EWMA    |    3800 |           43 |          38 | 1,13 % |      0,425 |              0,321 |                  2 | vert        |
