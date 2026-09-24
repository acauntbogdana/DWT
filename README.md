# dataset-report

Сравнение модели гистерезиса и EWMA из работы: [ 
Grant, A., Mrazik, M., & Satchell, S. (2026). Evaluating forecasts at multiple horizons: An extension of the Diebold–Mariano approach. Journal of Forecasting, *0*, 1–14. https://doi.org/10.1002/for.70150
]
## Требования

- Rust 1.75+ (edition 2021)
- Parquet-данные: IEX DEEP
  
## Сборка и запуск
```bash
cargo build --release
cargo run --release
