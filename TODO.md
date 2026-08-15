# QuantumOX – TODO

## Neural Evaluation Integration
- [ ] Implement **OXNN (OX Neural Network)** for move evaluation.
  - OXNN will evaluate candidate move sequences produced by Minimax.
  - Integrate OXNN evaluations into the main search pipeline for hybrid decision-making.
  - Before printing any `info depth` lines after a `go` command, the engine should display:
    ```
    info string OXNN evaluation using <oxnn file name> (<file size>, <model information>)
    ```
  - OXNN model filenames must follow this format:
    ```
    nn-<first 12 digits of SHA256 hash of the OXNN file’s binary contents>.oxnn
    ```
    ensuring deterministic and version-safe naming for all neural evaluation models.
    (*Inspired by NNUE*)

## Planned Improvements
- [ ] Implement self-play training to generate evaluation data for OXNN.
- [ ] Improve transposition table efficiency and move ordering heuristics.
- [ ] Optimize thread workload distribution and synchronization for the multi-threaded search.
- [ ] Add benchmarking mode for comparing Minimax and OXNN-evaluated results.

## Documentation
- [ ] Add developer documentation detailing OXNN’s role in the hybrid evaluation process.

**Version:** 1.3-pre  
**Maintained by:** Kartik  
