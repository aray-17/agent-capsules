# T-059 Phase 0 baseline H2H — AC vs DSPy

Pipeline: startup_due_diligence | Worker: claude-sonnet-4-6 | Judge: claude-opus-4-6 | 7 tasks, shared across cells.

## Per-cell aggregate

| cell | n | mean input | mean output | mean total | mean wall (s) | mean q | Δtok vs ac | Δq vs ac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ac_auto_with_evaluator | 7 | 27,910 | 15,594 | 43,505 | 325 | 0.680 | +25.2% | +0.046 |
| ac_compound_sequential | 7 | 16,443 | 24,268 | 40,711 | 485 | 0.761 | +17.1% | +0.128 |
| ac_fine | 7 | 21,455 | 13,297 | 34,753 | 274 | 0.633 | — | — |
| dspy_mipro | 7 | 102,612 | 26,121 | 128,733 | 534 | 0.709 | +270.4% | +0.075 |
| dspy_uncompiled | 7 | 34,851 | 15,650 | 50,501 | 326 | 0.749 | +45.3% | +0.115 |

## Per-task detail

### ac_auto_with_evaluator

| task | company | input | output | total | calls | wall (s) | q |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Acme Fintech Inc. | 25,806 | 18,817 | 44,623 | 5 | 391 | 0.750 |
| 2 | Beta Robotics | 31,915 | 22,537 | 54,452 | 5 | 432 | 0.697 |
| 3 | Citadel Bio | 69,694 | 37,066 | 106,760 | 7 | 766 | 0.630 |
| 4 | Delta Logistics | 17,927 | 8,387 | 26,314 | 5 | 187 | 0.630 |
| 5 | Epoch Semiconductors | 15,488 | 6,475 | 21,963 | 5 | 152 | 0.650 |
| 6 | Forge Analytics | 17,550 | 8,272 | 25,822 | 5 | 176 | 0.650 |
| 7 | Gnosis Health | 16,996 | 7,608 | 24,604 | 5 | 171 | 0.750 |

### ac_compound_sequential

| task | company | input | output | total | calls | wall (s) | q |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Acme Fintech Inc. | 14,504 | 20,607 | 35,111 | 5 | 475 | 0.823 |
| 2 | Beta Robotics | 16,048 | 23,073 | 39,121 | 5 | 491 | 0.817 |
| 3 | Citadel Bio | 17,884 | 28,061 | 45,945 | 5 | 545 | 0.773 |
| 4 | Delta Logistics | 16,353 | 25,746 | 42,099 | 5 | 482 | 0.640 |
| 5 | Epoch Semiconductors | 16,691 | 24,609 | 41,300 | 5 | 482 | 0.730 |
| 6 | Forge Analytics | 15,785 | 22,296 | 38,081 | 5 | 405 | 0.883 |
| 7 | Gnosis Health | 17,838 | 25,487 | 43,325 | 5 | 512 | 0.663 |

### ac_fine

| task | company | input | output | total | calls | wall (s) | q |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Acme Fintech Inc. | 28,258 | 21,210 | 49,468 | 5 | 438 | 0.697 |
| 2 | Beta Robotics | 29,690 | 22,189 | 51,879 | 5 | 439 | 0.690 |
| 3 | Citadel Bio | 22,591 | 15,934 | 38,525 | 5 | 318 | 0.663 |
| 4 | Delta Logistics | 15,622 | 8,133 | 23,755 | 5 | 174 | 0.607 |
| 5 | Epoch Semiconductors | 20,881 | 10,529 | 31,410 | 5 | 222 | 0.597 |
| 6 | Forge Analytics | 18,852 | 8,095 | 26,947 | 5 | 181 | 0.597 |
| 7 | Gnosis Health | 14,293 | 6,995 | 21,288 | 5 | 149 | 0.583 |

### dspy_mipro

| task | company | input | output | total | calls | wall (s) | q |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Acme Fintech Inc. | 99,657 | 25,291 | 124,948 | 9 | 514 | 0.730 |
| 2 | Beta Robotics | 105,788 | 27,813 | 133,601 | 9 | 586 | 0.630 |
| 3 | Citadel Bio | 102,121 | 25,665 | 127,786 | 9 | 531 | 0.640 |
| 4 | Delta Logistics | 98,984 | 24,032 | 123,016 | 9 | 470 | 0.750 |
| 5 | Epoch Semiconductors | 108,589 | 28,514 | 137,103 | 9 | 594 | 0.663 |
| 6 | Forge Analytics | 105,294 | 25,983 | 131,277 | 9 | 536 | 0.850 |
| 7 | Gnosis Health | 97,854 | 25,549 | 123,403 | 9 | 511 | 0.697 |

### dspy_uncompiled

| task | company | input | output | total | calls | wall (s) | q |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Acme Fintech Inc. | 30,602 | 14,608 | 45,210 | 9 | 277 | 0.707 |
| 2 | Beta Robotics | 39,919 | 16,388 | 56,307 | 12 | 324 | 0.797 |
| 3 | Citadel Bio | 33,783 | 15,637 | 49,420 | 11 | 387 | 0.773 |
| 4 | Delta Logistics | 37,847 | 17,324 | 55,171 | 11 | 347 | 0.807 |
| 5 | Epoch Semiconductors | 35,920 | 18,364 | 54,284 | 10 | 388 | 0.740 |
| 6 | Forge Analytics | 34,018 | 13,156 | 47,174 | 12 | 275 | 0.833 |
| 7 | Gnosis Health | 31,870 | 14,075 | 45,945 | 11 | 282 | 0.583 |

