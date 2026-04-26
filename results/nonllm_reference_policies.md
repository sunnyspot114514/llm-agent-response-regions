# Non-LLM Reference Policies

| Policy | Construction | Mean Entropy | Mean Persistence |
| --- | --- | ---: | ---: |
| always_cooperate | deterministic reference | 0.000 | 1.000 |
| uniform_random | iid uniform over 5 actions | 2.212 | 0.192 |
| sticky_markov_p0.85 | uniform start, then stay with p=0.85 | 1.460 | 0.859 |

