| capture | version | acc_off | acc_on | needs_thinking | rescued n | helped | hurt |
|---|---|---|---|---|---|---|---|
| gsm8k_thinking_qwen3v3 | before | 0.933 | 0.940 | 0.067 | 89 | 48 | 38 |
| | after  | 0.933 | 0.923 | 0.067 | 89 | 42 | 54 |
| | audit  | 0.933 | 0.923 | 0.067 | 89 | 42 | 54 |
| math500_thinking_qwen3v3 | before | 0.764 | 0.782 | 0.236 | 118 | 39 | 30 |
| | after  | 0.824 | 0.820 | 0.176 | 88 | 33 | 35 |
| | audit  | 0.824 | 0.822 | 0.176 | 88 | 33 | 34 |
| mmlu_pro_thinking_qwen3v3 | before | 0.624 | 0.693 | 0.376 | 376 | 131 | 62 |
| | after  | 0.617 | 0.648 | 0.383 | 383 | 102 | 71 |
| | audit  | 0.617 | 0.648 | 0.383 | 383 | 102 | 71 |
| bbh_thinking_qwen3v3 | before | 0.676 | 0.822 | 0.324 | 175 | 100 | 21 |
| | after  | 0.761 | 0.843 | 0.239 | 129 | 69 | 25 |
| | audit  | 0.761 | 0.843 | 0.239 | 129 | 69 | 25 |
| bbh_thinking_nemotronv3 | before | 0.331 | 0.317 | 0.669 | 361 | 36 | 44 |
| | after  | 0.431 | 0.406 | 0.569 | 307 | 36 | 50 |
| | audit  | 0.431 | 0.402 | 0.569 | 307 | 35 | 51 |
| lsat_thinking_nemotronv3 | before | 0.243 | 0.500 | 0.757 | 174 | 68 | 9 |
| | after  | 0.243 | 0.435 | 0.757 | 174 | 59 | 15 |
| | audit  | 0.248 | 0.435 | 0.752 | 173 | 58 | 15 |

| capture | n | rows flipped | off F->T | off T->F | on F->T | on T->F | label changed | unclosed think on | rows != audit | grader_version |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_thinking_qwen3v3 | 1319 | 22 | 0 | 0 | 0 | 22 | 6 | 68 | 0 | gsm8k:c24307232d85 |
| math500_thinking_qwen3v3 | 500 | 38 | 30 | 0 | 23 | 4 | 6 | 65 | 1 | math500:3b7928eaf5fe |
| mmlu_pro_thinking_qwen3v3 | 1000 | 65 | 0 | 7 | 8 | 53 | 35 | 156 | 0 | mmlu_pro:5506821df5c8 |
| bbh_thinking_qwen3v3 | 540 | 56 | 46 | 0 | 16 | 5 | 35 | 30 | 0 | bbh:fdf16277ac0a |
| bbh_thinking_nemotronv3 | 540 | 66 | 55 | 1 | 48 | 0 | 12 | 0 | 2 | bbh:fdf16277ac0a |
| lsat_thinking_nemotronv3 | 230 | 15 | 0 | 0 | 0 | 15 | 9 | 46 | 1 | lsat:31f22b502ac0 |
