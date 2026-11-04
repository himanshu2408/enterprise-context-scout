# Unseen-Repository Two-Arm A/B Benchmark Report

- **Generated (UTC):** `2026-11-04T23:01:30.050933+00:00`
- **Sample Size ($N$):** `30` real merged bug-fix PRs (`15` `encode/httpx`, `15` `pallets/click`)
- **Frozen Code Adapter SHA-256:** `261f1e4a9902cd2621b6db3ef631e2813380d68dfca9d8da800fefc8378b09e6` (zero retraining on unseen repos)
- **Frozen QLoRA Schema Validity on Unseen Repos:** `29 / 30` (`96.67%`)
- **Execution Model (Both Arms):** `mlx-community/gemma-4-26b-a4b-it-4bit`
- **Token Accounting Method:** `mlx_tokenizer_exact_bpe`
- **PostgreSQL 17 `FORCE RLS` Verified on Unseen Repo:** `True`

## 1. Executive Summary Table ($N=30$ Unseen PRs)

| Metric | Arm A: Direct Exploration (Multi-Turn `grep`/`read_file`) | Arm B: Enterprise Context Scout + 1-Shot Execution | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Input Tokens (Median / IQR)** | `27,498` (IQR `21,436`–`36,900`, mean `27,928`) | `4,798` (IQR `2,650`–`7,350`, mean `6,927`) | **`81.28%` median savings** (IQR `60.78%`–`89.56%`) |
| **Total Tokens (Median / IQR)** | `27,753` (IQR `21,670`–`37,141`, mean `28,156`) | `4,862` (IQR `2,714`–`7,414`, mean `6,991`) | **`81.20%` median savings** (IQR `60.91%`–`89.42%`) |
| **File `Recall@5`** | `12 / 30` (`40.00%`) | `27 / 30` (`90.00%`) | `+50.00%` (`+15` tasks) |
| **Symbol `Recall@5`** | `1 / 30` (`3.33%`) | `23 / 30` (`76.67%`) | `+73.34%` (`+22` tasks) |
| **Frozen QLoRA `CodeSlicePointer` Schema Validity** | N/A (Direct Tool Loop) | `29 / 30` (`96.67%`) | Zero retraining on unseen repos |
| **Worktree `pytest` Verification ($n=10$)** | `0 / 10` | `4 / 10` | `+4` tasks |
| **Wall-Clock Latency (Median / IQR)** | `5,304.1 ms` (IQR `4,264.9`–`5,797.2 ms`) | `4,818.9 ms` (IQR `3,556.8`–`5,909.9 ms`) | `9.1%` faster median |
| **Cumulative Cost at Gemini 3.8 Flash Rates (`$0.75`/`$3.75` per 1M)** | `$0.6540` | `$0.1631` | `$0.4910` saved across 30 tasks |

## 2. Honest Losses & Failure Taxonomy (Where Arm B Missed File or Symbol `Recall@5`)

| Task ID | Repo & PR | Ground-Truth File / Symbol | Arm A File Hit | Arm B File Hit | Arm B Symbol Hit | Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-01` | `encode/httpx#3412` | `httpx/_decoders.py` (`httpx._decoders.ZStandardDecoder`, `httpx._decoders.ZStandardDecoder.__init__`) | `True` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-02` | `encode/httpx#3376` | `httpx/_urls.py` (`httpx._urls.QueryParams`, `httpx._urls.QueryParams.__len__`) | `False` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-04` | `encode/httpx#3371` | `httpx/_urlparse.py` (`httpx._urlparse.ParseResult`, `httpx._urlparse.urlparse`) | `False` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `HTTPX-05` | `encode/httpx#3367` | `httpx/_content.py` (`httpx._content.encode_json`) | `False` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-09` | `encode/httpx#3250` | `httpx/_urlparse.py` (`httpx._urlparse.urlparse`) | `False` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-02` | `pallets/click#3884` | `src/click/core.py` (`click.core.Option`, `click.core.Option.get_error_hint`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-03` | `pallets/click#3866` | `src/click/core.py` (`click.core._format_deprecated_label`, `click.core._format_deprecated_suffix`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |

## 3. Per-Task Breakdown ($N=30$)

| Task ID | Repo & PR | Arm A Tokens (In / Total) | Arm B Tokens (In / Total) | Input Savings | Arm A Recall (F / S) | Arm B Recall (F / S) | Worktree `pytest` (A / B) | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-01` | [encode/httpx#3412](https://github.com/encode/httpx/pull/3412) | `25,873` / `26,112` | `2,631` / `2,695` | `89.8%` | `True` / `False` | `False` / `False` | `False` / `False` | `ARM_A_ONLY_HIT` |
| `HTTPX-02` | [encode/httpx#3376](https://github.com/encode/httpx/pull/3376) | `7,644` / `7,791` | `2,352` / `2,415` | `69.2%` | `False` / `False` | `False` / `False` | `False` / `False` | `BOTH_MISSED` |
| `HTTPX-03` | [encode/httpx#3373](https://github.com/encode/httpx/pull/3373) | `26,446` / `26,685` | `1,777` / `1,841` | `93.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-04` | [encode/httpx#3371](https://github.com/encode/httpx/pull/3371) | `27,958` / `28,203` | `7,353` / `7,416` | `73.7%` | `False` / `False` | `True` / `False` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-05` | [encode/httpx#3367](https://github.com/encode/httpx/pull/3367) | `24,764` / `24,996` | `2,780` / `2,844` | `88.8%` | `False` / `False` | `False` / `False` | `False` / `False` | `BOTH_MISSED` |
| `HTTPX-06` | [encode/httpx#3335](https://github.com/encode/httpx/pull/3335) | `6,933` / `7,084` | `4,512` / `4,576` | `34.9%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-07` | [encode/httpx#3317](https://github.com/encode/httpx/pull/3317) | `36,940` / `37,181` | `7,268` / `7,332` | `80.3%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-08` | [encode/httpx#3312](https://github.com/encode/httpx/pull/3312) | `20,642` / `20,879` | `1,266` / `1,330` | `93.9%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `HTTPX-09` | [encode/httpx#3250](https://github.com/encode/httpx/pull/3250) | `20,852` / `21,090` | `8,255` / `8,319` | `60.4%` | `False` / `False` | `True` / `False` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-10` | [encode/httpx#3187](https://github.com/encode/httpx/pull/3187) | `13,500` / `13,702` | `11,395` / `11,459` | `15.6%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-11` | [encode/httpx#3178](https://github.com/encode/httpx/pull/3178) | `34,112` / `34,343` | `1,649` / `1,713` | `95.2%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-12` | [encode/httpx#3116](https://github.com/encode/httpx/pull/3116) | `40,183` / `40,417` | `7,343` / `7,407` | `81.7%` | `False` / `False` | `True` / `True` | `False` / `False` | `ARM_B_ONLY_HIT` |
| `HTTPX-13` | [encode/httpx#3045](https://github.com/encode/httpx/pull/3045) | `23,188` / `23,408` | `4,274` / `4,338` | `81.6%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-14` | [encode/httpx#2999](https://github.com/encode/httpx/pull/2999) | `31,080` / `31,316` | `7,262` / `7,326` | `76.6%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-15` | [encode/httpx#2998](https://github.com/encode/httpx/pull/2998) | `36,696` / `36,937` | `3,426` / `3,490` | `90.7%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `CLICK-01` | [pallets/click#3901](https://github.com/pallets/click/pull/3901) | `20,399` / `20,641` | `4,658` / `4,722` | `77.2%` | `True` / `False` | `True` / `True` | `False` / `True` | `BOTH_HIT` |
| `CLICK-02` | [pallets/click#3884](https://github.com/pallets/click/pull/3884) | `37,466` / `37,716` | `14,270` / `14,334` | `61.9%` | `True` / `False` | `True` / `False` | `False` / `False` | `BOTH_HIT` |
| `CLICK-03` | [pallets/click#3866](https://github.com/pallets/click/pull/3866) | `38,670` / `38,911` | `2,324` / `2,388` | `94.0%` | `True` / `False` | `True` / `False` | — | `BOTH_HIT` |
| `CLICK-04` | [pallets/click#3865](https://github.com/pallets/click/pull/3865) | `6,119` / `6,263` | `2,709` / `2,773` | `55.7%` | `False` / `False` | `True` / `True` | `False` / `True` | `ARM_B_ONLY_HIT` |
| `CLICK-05` | [pallets/click#3861](https://github.com/pallets/click/pull/3861) | `44,246` / `44,493` | `26,222` / `26,284` | `40.7%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `CLICK-06` | [pallets/click#3860](https://github.com/pallets/click/pull/3860) | `47,810` / `48,064` | `5,791` / `5,855` | `87.9%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `CLICK-07` | [pallets/click#3821](https://github.com/pallets/click/pull/3821) | `27,039` / `27,303` | `17,730` / `17,794` | `34.4%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-08` | [pallets/click#3818](https://github.com/pallets/click/pull/3818) | `31,800` / `32,051` | `13,335` / `13,399` | `58.1%` | `False` / `False` | `True` / `True` | `False` / `True` | `ARM_B_ONLY_HIT` |
| `CLICK-09` | [pallets/click#3805](https://github.com/pallets/click/pull/3805) | `26,912` / `27,160` | `25,376` / `25,440` | `5.7%` | `False` / `False` | `True` / `True` | `False` / `True` | `ARM_B_ONLY_HIT` |
| `CLICK-10` | [pallets/click#3781](https://github.com/pallets/click/pull/3781) | `38,768` / `39,009` | `4,939` / `5,003` | `87.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-11` | [pallets/click#3777](https://github.com/pallets/click/pull/3777) | `31,464` / `31,714` | `5,985` / `6,049` | `81.0%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-12` | [pallets/click#3769](https://github.com/pallets/click/pull/3769) | `6,233` / `6,389` | `1,109` / `1,173` | `82.2%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `CLICK-13` | [pallets/click#3767](https://github.com/pallets/click/pull/3767) | `40,978` / `41,215` | `3,084` / `3,148` | `92.5%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-14` | [pallets/click#3764](https://github.com/pallets/click/pull/3764) | `36,781` / `37,022` | `5,006` / `5,070` | `86.4%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `CLICK-15` | [pallets/click#3681](https://github.com/pallets/click/pull/3681) | `26,344` / `26,589` | `1,743` / `1,807` | `93.4%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
