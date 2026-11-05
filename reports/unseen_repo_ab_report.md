# Unseen-Repository Two-Arm A/B Benchmark Report

- **Generated (UTC):** `2026-11-05T00:35:05.031633+00:00`
- **Sample Size ($N$):** `30` real merged bug-fix PRs (`15` `encode/httpx`, `15` `pallets/click`)
- **Frozen Code Adapter SHA-256:** `261f1e4a9902cd2621b6db3ef631e2813380d68dfca9d8da800fefc8378b09e6` (zero retraining on unseen repos)
- **Frozen QLoRA Schema Validity on Unseen Repos:** `4 / 30` (`13.33%`)
- **Execution Model (Both Arms):** `mlx-community/gemma-4-26b-a4b-it-4bit`
- **Token Accounting Method:** `mlx_tokenizer_exact_bpe`
- **PostgreSQL 17 `FORCE RLS` Verified on Unseen Repo:** `True`

## 1. Executive Summary Table ($N=30$ Unseen PRs)

| Metric | Arm A: Direct Exploration (Multi-Turn `grep`/`read_file`) | Arm B: Enterprise Context Scout + 1-Shot Execution | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Input Tokens (Median / IQR)** | `26,481` (IQR `22,850`-`31,521`, mean `26,633`) | `4,616` (IQR `2,690`-`7,384`, mean `6,939`) | **`81.81%` median savings** (IQR `69.80%`-`89.67%`) |
| **Total Tokens (Median / IQR)** | `26,732` (IQR `23,302`-`31,768`, mean `26,969`) | `4,856` (IQR `3,074`-`7,679`, mean `7,131`) | **`81.19%` median savings** (IQR `67.66%`-`89.26%`) |
| **File `Recall@5`** | `27 / 30` (`90.00%`) | `27 / 30` (`90.00%`) | `+0.00%` (`+0` tasks) |
| **Symbol `Recall@5`** | `6 / 30` (`20.00%`) | `23 / 30` (`76.67%`) | `+56.67%` (`+17` tasks) |
| **Frozen QLoRA `CodeSlicePointer` Schema Validity** | N/A (Direct Tool Loop) | `4 / 30` (`13.33%`) | Zero retraining on unseen repos |
| **Worktree `pytest` Verification ($n=10$)** | `0 / 10` | `0 / 10` | `+0` tasks |
| **Wall-Clock Latency (Median / IQR)** | `16,125.3 ms` (IQR `13,516.4`-`18,903.1 ms`) | `5,508.9 ms` (IQR `4,390.6`-`8,454.9 ms`) | `65.8%` faster median |
| **Cumulative Cost at Gemini 3.8 Flash Rates (`$0.75`/`$3.75` per 1M)** | `$0.6371` | `$0.1777` | `$0.4594` saved across 30 tasks |

## 2. Honest Losses & Failure Taxonomy (Where Arm B Missed File or Symbol `Recall@5`)

| Task ID | Repo & PR | Ground-Truth File / Symbol | Arm A File Hit | Arm B File Hit | Arm B Symbol Hit | Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-01` | `encode/httpx#3412` | `httpx/_decoders.py` (`httpx._decoders.ZStandardDecoder`, `httpx._decoders.ZStandardDecoder.__init__`) | `True` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-02` | `encode/httpx#3376` | `httpx/_urls.py` (`httpx._urls.QueryParams`, `httpx._urls.QueryParams.__len__`) | `False` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-04` | `encode/httpx#3371` | `httpx/_urlparse.py` (`httpx._urlparse.ParseResult`, `httpx._urlparse.urlparse`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `HTTPX-05` | `encode/httpx#3367` | `httpx/_content.py` (`httpx._content.encode_json`) | `True` | `False` | `False` | `ast_lexical_retrieval_missed_target_file` |
| `HTTPX-09` | `encode/httpx#3250` | `httpx/_urlparse.py` (`httpx._urlparse.urlparse`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-02` | `pallets/click#3884` | `src/click/core.py` (`click.core.Option`, `click.core.Option.get_error_hint`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-03` | `pallets/click#3866` | `src/click/core.py` (`click.core._format_deprecated_label`, `click.core._format_deprecated_suffix`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |

## 3. Per-Task Breakdown ($N=30$)

| Task ID | Repo & PR | Arm A Tokens (In / Total) | Arm B Tokens (In / Total) | Input Savings | Arm A Recall (F / S) | Arm B Recall (F / S) | Worktree `pytest` (A / B) | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-01` | [encode/httpx#3412](https://github.com/encode/httpx/pull/3412) | `21,607` / `22,117` | `2,670` / `3,054` | `87.6%` | `True` / `False` | `False` / `False` | `False` / `False` | `ARM_A_ONLY_HIT` |
| `HTTPX-02` | [encode/httpx#3376](https://github.com/encode/httpx/pull/3376) | `7,755` / `8,243` | `2,391` / `2,774` | `69.2%` | `False` / `False` | `False` / `False` | `False` / `False` | `BOTH_MISSED` |
| `HTTPX-03` | [encode/httpx#3373](https://github.com/encode/httpx/pull/3373) | `23,860` / `24,104` | `1,816` / `1,912` | `92.4%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-04` | [encode/httpx#3371](https://github.com/encode/httpx/pull/3371) | `33,618` / `33,871` | `7,392` / `7,488` | `78.0%` | `True` / `True` | `True` / `False` | — | `BOTH_HIT` |
| `HTTPX-05` | [encode/httpx#3367](https://github.com/encode/httpx/pull/3367) | `30,441` / `30,962` | `2,819` / `3,203` | `90.7%` | `True` / `False` | `False` / `False` | `False` / `False` | `ARM_A_ONLY_HIT` |
| `HTTPX-06` | [encode/httpx#3335](https://github.com/encode/httpx/pull/3335) | `23,347` / `23,588` | `4,551` / `4,647` | `80.5%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-07` | [encode/httpx#3317](https://github.com/encode/httpx/pull/3317) | `30,804` / `31,047` | `7,307` / `7,403` | `76.3%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-08` | [encode/httpx#3312](https://github.com/encode/httpx/pull/3312) | `34,042` / `34,571` | `1,305` / `1,689` | `96.2%` | `False` / `False` | `True` / `True` | `False` / `False` | `ARM_B_ONLY_HIT` |
| `HTTPX-09` | [encode/httpx#3250](https://github.com/encode/httpx/pull/3250) | `29,307` / `29,555` | `8,294` / `8,390` | `71.7%` | `True` / `False` | `True` / `False` | — | `BOTH_HIT` |
| `HTTPX-10` | [encode/httpx#3187](https://github.com/encode/httpx/pull/3187) | `41,787` / `42,033` | `11,434` / `11,530` | `72.6%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-11` | [encode/httpx#3178](https://github.com/encode/httpx/pull/3178) | `37,540` / `37,787` | `1,688` / `1,784` | `95.5%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-12` | [encode/httpx#3116](https://github.com/encode/httpx/pull/3116) | `21,640` / `22,171` | `7,361` / `7,743` | `66.0%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `HTTPX-13` | [encode/httpx#3045](https://github.com/encode/httpx/pull/3045) | `26,639` / `26,888` | `4,313` / `4,407` | `83.8%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-14` | [encode/httpx#2999](https://github.com/encode/httpx/pull/2999) | `11,568` / `11,785` | `7,301` / `7,397` | `36.9%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-15` | [encode/httpx#2998](https://github.com/encode/httpx/pull/2998) | `22,297` / `22,545` | `3,624` / `3,720` | `83.8%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-01` | [pallets/click#3901](https://github.com/pallets/click/pull/3901) | `27,581` / `28,115` | `4,681` / `5,064` | `83.0%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-02` | [pallets/click#3884](https://github.com/pallets/click/pull/3884) | `27,563` / `28,101` | `14,309` / `14,693` | `48.1%` | `True` / `False` | `True` / `False` | `False` / `False` | `BOTH_HIT` |
| `CLICK-03` | [pallets/click#3866](https://github.com/pallets/click/pull/3866) | `43,119` / `43,369` | `2,363` / `2,459` | `94.5%` | `True` / `False` | `True` / `False` | — | `BOTH_HIT` |
| `CLICK-04` | [pallets/click#3865](https://github.com/pallets/click/pull/3865) | `22,784` / `23,323` | `2,748` / `3,132` | `87.9%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-05` | [pallets/click#3861](https://github.com/pallets/click/pull/3861) | `26,323` / `26,575` | `26,261` / `26,357` | `0.2%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-06` | [pallets/click#3860](https://github.com/pallets/click/pull/3860) | `48,235` / `48,486` | `5,704` / `5,800` | `88.2%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-07` | [pallets/click#3821](https://github.com/pallets/click/pull/3821) | `24,603` / `24,855` | `17,769` / `17,865` | `27.8%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-08` | [pallets/click#3818](https://github.com/pallets/click/pull/3818) | `7,497` / `7,969` | `13,374` / `13,758` | `-78.4%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-09` | [pallets/click#3805](https://github.com/pallets/click/pull/3805) | `6,812` / `7,313` | `25,415` / `25,799` | `-273.1%` | `True` / `True` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-10` | [pallets/click#3781](https://github.com/pallets/click/pull/3781) | `25,631` / `25,876` | `4,978` / `5,074` | `80.6%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-11` | [pallets/click#3777](https://github.com/pallets/click/pull/3777) | `25,691` / `25,941` | `5,732` / `5,828` | `77.7%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-12` | [pallets/click#3769](https://github.com/pallets/click/pull/3769) | `29,885` / `30,135` | `1,148` / `1,244` | `96.2%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-13` | [pallets/click#3767](https://github.com/pallets/click/pull/3767) | `31,760` / `32,009` | `3,123` / `3,219` | `90.2%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-14` | [pallets/click#3764](https://github.com/pallets/click/pull/3764) | `32,198` / `32,447` | `4,523` / `4,619` | `86.0%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-15` | [pallets/click#3681](https://github.com/pallets/click/pull/3681) | `23,050` / `23,295` | `1,782` / `1,878` | `92.3%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
