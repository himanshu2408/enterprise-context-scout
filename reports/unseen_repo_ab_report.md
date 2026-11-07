# Unseen-Repository Two-Arm A/B Benchmark Report

- **Generated (UTC):** `2026-11-07T00:13:21.506344+00:00`
- **Sample Size ($N$):** `30` real merged bug-fix PRs (`15` `encode/httpx`, `15` `pallets/click`)
- **Frozen Code Adapter SHA-256:** `261f1e4a9902cd2621b6db3ef631e2813380d68dfca9d8da800fefc8378b09e6` (zero retraining on unseen repos)
- **Frozen QLoRA Schema Validity on Unseen Repos:** `29 / 30` (`96.67%`)
- **Execution Model (Both Arms):** `mlx-community/gemma-4-26b-a4b-it-4bit`
- **Token Accounting Method:** `mlx_tokenizer_exact_bpe`
- **PostgreSQL 17 `FORCE RLS` Verified on Unseen Repo:** `True`

## 1. Executive Summary Table ($N=30$ Unseen PRs)

| Metric | Arm A: Direct Exploration (Multi-Turn `grep`/`read_file`) | Arm B: Enterprise Context Scout + 1-Shot Execution | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Input Tokens (Median / IQR)** | `26,509` (IQR `22,878`-`31,549`, mean `26,661`) | `4,898` (IQR `3,279`-`7,826`, mean `6,216`) | **`81.96%` median savings** (IQR `63.48%`-`87.12%`) |
| **Total Tokens (Median / IQR)** | `26,760` (IQR `23,393`-`31,796`, mean `27,073`) | `5,072` (IQR `3,486`-`8,085`, mean `6,449`) | **`81.07%` median savings** (IQR `63.65%`-`86.88%`) |
| **File `Recall@5`** | `27 / 30` (`90.00%`) | `30 / 30` (`100.00%`) | `+10.00%` (`+3` tasks) |
| **Symbol `Recall@5`** | `6 / 30` (`20.00%`) | `27 / 30` (`90.00%`) | `+70.00%` (`+21` tasks) |
| **Frozen QLoRA `CodeSlicePointer` Schema Validity** | N/A (Direct Tool Loop) | `29 / 30` (`96.67%`) | Zero retraining on unseen repos |
| **Worktree `pytest` Verification ($n=10$)** | `0 / 10` | `0 / 10` | `+0` tasks |
| **Wall-Clock Latency (Median / IQR)** | `18,756.9 ms` (IQR `15,909.1`-`24,163.8 ms`) | `9,616.0 ms` (IQR `6,477.4`-`17,914.7 ms`) | `48.7%` faster median |
| **Cumulative Cost at Gemini 3.8 Flash Rates (`$0.75`/`$3.75` per 1M)** | `$0.6463` | `$0.1661` | `$0.4802` saved across 30 tasks |

## 2. Honest Losses & Failure Taxonomy (Where Arm B Missed File or Symbol `Recall@5`)

| Task ID | Repo & PR | Ground-Truth File / Symbol | Arm A File Hit | Arm B File Hit | Arm B Symbol Hit | Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-04` | `encode/httpx#3371` | `httpx/_urlparse.py` (`httpx._urlparse.ParseResult`, `httpx._urlparse.urlparse`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-01` | `pallets/click#3901` | `src/click/core.py` (`click.core.Command`, `click.core.Command.get_short_help_str`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |
| `CLICK-14` | `pallets/click#3764` | `src/click/_termui_impl.py` (`click._termui_impl._tempfilepager`, `click._termui_impl._nullpager`) | `True` | `True` | `False` | `file_localized_but_target_symbol_outside_top5` |

## 3. Per-Task Breakdown ($N=30$)

| Task ID | Repo & PR | Arm A Tokens (In / Total) | Arm B Tokens (In / Total) | Input Savings | Arm A Recall (F / S) | Arm B Recall (F / S) | Worktree `pytest` (A / B) | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HTTPX-01` | [encode/httpx#3412](https://github.com/encode/httpx/pull/3412) | `21,635` / `22,426` | `4,427` / `5,067` | `79.5%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `HTTPX-02` | [encode/httpx#3376](https://github.com/encode/httpx/pull/3376) | `7,783` / `8,527` | `2,980` / `3,167` | `61.7%` | `False` / `False` | `True` / `True` | `False` / `False` | `ARM_B_ONLY_HIT` |
| `HTTPX-03` | [encode/httpx#3373](https://github.com/encode/httpx/pull/3373) | `23,888` / `24,133` | `1,589` / `1,685` | `93.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-04` | [encode/httpx#3371](https://github.com/encode/httpx/pull/3371) | `33,646` / `33,899` | `5,294` / `5,390` | `84.3%` | `True` / `True` | `True` / `False` | — | `BOTH_HIT` |
| `HTTPX-05` | [encode/httpx#3367](https://github.com/encode/httpx/pull/3367) | `30,469` / `30,946` | `2,777` / `3,417` | `90.9%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `HTTPX-06` | [encode/httpx#3335](https://github.com/encode/httpx/pull/3335) | `23,375` / `23,616` | `4,883` / `4,979` | `79.1%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-07` | [encode/httpx#3317](https://github.com/encode/httpx/pull/3317) | `30,832` / `31,075` | `5,118` / `5,214` | `83.4%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-08` | [encode/httpx#3312](https://github.com/encode/httpx/pull/3312) | `34,070` / `34,853` | `2,708` / `2,938` | `92.0%` | `False` / `False` | `True` / `True` | `False` / `False` | `ARM_B_ONLY_HIT` |
| `HTTPX-09` | [encode/httpx#3250](https://github.com/encode/httpx/pull/3250) | `29,335` / `29,583` | `4,220` / `4,312` | `85.6%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-10` | [encode/httpx#3187](https://github.com/encode/httpx/pull/3187) | `41,815` / `42,061` | `4,913` / `5,009` | `88.2%` | `False` / `False` | `True` / `True` | — | `ARM_B_ONLY_HIT` |
| `HTTPX-11` | [encode/httpx#3178](https://github.com/encode/httpx/pull/3178) | `37,568` / `37,815` | `3,595` / `3,691` | `90.4%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-12` | [encode/httpx#3116](https://github.com/encode/httpx/pull/3116) | `21,668` / `22,455` | `8,100` / `8,740` | `62.6%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `HTTPX-13` | [encode/httpx#3045](https://github.com/encode/httpx/pull/3045) | `26,667` / `26,916` | `4,335` / `4,429` | `83.7%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-14` | [encode/httpx#2999](https://github.com/encode/httpx/pull/2999) | `11,596` / `11,813` | `5,264` / `5,360` | `54.6%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `HTTPX-15` | [encode/httpx#2998](https://github.com/encode/httpx/pull/2998) | `22,325` / `22,573` | `2,842` / `2,936` | `87.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-01` | [pallets/click#3901](https://github.com/pallets/click/pull/3901) | `27,609` / `28,399` | `2,269` / `2,467` | `91.8%` | `True` / `False` | `True` / `False` | `False` / `False` | `BOTH_HIT` |
| `CLICK-02` | [pallets/click#3884](https://github.com/pallets/click/pull/3884) | `27,591` / `28,385` | `10,803` / `11,443` | `60.9%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-03` | [pallets/click#3866](https://github.com/pallets/click/pull/3866) | `43,147` / `43,396` | `17,209` / `17,305` | `60.1%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-04` | [pallets/click#3865](https://github.com/pallets/click/pull/3865) | `22,812` / `23,607` | `4,446` / `5,076` | `80.5%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-05` | [pallets/click#3861](https://github.com/pallets/click/pull/3861) | `26,351` / `26,603` | `16,795` / `16,891` | `36.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-06` | [pallets/click#3860](https://github.com/pallets/click/pull/3860) | `48,263` / `48,514` | `6,437` / `6,533` | `86.7%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-07` | [pallets/click#3821](https://github.com/pallets/click/pull/3821) | `24,631` / `24,883` | `8,357` / `8,452` | `66.1%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-08` | [pallets/click#3818](https://github.com/pallets/click/pull/3818) | `7,525` / `8,253` | `8,813` / `9,453` | `-17.1%` | `True` / `False` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-09` | [pallets/click#3805](https://github.com/pallets/click/pull/3805) | `6,840` / `7,597` | `16,758` / `17,397` | `-145.0%` | `True` / `True` | `True` / `True` | `False` / `False` | `BOTH_HIT` |
| `CLICK-10` | [pallets/click#3781](https://github.com/pallets/click/pull/3781) | `25,659` / `25,904` | `5,400` / `5,496` | `79.0%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-11` | [pallets/click#3777](https://github.com/pallets/click/pull/3777) | `25,719` / `25,969` | `7,006` / `7,102` | `72.8%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-12` | [pallets/click#3769](https://github.com/pallets/click/pull/3769) | `29,913` / `30,163` | `8,317` / `8,413` | `72.2%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-13` | [pallets/click#3767](https://github.com/pallets/click/pull/3767) | `31,788` / `32,037` | `3,070` / `3,166` | `90.3%` | `True` / `False` | `True` / `True` | — | `BOTH_HIT` |
| `CLICK-14` | [pallets/click#3764](https://github.com/pallets/click/pull/3764) | `32,226` / `32,475` | `4,578` / `4,674` | `85.8%` | `True` / `False` | `True` / `False` | — | `BOTH_HIT` |
| `CLICK-15` | [pallets/click#3681](https://github.com/pallets/click/pull/3681) | `23,078` / `23,322` | `3,174` / `3,270` | `86.2%` | `True` / `True` | `True` / `True` | — | `BOTH_HIT` |
