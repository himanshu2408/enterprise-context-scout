"""Unseen-Repository Bug-Fix Benchmark Dataset (N=30 Real Merged PRs).

Contains 30 real merged bug-fix pull requests extracted from two unseen public Python
repositories not present anywhere in training or distillation:
- encode/httpx (15 tasks: HTTPX-01 .. HTTPX-15)
- pallets/click (15 tasks: CLICK-01 .. CLICK-15)

Each task pins the exact parent_commit_sha (pre-fix repository state), the GitHub
issue / PR problem description, and the ground-truth files and AST symbols modified
in merge_commit_sha.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

FROZEN_CODE_ADAPTER_SHA256: Final[str] = (
    "261f1e4a9902cd2621b6db3ef631e2813380d68dfca9d8da800fefc8378b09e6"
)
FROZEN_QUANT_ADAPTER_SHA256: Final[str] = (
    "ec5428a8f5cfe6604dc5ecb7e9e5aec04fed92fa18dfb60176773d61fbc469e8"
)


class UnseenRepoBugTask(BaseModel):
    """Single real merged bug-fix PR task on an unseen repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=3)
    repo_slug: Literal["encode/httpx", "pallets/click"]
    pr_number: int = Field(..., ge=1)
    pr_url: str = Field(..., min_length=10)
    merge_commit_sha: str = Field(..., min_length=40, max_length=40)
    parent_commit_sha: str = Field(..., min_length=40, max_length=40)
    title: str = Field(..., min_length=5)
    problem_statement: str = Field(..., min_length=10)
    ground_truth_files: tuple[str, ...] = Field(..., min_length=1)
    ground_truth_symbols: tuple[str, ...] = Field(..., min_length=1)
    test_files: tuple[str, ...] = Field(default_factory=tuple)
    verify_with_pytest: bool = Field(default=False)


class UnseenRepoBenchmarkSuite(BaseModel):
    """Complete 30-task unseen-repository benchmark suite with frozen adapter hashes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_version: str = "1.0.0"
    frozen_code_adapter_sha256: str = FROZEN_CODE_ADAPTER_SHA256
    frozen_quant_adapter_sha256: str = FROZEN_QUANT_ADAPTER_SHA256
    tasks: tuple[UnseenRepoBugTask, ...]

    @property
    def httpx_tasks(self) -> list[UnseenRepoBugTask]:
        return [t for t in self.tasks if t.repo_slug == "encode/httpx"]

    @property
    def click_tasks(self) -> list[UnseenRepoBugTask]:
        return [t for t in self.tasks if t.repo_slug == "pallets/click"]

    @property
    def pytest_verification_tasks(self) -> list[UnseenRepoBugTask]:
        return [t for t in self.tasks if t.verify_with_pytest]


def verify_frozen_adapter_integrity(repo_root: Path = Path(".")) -> dict[str, Any]:
    """Verify that QLoRA adapters match their pre-benchmark SHA-256 digests."""
    code_adapter = (
        repo_root
        / "domains"
        / "trading_code_scout"
        / "adapters"
        / "gemma4_26b_code_scout"
        / "adapters.safetensors"
    )
    quant_adapter = (
        repo_root
        / "domains"
        / "quant_research_scout"
        / "adapters"
        / "gemma4_26b_quant_research"
        / "adapters.safetensors"
    )
    if not code_adapter.exists() or not quant_adapter.exists():
        raise FileNotFoundError("Frozen QLoRA adapter safetensors not found under domains/.")

    code_sha = hashlib.sha256(code_adapter.read_bytes()).hexdigest()
    quant_sha = hashlib.sha256(quant_adapter.read_bytes()).hexdigest()
    if code_sha != FROZEN_CODE_ADAPTER_SHA256:
        raise ValueError(
            f"Frozen adapter SHA-256 mismatch for trading_code_scout: "
            f"{code_sha} != {FROZEN_CODE_ADAPTER_SHA256}"
        )
    if quant_sha != FROZEN_QUANT_ADAPTER_SHA256:
        raise ValueError(
            f"Frozen adapter SHA-256 mismatch for quant_research_scout: "
            f"{quant_sha} != {FROZEN_QUANT_ADAPTER_SHA256}"
        )
    return {
        "verified": True,
        "code_adapter_sha256": code_sha,
        "quant_adapter_sha256": quant_sha,
        "trading_code_scout": True,
        "quant_research_scout": True,
    }


UNSEEN_REPO_TASKS: Final[tuple[UnseenRepoBugTask, ...]] = (
    UnseenRepoBugTask(
        task_id="HTTPX-01",
        repo_slug="encode/httpx",
        pr_number=3412,
        pr_url="https://github.com/encode/httpx/pull/3412",
        merge_commit_sha="47f4a96ffaaaa07dca1614409549b5d7a6e7af49",
        parent_commit_sha="189fc4bcbe5f314128775dec66a616ac9a31ad48",
        title=("Handle empty zstd responses"),
        problem_statement=("Handle empty zstd responses\n\nCloses #3411."),
        ground_truth_files=("httpx/_decoders.py",),
        ground_truth_symbols=(
            "httpx._decoders.ZStandardDecoder",
            "httpx._decoders.ZStandardDecoder.__init__",
            "httpx._decoders.ZStandardDecoder.decode",
            "httpx._decoders.ZStandardDecoder.flush",
        ),
        test_files=("tests/test_decoders.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-02",
        repo_slug="encode/httpx",
        pr_number=3376,
        pr_url="https://github.com/encode/httpx/pull/3376",
        merge_commit_sha="5dda2aa30692e8529555f57efa7bcd203a7d5ac1",
        parent_commit_sha="544038155382473ce87b17ef2238c96e48dee700",
        title=("Just use default `safe=...` characters for `urlescape`"),
        problem_statement=(
            "Just use default `safe=...` characters for `urlescape`\n\nDon't tr"
            "y to be clever about which `safe=...` characters to use when cal"
            "ling `urlescape()`."
        ),
        ground_truth_files=("httpx/_urls.py",),
        ground_truth_symbols=(
            "httpx._urls.QueryParams",
            "httpx._urls.QueryParams.__len__",
            "httpx._urls.QueryParams.__str__",
        ),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-03",
        repo_slug="encode/httpx",
        pr_number=3373,
        pr_url="https://github.com/encode/httpx/pull/3373",
        merge_commit_sha="ba2e51215e4e619540f2a24282b84ed8a738e6fd",
        parent_commit_sha="d293374b6603816420a2bb8e7f4fe5a5f09cbfcc",
        title=("Review urlescape percent-safe set, and use + behavior for form spaces."),
        problem_statement=(
            "Review urlescape percent-safe set, and use + behavior for form s"
            "paces.\n\nPrompted by #3369\n\n* Review the percent-safe set that we"
            " should use for urlescaping form values.\n* Use the stdlib `urles"
            "cape`, including form query parameters using `+` for spaces."
        ),
        ground_truth_files=(
            "httpx/_urlparse.py",
            "httpx/_urls.py",
        ),
        ground_truth_symbols=(
            "httpx._urlparse.percent_encoded",
            "httpx._urlparse.quote",
            "httpx._urlparse.urlencode",
            "httpx._urls.URL",
            "httpx._urls.QueryParams",
            "httpx._urls.QueryParams.__str__",
        ),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-04",
        repo_slug="encode/httpx",
        pr_number=3371,
        pr_url="https://github.com/encode/httpx/pull/3371",
        merge_commit_sha="d293374b6603816420a2bb8e7f4fe5a5f09cbfcc",
        parent_commit_sha="489fef48ba8619c7604376165c7e8007398e1b5d",
        title=("Review URL percent escaping sets, from whatwg."),
        problem_statement=(
            "Review URL percent escaping sets, from whatwg.\n\nReview and updat"
            "e which characters should be percent escaped when parsing & norm"
            "alising URLs.\nBased on the URL living standard... https://url.sp"
            "ec.whatwg.org/\n\nPrompted by https://github.com/encode/httpx/disc"
            "ussions/3360"
        ),
        ground_truth_files=("httpx/_urlparse.py",),
        ground_truth_symbols=(
            "httpx._urlparse.ParseResult",
            "httpx._urlparse.urlparse",
            "httpx._urlparse.encode_host",
        ),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-05",
        repo_slug="encode/httpx",
        pr_number=3367,
        pr_url="https://github.com/encode/httpx/pull/3367",
        merge_commit_sha="9fd6f0ca6616d0310a3ee0b0c6ef509a97995797",
        parent_commit_sha="8e36f2bc685dfbe43cd7503bc1c422a6ed6e05a5",
        title=("Ensure JSON representation is compact. #3363"),
        problem_statement=(
            "Ensure JSON representation is compact. #3363\n\n# Summary\n\nthis PR"
            " solves the issues mentionned here #3363\n\n> The following relate"
            "d issues could all be resolved together...\n> \n> We should use en"
            "sure_ascii=False for more compact text representations.\n> We sho"
            "uld use separators = (',', ':') for more compact list and object"
            " representations.\n> We should use allow_nan=False to disallow in"
            "valid Infinity and NaN representations.\n\n# Checklist\n\n- [x] I un"
            "derstand that this PR may be closed in case there was no previou"
            "s discussion. (This doesn't apply to typos!)\n- [x] I've added a "
            "test for each change that was introduced, and I tried as much as"
            " possible to make a single atomic change.\n- [x] I've updated the"
            " documentation accordingly."
        ),
        ground_truth_files=("httpx/_content.py",),
        ground_truth_symbols=("httpx._content.encode_json",),
        test_files=(
            "tests/client/test_auth.py",
            "tests/models/test_requests.py",
            "tests/models/test_responses.py",
            "tests/models/test_whatwg.py",
            "tests/test_content.py",
            "tests/test_main.py",
        ),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-06",
        repo_slug="encode/httpx",
        pr_number=3335,
        pr_url="https://github.com/encode/httpx/pull/3335",
        merge_commit_sha="1aa0a14ec1fc744f6b362277d600955feee025e1",
        parent_commit_sha="2763690e1b2c0f72505397f34fc3c7dc2971955a",
        title=("Drop overloaded usage of 'verify' and 'cert'."),
        problem_statement=(
            "Drop overloaded usage of 'verify' and 'cert'.\n\nThis PR follows u"
            "p on #3022, and...\n\n* Drops the overloaded style of `verify=str`"
            ", in favor of the existing `.load_verify_locations(...)` API.\n* "
            "Drops the `cert` parameter, in favor of the existing `.load_cert"
            "_chain(...)` API.\n* Simplifies the internal `SSLContext` setup ["
            "in line with stdlib](https://github.com/python/cpython/blob/8b9a"
            "8e0e082f849f6db03b2ed9074a86813b3b77/Lib/ssl.py#L735-L757).\n* Re"
            "moves the certificate debug logging, which was odd as the single"
            " debug case in `httpx` and which is defunct now that we're using"
            " a more explicit style.\n\nAt this point we have a nice simple `ht"
            "tpx.SSLContext(verify: bool)` API, with an uncomplicated impleme"
            "ntation...."
        ),
        ground_truth_files=(
            "httpx/_client.py",
            "httpx/_compat.py",
            "httpx/_config.py",
            "httpx/_transports/default.py",
            "httpx/_types.py",
        ),
        ground_truth_symbols=(
            "httpx._client.Client",
            "httpx._client.Client._init_transport",
            "httpx._client.Client._init_proxy_transport",
            "httpx._config.UnsetType",
            "httpx._config.SSLContext",
            "httpx._config.SSLContext.__init__",
        ),
        test_files=(
            "tests/test_config.py",
            "tests/test_utils.py",
        ),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-07",
        repo_slug="encode/httpx",
        pr_number=3317,
        pr_url="https://github.com/encode/httpx/pull/3317",
        merge_commit_sha="a18f79610141500cf540b2cf4783559731109d7d",
        parent_commit_sha="f06171fd5a86cf89387f4ee860ea0da207160418",
        title=("Drop unneccessary 'URL.raw' property"),
        problem_statement=(
            "Drop unneccessary 'URL.raw' property\n\nRemove the unneccessary `U"
            "RL.raw` property.\n\nPreviously dropped in 0.23.1, and then reintr"
            "oduced as an interim measure. See #2461 and #2462.\n\nRelated: #33"
            "14, #3315"
        ),
        ground_truth_files=(
            "httpx/_types.py",
            "httpx/_urls.py",
        ),
        ground_truth_symbols=(
            "httpx._types.SyncByteStream",
            "httpx._urls.URL",
            "httpx._urls.URL.raw",
        ),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-08",
        repo_slug="encode/httpx",
        pr_number=3312,
        pr_url="https://github.com/encode/httpx/pull/3312",
        merge_commit_sha="49d74a2e7f773f88a84dd4af900d7a8543636ae2",
        parent_commit_sha="2e01aa0075327c5c30632207e8b7a1961dda5303",
        title=("Clarified error when header value is None"),
        problem_statement=(
            "Clarified error when header value is None\n\nPreviously, when the "
            'header value was None, a misleading error was thrown: "Attribute'
            "Error: Object 'NoneType' does not have attribute 'encode'.\" Now "
            "it looks better.\n\n![image](https://github.com/user-attachments/a"
            "ssets/ec12330f-aa7e-4084-8c81-c89aef8fdea9)"
        ),
        ground_truth_files=("httpx/_utils.py",),
        ground_truth_symbols=("httpx._utils.normalize_header_value",),
        test_files=("tests/client/test_headers.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-09",
        repo_slug="encode/httpx",
        pr_number=3250,
        pr_url="https://github.com/encode/httpx/pull/3250",
        merge_commit_sha="7c0cda153d301bde9a011e1dd7157d7e2b20889d",
        parent_commit_sha="beb501fc28cb24ef7cd584159bc6ac55d2244dea",
        title=("Improve InvalidURL error message."),
        problem_statement=(
            "Improve InvalidURL error message.\n\nPrompted by https://github.co"
            "m/encode/httpx/discussions/3248.\n\n*before*...\n\n```python\n>>> imp"
            'ort httpx\n>>> httpx.get("https://www.example.com\\n")\nTraceback ('
            'most recent call last):\n  ...\n  File "/workspace/GitHub/encode/h'
            'ttpx/httpx/_urlparse.py", line 163, in urlparse\n    raise Invali'
            "dURL(error)\nhttpx.InvalidURL: Invalid non-printable ASCII charac"
            "ter in URL.\n```\n\n*after*...\n\n```python\n>>> import httpx\n>>> http"
            'x.get("https://www.example.com\\n")\nTraceback (most recent call l'
            'ast):\n  ...\n  File "/workspace/GitHub/encode/httpx/httpx/_urlpar'
            'se.py", line 168, in urlparse\n    raise InvalidURL(error)\nhttpx.'
            "InvalidURL: Invalid non-printable ASCII character in URL, '\\n' a"
            "t position 23...."
        ),
        ground_truth_files=("httpx/_urlparse.py",),
        ground_truth_symbols=("httpx._urlparse.urlparse",),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-10",
        repo_slug="encode/httpx",
        pr_number=3187,
        pr_url="https://github.com/encode/httpx/pull/3187",
        merge_commit_sha="a7092af2fda78d92daaad7627e4cf0cf5e94b019",
        parent_commit_sha="be56b747350008b7b0456242691919f9086c0f32",
        title=("Resolve queryparam quoting"),
        problem_statement=(
            "Resolve queryparam quoting\n\n* Percent-encoding in `params={}` sh"
            "ould match browser form behavior. (Always percent-encode.)\n* Per"
            "cent-encoding in `url={}` should match browser URL bar behavior."
            " (Percent-encode, preserving existing escape sequences.)\n\nCloses"
            " #3135.\nCloses #3140."
        ),
        ground_truth_files=("httpx/_urlparse.py",),
        ground_truth_symbols=(
            "httpx._urlparse.percent_encode",
            "httpx._urlparse.is_safe",
            "httpx._urlparse.percent_encoded",
        ),
        test_files=("tests/models/test_url.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-11",
        repo_slug="encode/httpx",
        pr_number=3178,
        pr_url="https://github.com/encode/httpx/pull/3178",
        merge_commit_sha="12be5c44caa7a34bfe19f82eace5150ddeba2612",
        parent_commit_sha="e9cabc8e1deb6548c5c5e2f453e343d39a2a5321",
        title=("add socks5h proxy support"),
        problem_statement=(
            "add socks5h proxy support\n\nIn some case socks5h is needed to res"
            "olve the hostname.\n\n\n\n# Summary\n\nadd socks5h proxy support.\n\n\n# "
            "Checklist\n\n- [x] I understand that this PR may be closed in case"
            " there was no previous discussion. (This doesn't apply to typos!"
            ")\n- [x] I've added a test for each change that was introduced, a"
            "nd I tried as much as possible to make a single atomic change.\n-"
            " [x] I've updated the documentation accordingly."
        ),
        ground_truth_files=(
            "httpx/_config.py",
            "httpx/_transports/default.py",
        ),
        ground_truth_symbols=(
            "httpx._config.Proxy",
            "httpx._config.Proxy.__init__",
            "httpx._transports.default.HTTPTransport",
            "httpx._transports.default.HTTPTransport.__init__",
            "httpx._transports.default.AsyncHTTPTransport",
            "httpx._transports.default.AsyncHTTPTransport.__init__",
        ),
        test_files=("tests/client/test_proxies.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-12",
        repo_slug="encode/httpx",
        pr_number=3116,
        pr_url="https://github.com/encode/httpx/pull/3116",
        merge_commit_sha="6d852d319acd5d97caf14037dff15ede04b37542",
        parent_commit_sha="df5345140e09ac6c2de0d9589bcd6f3e31c6aa3f",
        title=("Fix client.send() timeout when used with a new Request instance"),
        problem_statement=(
            "Fix client.send() timeout when used with a new Request instance\n"
            "\n# Summary\n\nDiscussion here https://github.com/encode/httpx/disc"
            "ussions/3110\n\nOn `httpx.client.send()`, now the timeout of the r"
            "equest is set if not set already.\nThis is for cases in which `ht"
            "tpx.client.send()` is used with `httpx.Request()` instead of `cl"
            "ient.build_request()`\n\nI expected timeout tests for the non-asyn"
            "c client too. But since there are only async tests I didn't crea"
            "te a non-async test.\n\nI didn't think a documentation change was "
            "needed since this is (for me) the expect behavior.\n\n# Checklist\n"
            "\n- [X] I understand that this PR may be closed in case there was"
            " no previous discussion. (This doesn't apply to typos!)..."
        ),
        ground_truth_files=("httpx/_client.py",),
        ground_truth_symbols=(
            "httpx._client.Client",
            "httpx._client.Client.send",
            "httpx._client.AsyncClient",
            "httpx._client.AsyncClient.send",
        ),
        test_files=("tests/test_timeouts.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-13",
        repo_slug="encode/httpx",
        pr_number=3045,
        pr_url="https://github.com/encode/httpx/pull/3045",
        merge_commit_sha="99cba6ac642b12f0331d069ae324402c54c4c600",
        parent_commit_sha="ca51b4532a4495c38c8abf1a393dbd80099deae3",
        title=("Fix RFC 2069 mode digest authentication"),
        problem_statement=(
            "Fix RFC 2069 mode digest authentication\n\n# Summary\n\nFixes a typo"
            " that breaks old RFC 2069 mode digest auth.\n\nThe issue is that H"
            "A1 is added both on line 283, and again on line 293. ([According"
            " to RFC, it should be there only once.](https://datatracker.ietf"
            ".org/doc/html/rfc2069#section-2.1.2))\n\nAlso mentioned in https:/"
            "/github.com/encode/httpx/issues/2788#issuecomment-1677260605\n\n# "
            "Checklist\n\n- [x] I understand that this PR may be closed in case"
            " there was no previous discussion. (This doesn't apply to typos!"
            ")\n- [x] I've added a test for each change that was introduced, a"
            "nd I tried as much as possible to make a single atomic change.\n-"
            " [ ] I've updated the documentation accordingly."
        ),
        ground_truth_files=("httpx/_auth.py",),
        ground_truth_symbols=(
            "httpx._auth.DigestAuth",
            "httpx._auth.DigestAuth._build_auth_header",
        ),
        test_files=("tests/test_auth.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-14",
        repo_slug="encode/httpx",
        pr_number=2999,
        pr_url="https://github.com/encode/httpx/pull/2999",
        merge_commit_sha="90538a3b4610ba49ce16c997bcfdb9101f9503be",
        parent_commit_sha="f8981f3d124f9b8db9073fd5c8afa11acb55a738",
        title=("Ensure that ASGI 'raw_path' does not include query component of URL."),
        problem_statement=(
            "Ensure that ASGI 'raw_path' does not include query component of "
            "URL.\n\nCloses #2810.\n\nUpdate the ASGI transport to ensure that `r"
            "aw_path` does not include the query component.\n\nThe spec was ori"
            "ginally a little ambiguous here, and it might(?) have made sense"
            " for it to include the query portion, since that way around it'd"
            " be a direct representation of the HTTP/1.1 target byte string t"
            "hat was received on the wire. But the spec's been clarified sinc"
            "e, and we're not matching that correctly at the moment. Let's fi"
            "x that! ☺️"
        ),
        ground_truth_files=("httpx/_transports/asgi.py",),
        ground_truth_symbols=(
            "httpx._transports.asgi.ASGITransport",
            "httpx._transports.asgi.ASGITransport.handle_async_request",
        ),
        test_files=("tests/test_asgi.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="HTTPX-15",
        repo_slug="encode/httpx",
        pr_number=2998,
        pr_url="https://github.com/encode/httpx/pull/2998",
        merge_commit_sha="1e110964736a95d822144390b99d7ecf17fca527",
        parent_commit_sha="90538a3b4610ba49ce16c997bcfdb9101f9503be",
        title=("Fixed `iter_text` adding an empty string"),
        problem_statement=(
            "Fixed `iter_text` adding an empty string\n\n# Summary\n\nFixes `Text"
            'Chunker().decode("")` not handling empty string, as exposed in h'
            "ttps://github.com/encode/httpx/discussions/2995\n\n# Checklist\n\n- "
            "[x] I understand that this PR may be closed in case there was no"
            " previous discussion. (This doesn't apply to typos!)\n- [x] I've "
            "added a test for each change that was introduced, and I tried as"
            " much as possible to make a single atomic change.\n- [ ] I've upd"
            "ated the documentation accordingly."
        ),
        ground_truth_files=(
            "httpx/_decoders.py",
            "httpx/_models.py",
        ),
        ground_truth_symbols=(
            "httpx._decoders.TextChunker",
            "httpx._decoders.TextChunker.decode",
            "httpx._decoders.LineDecoder",
            "httpx._decoders.LineDecoder.decode",
            "httpx._models.Response",
            "httpx._models.Response.iter_text",
        ),
        test_files=("tests/test_decoders.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-01",
        repo_slug="pallets/click",
        pr_number=3901,
        pr_url="https://github.com/pallets/click/pull/3901",
        merge_commit_sha="5125851aaa19840c82b9811193b2c2d6cf245326",
        parent_commit_sha="6d34602f4b0668a61873af1c547bfd40cb7deffc",
        title=("Stop passing a deprecated command's help text through `gettext`"),
        problem_statement=(
            "Stop passing a deprecated command's help text through `gettext`\n"
            "\nThis PR is a small consistency fix, that is aligning the way th"
            "e `(DEPRECATED)` label is translated in options, arguments and c"
            "ommands.\n\nIn the latter, the deprecation text was translated in "
            "addition to the label, which is not the behavior of options and "
            "arguments.\n\nRelated to: https://github.com/janluke/cloup/pull/24"
            "4"
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core.Command",
            "click.core.Command.get_short_help_str",
            "click.core.Command.format_help_text",
        ),
        test_files=("tests/test_commands.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-02",
        repo_slug="pallets/click",
        pr_number=3884,
        pr_url="https://github.com/pallets/click/pull/3884",
        merge_commit_sha="8c73ff134480f436faa0d041bd19aaf949aac22c",
        parent_commit_sha="bb695aaecdef95832d167fc1d508921745aa824e",
        title=("Fix `show_envvar` output for a sequence or an empty `envvar`"),
        problem_statement=(
            "Fix `show_envvar` output for a sequence or an empty `envvar`\n\nTh"
            "is PR fixes some awkward rendering in the `Error: Invalid value "
            "for '--units' (env var: xxxxx)` error message's hint for special"
            " cases of environment variables:\n\n| Parameter value | Hint befor"
            "e | Hint after |\n| --- | --- | --- |\n| `envvar=None` | Nothing |"
            " Nothing |\n| `envvar=\"\"` | `(env var: '')` | Nothing |\n| `envvar"
            "=[]` | `(env var: '[]')` | Nothing |\n| `envvar=[\"WEATHER_UNITS\","
            " \"UNITS\"]` | `(env var: '['WEATHER_UNITS', 'UNITS']')` | `(env v"
            "ar: 'WEATHER_UNITS', 'UNITS')` |\n\nAlso fixes the rendering of he"
            "lp screen:\n| Parameter value | Help screen before | Help screen "
            "after |\n| --- | --- | --- |\n| `envvar=None` | Nothing | Nothing "
            '|\n| `envvar=""` | `[env var: ]` | Nothing |...'
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core.Option",
            "click.core.Option.get_error_hint",
            "click.core.Option._parse_decls",
            "click.core.Option.get_help_extra",
        ),
        test_files=("tests/test_options.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-03",
        repo_slug="pallets/click",
        pr_number=3866,
        pr_url="https://github.com/pallets/click/pull/3866",
        merge_commit_sha="06b2a678741131fd577ce170e23e5ca0aeba0309",
        parent_commit_sha="3cbaa76b6014be7427d1ab06b4af40f01e7c278e",
        title=(
            "Deprecate parameter names that are: not a Python identifier, or "
            "a Python Keyword, or not lower-cased"
        ),
        problem_statement=(
            "Deprecate parameter names that are: not a Python identifier, or "
            "a Python Keyword, or not lower-cased\n\nThis PR prepares the groun"
            "d for #3827, which is going to refuse parameter names that are n"
            "ot valid Python identifiers.\n\nThis add `DeprecationWarning` that"
            " detect when `Parameter`, `Argument` or `Option` gets a `name` t"
            "hat will be not allowed in the future 9.0.0 release:\n- a paramet"
            "er that is not a Python identifier (check against `str.identifie"
            "r`)\n- a parameter that is a Python keyword (check against `keywo"
            "rd.iskeyword`)\n- an option name that is nor lower-cased\n\nThis PR"
            " also copy the majority of tests that were originally proposed i"
            "n #3827, so the behavior of the `name` is locked down and frozen"
            ". This will prevent tweaking or introducing regressions during t"
            "he full 8.x cycle...."
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core._format_deprecated_label",
            "click.core._format_deprecated_suffix",
            "click.core.batch",
            "click.core.Parameter",
            "click.core.Parameter.get_default",
            "click.core.Option",
        ),
        test_files=(
            "tests/test_arguments.py",
            "tests/test_deprecations.py",
            "tests/test_options.py",
        ),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-04",
        repo_slug="pallets/click",
        pr_number=3865,
        pr_url="https://github.com/pallets/click/pull/3865",
        merge_commit_sha="70689853e39c30e36eb0d83d586cc914a74db9d3",
        parent_commit_sha="d036881798e34289a49b18a5c550c7cf687e5a7a",
        title=("Fix short help abbreviation"),
        problem_statement=(
            "Fix short help abbreviation\n\nJust a small bug I encountered in m"
            "y CLIs, where as `vs.` or `e.g.` was prematurely ending the shor"
            "ten version of a help message produced by `_make_default_short_h"
            "elp`.\n\nAlso use the opportunity to simplify the calculations and"
            " depends on Python's `textwrap.shorten`."
        ),
        ground_truth_files=("src/click/utils.py",),
        ground_truth_symbols=("click.utils._make_default_short_help",),
        test_files=("tests/test_utils/test_make_default_short_help.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-05",
        repo_slug="pallets/click",
        pr_number=3861,
        pr_url="https://github.com/pallets/click/pull/3861",
        merge_commit_sha="6d5bac5c712195cbfa9666ba0ecbf0573cdea356",
        parent_commit_sha="86949d19b930ef68a45566544cb4b9508d895d0d",
        title=("Move common `help` argument from `Option` and `Argument` to `Parameter`"),
        problem_statement=(
            "Move common `help` argument from `Option` and `Argument` to `Par"
            "ameter`\n\nThis is a small refactor which deduplicate the manageme"
            "nt of `help` argument from `Option` and `Argument` and move it t"
            "o the common `Parameter` parent class.\n\nIts only side effect is "
            "that `Parameter.to_info_dict` now reports it.\n\nThis is a follow "
            "up on #3860."
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core.Parameter",
            "click.core.Parameter.__init__",
            "click.core.Parameter._hide_unset",
            "click.core.Parameter.to_info_dict",
            "click.core.Parameter.make_metavar",
            "click.core.Option",
        ),
        test_files=("tests/test_info_dict.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-06",
        repo_slug="pallets/click",
        pr_number=3860,
        pr_url="https://github.com/pallets/click/pull/3860",
        merge_commit_sha="bb695aaecdef95832d167fc1d508921745aa824e",
        parent_commit_sha="70689853e39c30e36eb0d83d586cc914a74db9d3",
        title=("Always list all positional arguments"),
        problem_statement=(
            "Always list all positional arguments\n\nThis is inconsistent with "
            "the way options are displayed, as every options are displayed in"
            "cluding those without help.\n\nThe reason to display all arguments"
            " in the `Positional arguments` section is to not introduce subtl"
            "e confusion, as an argument without help would only be noticeabl"
            "e in the usage line.\n\nCloup already renders the section this way"
            " (see https://github.com/janluke/cloup/issues/210) and is the so"
            "urce discussion with @janluke for that bug fix...."
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core.Command",
            "click.core.Command.format_arguments",
            "click.core.Argument",
            "click.core.Argument.get_help_record",
            "click.core.Argument.get_error_hint",
            "click.core.Argument.add_to_parser",
        ),
        test_files=("tests/test_arguments.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-07",
        repo_slug="pallets/click",
        pr_number=3821,
        pr_url="https://github.com/pallets/click/pull/3821",
        merge_commit_sha="00aca23ee9135eedc651c6929c7bc1d39e522431",
        parent_commit_sha="be13b152cf5949a362926322c835fa76019cc6fe",
        title=("Add `Option.get_help_spec` to extract the help's left part of an hidden option"),
        problem_statement=(
            "Add `Option.get_help_spec` to extract the help's left part of an"
            " hidden option\n\nThis PR splits the production of an option's lef"
            "t help column into its own dedicated column. This always returns"
            " the invocation specification of an option, even if that option "
            "is hidden.\n\n`get_help_record()` on the other hand is unmodified "
            "and still keep returning the whole option's help that is to be d"
            "isplayed.\n\nThat distinction help separate the rendering layer:\n-"
            " `get_help_record()` is returning what should be displayed to th"
            "e user \n- `get_help_spec()` is returning what the option specifi"
            "cation is, regardless of the decision to display it to the user\n"
            "\nThis allows finer introspection of the Click machinery and its "
            "rendering pipeline...."
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core.Option",
            "click.core.Option.get_help_record",
            "click.core.Option.get_help_extra",
        ),
        test_files=("tests/test_options.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-08",
        repo_slug="pallets/click",
        pr_number=3818,
        pr_url="https://github.com/pallets/click/pull/3818",
        merge_commit_sha="00f257bb8e714064bea16425007543bbc0800445",
        parent_commit_sha="b47c0529e99129d4785fecc2cb39a8ec8b4aa5f8",
        title=("Fix race condition on `KeyboardInterrupt` in `Command.main()`"),
        problem_statement=(
            "Fix race condition on `KeyboardInterrupt` in `Command.main()`\n\nT"
            "his PR address https://github.com/pallets/click/issues/3802 by u"
            "nifying the policy that was already implemented elsewhere in the"
            " code.\n\nThis applies the policy the pager has had since #351, wh"
            "ere a late interrupt must not replace the intended outcome:\nhttp"
            "s://github.com/pallets/click/blob/8b19813f2bfca99f1018a587a8cf54"
            "fc959f2e5d/src/click/_termui_impl.py#L618-L632\n\nSo now in `main("
            ")`, every handler takes the same two steps:\n- propagate when sta"
            "ndalone mode is off, or\n- collect the message and the exit code."
            "\n\nA final step then prints the message and exits, so a late inte"
            "rrupt can only lose the message, not the code.\n\nThis save one ne"
            "sting level in the `try...except` cascade of `main()`...."
        ),
        ground_truth_files=("src/click/core.py",),
        ground_truth_symbols=(
            "click.core._format_deprecated_label",
            "click.core.Command",
            "click.core.Command.main",
            "click.core.Command._main_shell_completion",
        ),
        test_files=("tests/test_abort_interrupt.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-09",
        repo_slug="pallets/click",
        pr_number=3805,
        pr_url="https://github.com/pallets/click/pull/3805",
        merge_commit_sha="4295457ddc7b5ca9a733d59ca40d9651b02b9b1d",
        parent_commit_sha="3cbcf9b11546f4cf10b36d3e2e531733ba6fe001",
        title=("Fix `copy`, `deepcopy` and `pickle` of `Sentinel` members"),
        problem_statement=(
            "Fix `copy`, `deepcopy` and `pickle` of `Sentinel` members\n\nJust "
            "a small issue I encountered on Python 3.10, where `Enum` has no "
            "`__deepcopy__`. Now that every parameter have a sentinel, `copy."
            "deepcopy()` of any `Option`, `Argument` or `Command` fails on Py"
            "thon 3.10. And `pickle` fails on every Python version. This PR f"
            "ix this issue."
        ),
        ground_truth_files=("src/click/_utils.py",),
        ground_truth_symbols=("click._utils.Sentinel",),
        test_files=("tests/test_utils/test_sentinel.py",),
        verify_with_pytest=True,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-10",
        repo_slug="pallets/click",
        pr_number=3781,
        pr_url="https://github.com/pallets/click/pull/3781",
        merge_commit_sha="e1fd5946ab26aaf372009eaff1acf947140b40fb",
        parent_commit_sha="2103e157683c5e4cadc8ee1838df526a54bde9a4",
        title=("Add support of `pathlib.Path` to `edit`"),
        problem_statement=("Add support of `pathlib.Path` to `edit`\n\nCloses #2869."),
        ground_truth_files=(
            "src/click/_termui_impl.py",
            "src/click/termui.py",
        ),
        ground_truth_symbols=(
            "click._termui_impl.Editor",
            "click._termui_impl.Editor.edit_files",
            "click.termui.edit",
            "click.termui.launch",
        ),
        test_files=(
            "tests/test_termui.py",
            "tests/typing/typing_edit.py",
        ),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-11",
        repo_slug="pallets/click",
        pr_number=3777,
        pr_url="https://github.com/pallets/click/pull/3777",
        merge_commit_sha="2103e157683c5e4cadc8ee1838df526a54bde9a4",
        parent_commit_sha="61b69e967e525bd502f5bf42def4d551e761fe0e",
        title=("Forward all user's parameters set in `PAGER` and improve flag detection"),
        problem_statement=(
            "Forward all user's parameters set in `PAGER` and improve flag de"
            "tection\n\nAnother PR following up on pager refactors that:\n- Forw"
            "ards all user's parameters set in `PAGER`\n- Improve detection of"
            " raw mode by extending flag and option parsing\n- Fix detection o"
            "f `less.exe` on Windows (even if not currently exercised, this m"
            "ake the method future-proof)\n\nBut more importantly: adds dozen o"
            "f testing with weird edge-cases around pager's internal methods."
            "\n\nFollows up on:\n- https://github.com/pallets/click/pull/3767\n- "
            "https://github.com/pallets/click/pull/3776"
        ),
        ground_truth_files=("src/click/_termui_impl.py",),
        ground_truth_symbols=(
            "click._termui_impl._pager_contextmanager",
            "click._termui_impl._pipepager",
            "click._termui_impl._tempfilepager",
            "click._termui_impl._nullpager",
            "click._termui_impl.Editor",
            "click._termui_impl.Editor.get_editor",
        ),
        test_files=("tests/test_termui.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-12",
        repo_slug="pallets/click",
        pr_number=3769,
        pr_url="https://github.com/pallets/click/pull/3769",
        merge_commit_sha="150d1071d69c5cdad7de78590013ffe56cf9e3bb",
        parent_commit_sha="8b44edfff7d9a6c895fa804148c16b3a0bc9efb5",
        title=("Settles progress bar on its final position"),
        problem_statement=(
            "Settles progress bar on its final position\n\nThis fix has been sh"
            "ipped in Click Extra for a couple of weeks and has been tested o"
            "n a couple of production ready CLIs.\n\nCloses #3571.\n\nThis PR unf"
            "ortunately close the honeypot of lazy LLMs contributions. 😁"
        ),
        ground_truth_files=("src/click/_termui_impl.py",),
        ground_truth_symbols=(
            "click._termui_impl.ProgressBar",
            "click._termui_impl.ProgressBar.render_finish",
            "click._termui_impl.ProgressBar.pct",
        ),
        test_files=("tests/test_termui.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-13",
        repo_slug="pallets/click",
        pr_number=3767,
        pr_url="https://github.com/pallets/click/pull/3767",
        merge_commit_sha="f36d58bbd7f188178de2d4fe1d0292c510375ca7",
        parent_commit_sha="cbd7a4109da16ce58f54c2a618b4c986e3041fcf",
        title=("Refactor pager stream handling"),
        problem_statement=(
            "Refactor pager stream handling\n\nThis streamline the pager handli"
            "ng and removes the distinction between text and binary streams.\n"
            "\nIt is a follow up on:\n- #1572 \n- #3405"
        ),
        ground_truth_files=(
            "src/click/_compat.py",
            "src/click/_termui_impl.py",
        ),
        ground_truth_symbols=(
            "click._compat.should_strip_ansi",
            "click._termui_impl._BufferedTextPagerStream",
            "click._termui_impl._has_binary_buffer",
            "click._termui_impl.ProgressBar",
            "click._termui_impl.ProgressBar.generator",
            "click._termui_impl.MaybeStripAnsi",
        ),
        test_files=(
            "tests/test_termui.py",
            "tests/test_utils/test_echo_via_pager.py",
        ),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-14",
        repo_slug="pallets/click",
        pr_number=3764,
        pr_url="https://github.com/pallets/click/pull/3764",
        merge_commit_sha="fc5c7f45da0a4443f60ba2d322fe4bf829977739",
        parent_commit_sha="150d1071d69c5cdad7de78590013ffe56cf9e3bb",
        title=("Fix error reporting on Windows when an error is raised in the pager "),
        problem_statement=(
            "Fix error reporting on Windows when an error is raised in the pa"
            "ger\n\nThis is a follow up on #3739, which fix an issue on Windows"
            " where an exception gets replaced by a `PermissionError: [WinErr"
            "or 32]` error instead of being raised within a pager.\n\nAlso of n"
            "ote, this restore a fix that was pushed in 8.0.2 by #1905 / #195"
            "6 / #2036 which was not reintegrated by the #1572 new feature fo"
            "r 8.4.0."
        ),
        ground_truth_files=("src/click/_termui_impl.py",),
        ground_truth_symbols=(
            "click._termui_impl._tempfilepager",
            "click._termui_impl._nullpager",
        ),
        test_files=("tests/test_termui.py",),
        verify_with_pytest=False,
    ),
    UnseenRepoBugTask(
        task_id="CLICK-15",
        repo_slug="pallets/click",
        pr_number=3681,
        pr_url="https://github.com/pallets/click/pull/3681",
        merge_commit_sha="8c1a0a7abbc1c36f70d1f65f3604acc46c5ce6ab",
        parent_commit_sha="8a4ce842564ae94ab050062db8525196ad476c19",
        title=("Strip all ANSI sequences"),
        problem_statement=(
            "Strip all ANSI sequences\n\nThis PR augment the ANSI regexp used i"
            "n Click so it can recognize and strip all ANSI sequences, includ"
            "ing those not produced by Click itself.\n\nI made the tests a bit "
            "exhaustive to demonstrate the lack of regression between the old"
            "er naive regexp and my new proposal. But if you are convinced I "
            "can just get rid of them.\n\nThe code is based on work I produced "
            "6 years ago and contributed to `boltons`: https://github.com/mah"
            "moud/boltons/pull/258\n\nIt is in line with what is done upstream "
            "in the cpython project:\n- https://github.com/python/cpython/blob"
            "/cd9994e2831ea743e5423c1db56ca2ca42154e0a/Lib/_pyrepl/utils.py#L"
            "21..."
        ),
        ground_truth_files=("src/click/_compat.py",),
        ground_truth_symbols=(
            "click._compat._make_text_stream",
            "click._compat.strip_ansi",
        ),
        test_files=(
            "tests/test_compat.py",
            "tests/test_formatting.py",
            "tests/test_utils/test_style.py",
        ),
        verify_with_pytest=False,
    ),
)


def build_unseen_repo_suite() -> UnseenRepoBenchmarkSuite:
    """Return the validated 30-task UnseenRepoBenchmarkSuite."""
    return UnseenRepoBenchmarkSuite(tasks=UNSEEN_REPO_TASKS)
