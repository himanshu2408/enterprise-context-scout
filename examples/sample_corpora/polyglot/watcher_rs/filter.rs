//! File watcher debounce filter for clean-clone CI polyglot seeding.

pub struct DebounceFilter {
    pub debounce_ms: u64,
}

impl DebounceFilter {
    pub fn should_emit(&self, elapsed_ms: u64) -> bool {
        elapsed_ms >= self.debounce_ms
    }
}
