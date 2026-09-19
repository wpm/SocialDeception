//! A directory of one test's own under the temp dir.
//!
//! One file, included by `#[path]` into the crate's unit tests, the
//! `werewolf` binary's, and the integration tests, since a test helper
//! cannot be a library export and the three would otherwise each carry a
//! copy.

use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A directory of one test's own under the temp dir, unique across tests and
/// processes, removed with everything in it when this is dropped, so that a
/// failing test leaves nothing behind. Derefs to its [`Path`], so a file in
/// it is `dir.join("name")`.
pub struct TempDir(PathBuf);

impl TempDir {
    /// Creates the directory.
    ///
    /// # Panics
    ///
    /// If the directory cannot be created.
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "social-deception-{}-{}",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("the temp dir is writable");
        Self(path)
    }
}

impl Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
