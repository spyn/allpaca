//! How the server builds its [`Config`] before CLI overrides.

use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, Config};

/// Config plus the file it came from, when one was read.
pub(crate) struct LoadedConfig {
    pub config: Config,
    /// Explicit `--config`, or `allpaca.toml` in `cwd` when that flag is omitted.
    pub path: Option<PathBuf>,
}

/// Compiled defaults, then an explicit `--config`, or `allpaca.toml` in `cwd`
/// when that flag is omitted and the file is there. A present file that does
/// not parse is an error. CLI overrides are applied by the caller.
pub(crate) fn load_startup_config(explicit: Option<&Path>, cwd: &Path) -> Result<LoadedConfig, AllpacaError> {
    if let Some(path) = explicit {
        return Ok(LoadedConfig {
            config: Config::from_toml_file(path)?,
            path: Some(path.to_path_buf()),
        });
    }
    let auto = cwd.join("allpaca.toml");
    if auto.is_file() {
        tracing::info!(path = %auto.display(), "loading allpaca.toml from the working directory");
        return Ok(LoadedConfig {
            config: Config::from_toml_file(&auto)?,
            path: Some(auto),
        });
    }
    Ok(LoadedConfig {
        config: Config::compiled_defaults(),
        path: None,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use allpaca_types::CliOverrides;

    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "allpaca-q15-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cwd_allpaca_toml_sets_the_http_password_when_config_is_omitted() {
        let dir = TempDir::new();
        fs::write(dir.0.join("allpaca.toml"), "[http]\npassword = \"from-file\"\n").unwrap();
        let loaded = load_startup_config(None, &dir.0).unwrap();
        assert_eq!(loaded.config.http.password.as_deref(), Some("from-file"));
        assert_eq!(loaded.path.as_deref(), Some(dir.0.join("allpaca.toml").as_path()));
    }

    #[test]
    fn explicit_config_is_used_instead_of_the_working_directory_file() {
        let dir = TempDir::new();
        fs::write(dir.0.join("allpaca.toml"), "[http]\npassword = \"cwd\"\n").unwrap();
        let explicit = dir.0.join("other.toml");
        fs::write(&explicit, "[http]\npassword = \"explicit\"\n").unwrap();
        let loaded = load_startup_config(Some(&explicit), &dir.0).unwrap();
        assert_eq!(loaded.config.http.password.as_deref(), Some("explicit"));
        assert_eq!(loaded.path.as_deref(), Some(explicit.as_path()));
    }

    #[test]
    fn missing_cwd_file_keeps_compiled_defaults() {
        let dir = TempDir::new();
        let loaded = load_startup_config(None, &dir.0).unwrap();
        assert_eq!(loaded.config.http.password, None);
        assert!(loaded.path.is_none());
    }

    #[test]
    fn a_broken_cwd_file_is_an_error() {
        let dir = TempDir::new();
        fs::write(dir.0.join("allpaca.toml"), "not toml [[[").unwrap();
        assert!(load_startup_config(None, &dir.0).is_err());
    }

    #[test]
    fn cli_password_overrides_the_file() {
        let dir = TempDir::new();
        fs::write(dir.0.join("allpaca.toml"), "[http]\npassword = \"from-file\"\n").unwrap();
        let mut loaded = load_startup_config(None, &dir.0).unwrap();
        loaded.config.merge_cli(&CliOverrides {
            ui_password: Some("from-cli".into()),
            ..CliOverrides::default()
        });
        assert_eq!(loaded.config.http.password.as_deref(), Some("from-cli"));
    }
}
