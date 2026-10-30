//! AMD GPU stats from Linux amdgpu sysfs (`/sys/class/drm/cardN/device`).
//! Other platforms report no cards.

use crate::protocol::DeviceStat;

/// One AMD card's sysfs device directory.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
pub(crate) struct AmdCard {
    dir: std::path::PathBuf,
}

#[cfg_attr(not(feature = "hip"), allow(dead_code))]
impl AmdCard {
    fn read(&self, rel: &str) -> Option<String> {
        std::fs::read_to_string(self.dir.join(rel))
            .ok()
            .map(|s| s.trim().to_string())
    }

    fn num(&self, rel: &str) -> Option<u64> {
        self.read(rel)?.parse().ok()
    }

    /// First `hwmon/hwmon*/<file>` that parses.
    fn hwmon(&self, file: &str) -> Option<u64> {
        let rd = std::fs::read_dir(self.dir.join("hwmon")).ok()?;
        rd.flatten().find_map(|e| {
            std::fs::read_to_string(e.path().join(file))
                .ok()?
                .trim()
                .parse()
                .ok()
        })
    }

    /// Overwrite the fields sysfs reports; leave the rest.
    pub(crate) fn fill(&self, stat: &mut DeviceStat) {
        if let Some(name) = self.read("product_name").filter(|n| !n.is_empty()) {
            stat.name = name;
        }
        if let (Some(used), Some(total)) = (
            self.num("mem_info_vram_used"),
            self.num("mem_info_vram_total"),
        ) {
            stat.mem_used = used;
            stat.mem_total = total;
        }
        if let Some(busy) = self.num("gpu_busy_percent") {
            stat.util_pct = busy.min(100) as i32;
        }
        // millidegrees C; microwatts.
        if let Some(t) = self.hwmon("temp1_input") {
            stat.temp_c = (t / 1000) as i32;
        }
        if let Some(p) = self
            .hwmon("power1_average")
            .or_else(|| self.hwmon("power1_input"))
        {
            stat.power_w = (p / 1_000_000) as i32;
        }
    }
}

/// AMD cards under `root` (normally `/sys/class/drm`), in PCI address order.
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
pub(crate) fn amd_cards_in(root: &std::path::Path) -> Vec<AmdCard> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut cards: Vec<(String, AmdCard)> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            // `card0`, not connectors like `card0-DP-1`.
            let idx = name.strip_prefix("card")?;
            if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let dir = e.path().join("device");
            let vendor = std::fs::read_to_string(dir.join("vendor")).ok()?;
            if vendor.trim() != "0x1002" {
                return None;
            }
            // The device link points at its PCI address (0000:04:00.0). A
            // plain directory (no link) sorts by card name instead of every
            // card resolving to the same `device` and keeping readdir order.
            let pci = std::fs::read_link(&dir)
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or(name);
            Some((pci, AmdCard { dir }))
        })
        .collect();
    cards.sort_by(|a, b| a.0.cmp(&b.0));
    cards.into_iter().map(|(_, c)| c).collect()
}

#[cfg_attr(not(feature = "hip"), allow(dead_code))]
pub(crate) fn amd_cards() -> Vec<AmdCard> {
    if cfg!(target_os = "linux") {
        amd_cards_in(std::path::Path::new("/sys/class/drm"))
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(root: &std::path::Path, name: &str, files: &[(&str, &str)]) {
        let dir = root.join(name).join("device");
        std::fs::create_dir_all(dir.join("hwmon/hwmon3")).unwrap();
        for (f, v) in files {
            std::fs::write(dir.join(f), v).unwrap();
        }
    }

    #[test]
    fn reads_amd_cards_and_skips_others() {
        let root = std::env::temp_dir().join(format!("allpaca-sysfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        card(
            &root,
            "card1",
            &[
                ("vendor", "0x1002\n"),
                ("product_name", "Radeon Instinct MI50 16GB\n"),
                ("gpu_busy_percent", "96\n"),
                ("mem_info_vram_used", "2704924672\n"),
                ("mem_info_vram_total", "34342961152\n"),
                ("hwmon/hwmon3/temp1_input", "49000\n"),
                ("hwmon/hwmon3/power1_average", "153000000\n"),
            ],
        );
        card(&root, "card0", &[("vendor", "0x10de\n")]);
        std::fs::create_dir_all(root.join("card1-DP-1")).unwrap();
        let cards = amd_cards_in(&root);
        assert_eq!(cards.len(), 1);
        let mut stat = DeviceStat {
            name: "hip:0".into(),
            mem_used: 1,
            mem_total: 2,
            util_pct: -1,
            temp_c: -1,
            power_w: -1,
        };
        cards[0].fill(&mut stat);
        assert_eq!(
            stat,
            DeviceStat {
                name: "Radeon Instinct MI50 16GB".into(),
                mem_used: 2_704_924_672,
                mem_total: 34_342_961_152,
                util_pct: 96,
                temp_c: 49,
                power_w: 153,
            }
        );
        // A card with no stat files keeps what the runtime reported.
        card(&root, "card2", &[("vendor", "0x1002\n")]);
        let cards = amd_cards_in(&root);
        assert_eq!(cards.len(), 2);
        let mut bare = DeviceStat {
            name: "hip:1".into(),
            mem_used: 5,
            mem_total: 9,
            util_pct: -1,
            temp_c: -1,
            power_w: -1,
        };
        cards[1].fill(&mut bare);
        assert_eq!(
            (bare.name.as_str(), bare.mem_used, bare.util_pct),
            ("hip:1", 5, -1)
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
