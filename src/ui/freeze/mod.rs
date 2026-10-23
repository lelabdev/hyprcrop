//! # ui::freeze
//!
//! Entry point for freeze mode.
//! Captures all monitors, displays the result as a full-screen overlay,
//! waits for the user to select a capture region, then crops the frozen
//! image and saves the final output.
//!
//! The `app` submodule implements the iced state machine (`AppState`).
//! This module orchestrates initialization, execution, and post-processing.

mod app;

pub use app::CaptureMode;

use app::{
    AppState, AppStateConfig, FreezeSelection, Message, app_subscription, app_update, app_view,
};
use iced::Task;
use iced::widget::image as iced_image;
use iced_layershell::{
    reexport::{Anchor, IcedId, KeyboardInteractivity, Layer, NewLayerShellSettings, OutputOption},
    settings::LayerShellSettings,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
};

use crate::domain::config::Config;
use crate::domain::constants::FREEZE_LAYER_NAMESPACE;
use crate::domain::error::{AppError, Result};
use crate::domain::types::{BorderStyle, ScreenRect};
use crate::platform::capture::screencopy;
use crate::platform::capture::toplevel_export;
use crate::platform::system::compositor::{self, CompositorBackend, CompositorKind};

pub fn run_freeze(cfg: &Config) -> Result<PathBuf> {
    let backend = compositor::detect()?;

    // iced builds its system font database lazily on the first text layout, which
    // scans every installed font. Do it while the screen is being captured instead
    // of on the overlay's critical path; views omit text until this flag flips
    // (see `AppState::text_ready`).
    let fonts = FontWarmup::spawn();

    // Fetch compositor data in parallel threads for speed.  At most 3 concurrent
    // IPC requests are made; border style is fetched after joining.
    let monitors_backend = backend;
    let monitors_t = std::thread::spawn(move || monitors_backend.monitors());
    let clients_backend = backend;
    let clients_t = std::thread::spawn(move || clients_backend.windows());
    let layers_backend = backend;
    let layers_t = std::thread::spawn(move || layers_backend.overlay_layers());

    let monitors = monitors_t
        .join()
        .map_err(|_| AppError::Other("monitors thread panicked".into()))??;
    let windows = clients_t
        .join()
        .map_err(|_| AppError::Other("clients thread panicked".into()))??;
    let layers = layers_t
        .join()
        .unwrap_or_else(|_| {
            eprintln!("[hyprcrop] warning: overlay layers thread panicked");
            Ok(Vec::new())
        })
        .unwrap_or_else(|e| {
            eprintln!("[hyprcrop] warning: failed to fetch overlay layers: {e}");
            Vec::new()
        });

    // Border expansion is irrelevant when toplevel-export is used (the protocol
    // captures the raw window surface, no compositor decorations). Suppress it here
    // rather than mutating the global config flag, so non-freeze commands are unaffected.
    let border_style = if cfg.capture_window_border {
        backend.border_style()
    } else {
        BorderStyle::default()
    };
    let initial_mode =
        resolve_initial_mode(&cfg.freeze_buttons, crate::domain::state::load_last_mode());

    // Compute origin before monitors are moved into Arc.
    // capture_all_monitors places (min_x, min_y) at image pixel (0,0); we need this
    // to translate the UI's global logical coordinates into image coordinates later.
    let (min_x, min_y) = crate::domain::geometry::monitor_origin(&monitors);

    // Capture all monitors in a single Wayland session.
    // Using one session guarantees the overlay images and the final-crop source are
    // from the same frame — two separate captures would differ in time, breaking
    // the "freeze" guarantee (user selects based on a different frame than what gets saved).
    let (physical_per_monitor, full_rgba) =
        screencopy::capture_all_monitors_with_physical(&monitors)?;

    let monitor_images: Vec<iced_image::Handle> = physical_per_monitor
        .into_iter()
        .map(|img| iced_image::Handle::from_rgba(img.width(), img.height(), img.into_raw()))
        .collect();

    let focused_monitor_idx = monitors.iter().position(|m| m.focused).unwrap_or(0);

    let (fw, fh) = {
        let m = &monitors[focused_monitor_idx];
        (m.rect.w as u32, m.rect.h as u32)
    };

    let result: Arc<Mutex<Option<Option<FreezeSelection>>>> = Arc::new(Mutex::new(None));

    {
        let result_clone = result.clone();
        let mut window_to_monitor: HashMap<IcedId, usize> = HashMap::new();

        let extra_specs: Vec<(IcedId, NewLayerShellSettings)> = monitors
            .iter()
            .enumerate()
            .filter(|(_, m)| !m.focused)
            .map(|(idx, m)| {
                let id = IcedId::unique();
                window_to_monitor.insert(id, idx);
                let settings = NewLayerShellSettings {
                    size: Some((m.rect.w as u32, m.rect.h as u32)),
                    layer: Layer::Overlay,
                    anchor: Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right,
                    exclusive_zone: Some(-1),
                    keyboard_interactivity: KeyboardInteractivity::Exclusive,
                    output_option: OutputOption::OutputName(m.name.clone()),
                    namespace: Some(FREEZE_LAYER_NAMESPACE.to_owned()),
                    ..Default::default()
                };
                (id, settings)
            })
            .collect();

        let extra_specs = std::sync::Arc::new(extra_specs);

        let wins = Arc::new(windows);
        let mons = Arc::new(monitors);
        let lyrs = Arc::new(layers);
        let glyphs = cfg.freeze_glyphs.clone();
        let toolbar_position = cfg.toolbar_position;
        let colors = cfg.freeze_colors;
        let freeze_buttons = cfg.freeze_buttons;

        iced_layershell::daemon(
            move || {
                let spawn_tasks: Vec<Task<Message>> = extra_specs
                    .iter()
                    .map(|(id, settings)| {
                        Task::done(Message::NewLayerShell {
                            settings: settings.clone(),
                            id: *id,
                        })
                    })
                    .collect();

                let state = AppState::new(AppStateConfig {
                    monitor_images: monitor_images.clone(),
                    focused_monitor_idx,
                    window_to_monitor: window_to_monitor.clone(),
                    windows: Arc::clone(&wins),
                    monitors: Arc::clone(&mons),
                    layers: Arc::clone(&lyrs),
                    result: result_clone.clone(),
                    glyphs: glyphs.clone(),
                    toolbar_position,
                    border_style,
                    initial_mode,
                    colors,
                    freeze_buttons,
                    fonts: fonts.clone(),
                });
                (state, Task::batch(spawn_tasks))
            },
            FREEZE_LAYER_NAMESPACE,
            app_update,
            app_view,
        )
        .subscription(app_subscription)
        .layer_settings(LayerShellSettings {
            layer: Layer::Overlay,
            anchor: Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right,
            exclusive_zone: -1,
            keyboard_interactivity: KeyboardInteractivity::Exclusive,
            size: Some((fw, fh)),
            ..Default::default()
        })
        .run()
        .map_err(|e| AppError::LayerShell(e.to_string()))?;
    }

    let selected = result
        .lock()
        .expect("UI thread panicked and poisoned result mutex")
        .take();

    let out_path = cfg.output_path();

    match selected {
        None => Err(AppError::UserCancelled),
        Some(None) => {
            // "All" mode: save the full frozen image.
            screencopy::crop_and_save(full_rgba, None, &out_path)?;
            Ok(out_path)
        }
        Some(Some(FreezeSelection::Region(r))) => {
            // Translate global logical coordinates → image coordinates.
            // The image origin is (min_x, min_y) in global logical space.
            let adjusted = ScreenRect {
                x: r.x - min_x,
                y: r.y - min_y,
                w: r.w,
                h: r.h,
            };
            screencopy::crop_and_save(full_rgba, Some(adjusted), &out_path)?;
            Ok(out_path)
        }
        Some(Some(FreezeSelection::ToplevelWindow(window))) => {
            let mut captured = false;

            if cfg.window_use_toplevel_export && backend.kind() == CompositorKind::Hyprland {
                // Try toplevel-export first — captures the raw window buffer without
                // overlapping windows.
                if let Err(e) = toplevel_export::capture_toplevel_to_path(&window, &out_path) {
                    eprintln!(
                        "[hyprcrop] toplevel export failed ({}), falling back to screencopy crop",
                        e
                    );
                } else {
                    captured = true;
                }
            }

            if !captured {
                // Fall back to cropping the frozen screenshot.
                let adjusted = ScreenRect {
                    x: window.rect.x - min_x,
                    y: window.rect.y - min_y,
                    w: window.rect.w,
                    h: window.rect.h,
                }
                .expand(border_style.border_size);
                screencopy::crop_and_save(full_rgba, Some(adjusted), &out_path)?;
            }
            Ok(out_path)
        }
    }
}

/// Lifecycle of iced's lazily-built system font database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FontStatus {
    /// The scan is still running; text must not be shaped yet.
    Pending,
    /// The database is usable.
    Ready,
    /// The scan panicked. Text shaping is unsafe, so text widgets stay disabled
    /// for the rest of the session instead of retrying in a possibly poisoned
    /// state.
    Failed,
}

impl FontStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Ready,
            2 => Self::Failed,
            _ => Self::Pending,
        }
    }
}

/// Background warm-up of iced's system font database.
///
/// `iced_graphics` builds cosmic-text's font database lazily on the first call to
/// `font_system()`, which scans every installed font. That scan is usually around
/// 100 ms and can take seconds on a cold page cache with large font collections.
/// Starting it on a background thread lets it overlap screenshot capture, while
/// `AppState::text_ready` keeps views from shaping text until it completes.
#[derive(Debug, Clone)]
pub(crate) struct FontWarmup {
    state: Arc<AtomicU8>,
}

impl FontWarmup {
    /// Start the scan on a background thread.
    fn spawn() -> Self {
        Self::spawn_with(|| {
            let _ = iced::advanced::graphics::text::font_system();
        })
    }

    /// Run `init` on a background thread and publish the resulting status.
    ///
    /// Kept generic so orchestration can be tested without paying for a real font
    /// scan in the unit test suite.
    fn spawn_with(init: impl FnOnce() + Send + 'static) -> Self {
        let state = Arc::new(AtomicU8::new(FontStatus::Pending as u8));
        let shared = Arc::clone(&state);
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(init));
            let next = match outcome {
                Ok(()) => FontStatus::Ready,
                Err(_) => {
                    eprintln!(
                        "[hyprcrop] warning: font warm-up failed; freeze overlay text stays disabled"
                    );
                    FontStatus::Failed
                }
            };
            shared.store(next as u8, Ordering::Release);
        });
        Self { state }
    }

    fn status(&self) -> FontStatus {
        FontStatus::from_u8(self.state.load(Ordering::Acquire))
    }

    fn is_ready(&self) -> bool {
        self.status() == FontStatus::Ready
    }

    #[cfg(test)]
    fn with_status(status: FontStatus) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(status as u8)),
        }
    }

    #[cfg(test)]
    fn set_status(&self, status: FontStatus) {
        self.state.store(status as u8, Ordering::Release);
    }
}

fn resolve_initial_mode(
    buttons: &crate::domain::config::FreezeButtons,
    saved: CaptureMode,
) -> CaptureMode {
    // Saved mode may reference a button that has since been disabled.
    // Fall back to the first enabled mode so the UI starts in a valid state.
    let saved_enabled = match saved {
        CaptureMode::Crop => buttons.crop,
        CaptureMode::Window => buttons.window,
        CaptureMode::Monitor => buttons.monitor,
        CaptureMode::All => buttons.all,
    };
    if saved_enabled {
        saved
    } else if buttons.crop {
        CaptureMode::Crop
    } else if buttons.window {
        CaptureMode::Window
    } else if buttons.monitor {
        CaptureMode::Monitor
    } else if buttons.all {
        CaptureMode::All
    } else {
        // All capture-mode buttons are disabled; default to Crop so canvas
        // interactions (drag-to-select region) remain available even when the
        // toolbar is hidden or cancel-only.
        CaptureMode::Crop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn wait_for_status(warmup: &FontWarmup, expected: FontStatus) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while warmup.status() == FontStatus::Pending {
            assert!(
                Instant::now() < deadline,
                "warm-up status never left Pending"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(warmup.status(), expected);
    }

    #[test]
    fn warmup_becomes_ready_only_after_init_finishes() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();

        let warmup = FontWarmup::spawn_with(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });

        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("warm-up thread must start");
        assert_eq!(
            warmup.status(),
            FontStatus::Pending,
            "status must stay Pending while init runs"
        );

        release_tx.send(()).unwrap();
        wait_for_status(&warmup, FontStatus::Ready);
    }

    #[test]
    fn warmup_reports_failure_instead_of_ready() {
        let warmup = FontWarmup::spawn_with(|| panic!("simulated font scan failure"));

        wait_for_status(&warmup, FontStatus::Failed);
        assert!(
            !warmup.is_ready(),
            "a failed font scan must never be reported as ready"
        );
    }
}
