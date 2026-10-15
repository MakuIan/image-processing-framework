//! Zentrales Modul fuer Fortschrittsanzeigen (Progress Bars & Spinner).
//!
//! Stellt eine einheitliche, redundanzfreie Schnittstelle fuer alle
//! Verarbeitungsstufen (Bild-Pipeline, Video-Streaming, SOM-Kompression/Dekodierung)
//! bereit, inklusive genauer ETA-Restzeitanzeige und Fortschrittsmetriken.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::time::Duration;

/// Standard-Aktualisierungsrate fuer fluessige Anzeige ohne Konsolenflackern (10 Hz).
const DEFAULT_HZ: u8 = 10;

/// Erzeugt eine standardisierte Fortschrittsleiste mit ETA fuer Vorgaenge mit bekannter Schrittzahl.
pub fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_draw_target(ProgressDrawTarget::stdout_with_hz(DEFAULT_HZ));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta}) {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Erzeugt einen Spinner fuer Vorgaenge mit unbestimmter Gesamtzahl oder Stream-Verarbeitung.
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_draw_target(ProgressDrawTarget::stdout_with_hz(DEFAULT_HZ));
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("[{elapsed_precise}] {spinner:.cyan} {pos} Frames ({per_sec}) {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Erzeugt entweder eine Leiste mit ETA (falls `total` bekannt) oder einen Spinner.
pub fn create_task_progress(total: Option<usize>, message: &str) -> ProgressBar {
    match total {
        Some(t) => create_progress_bar(t as u64, message),
        None => create_spinner(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_progress_bar() {
        let pb = create_progress_bar(100, "Test Aufgabe");
        assert_eq!(pb.length(), Some(100));
        assert_eq!(pb.position(), 0);
        pb.inc(10);
        assert_eq!(pb.position(), 10);
        pb.finish_and_clear();
    }

    #[test]
    fn test_create_spinner() {
        let pb = create_spinner("Laufender Vorgang");
        assert_eq!(pb.length(), None);
        pb.inc(1);
        assert_eq!(pb.position(), 1);
        pb.finish_and_clear();
    }

    #[test]
    fn test_create_task_progress_dispatch() {
        let pb_fixed = create_task_progress(Some(50), "Mit Total");
        assert_eq!(pb_fixed.length(), Some(50));
        pb_fixed.finish_and_clear();

        let pb_stream = create_task_progress(None, "Ohne Total");
        assert_eq!(pb_stream.length(), None);
        pb_stream.finish_and_clear();
    }
}
