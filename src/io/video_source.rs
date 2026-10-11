use crate::base::Source;
use crate::frame::Frame;
use image::RgbImage;
use serde_json::Value;
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};

/// Eine VideoSource, die Videos über FFmpeg in rohe RGB-Bytes liest.
pub struct VideoSource {
    /// Pfad zum Videodatei
    path: String,

    /// Handle zum FFmpeg-Prozess
    process: Option<VideoProcess>,

    /// Aktuelle Frame-Index
    frame_index: usize,

    /// Bildbreite (aus ffprobe)
    width: u32,

    /// Bildhöhe (aus ffprobe)
    height: u32,

    /// FPS Wert (aus ffprobe)
    fps: f64,

    /// Gesamtzahl Frames (sofern aus Metadaten ermittelbar)
    total_frames: Option<usize>,
}

/// Wrapper um den FFmpeg-Prozess
struct VideoProcess {
    child: Child,
    stdout: ChildStdout,
}

impl Source for VideoSource {
    fn next_frame(&mut self) -> Option<Frame> {
        let bytes_per_frame = (self.width * self.height * 3) as usize;
        let process = self.process.as_mut()?;

        let mut frame_data = vec![0u8; bytes_per_frame];
        match process.stdout.read_exact(&mut frame_data) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return None;
            }
            Err(e) => {
                eprintln!("Fehler beim Lesen von FFmpeg stdout: {}", e);
                return None;
            }
        }

        // Erstelle RgbImage aus rohen Daten mit RgbImage::from_raw
        let img = RgbImage::from_raw(self.width, self.height, frame_data)?;

        // Berechne Timestamp: frame_index * (1/fps)
        let timestamp = (self.frame_index as f64) / self.fps;
        let index = self.frame_index;
        self.frame_index += 1;

        Some(Frame::new(img, index, timestamp))
    }

    fn release(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.child.kill().ok();
            process.child.wait().ok();
        }
    }
}

impl VideoSource {
    /// Erstellt eine neue VideoSource für die gegebene Videodatei.
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let path_str = path.as_ref().display().to_string();

        // Zuerst Metadaten mit ffprobe ermitteln (width, height, fps, total_frames)
        let (width, height, fps, total_frames) = VideoSource::get_video_metadata(&path_str)?;

        // FFmpeg-Prozess starten
        let mut child = Command::new("ffmpeg")
            .args(["-i", &path_str, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("FFmpeg-Prozess start fehlgeschlagen: {}", e),
                )
            })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "Konnte stdout von FFmpeg-Process nicht lesen",
            )
        })?;

        let process = VideoProcess { child, stdout };

        Ok(Self {
            path: path_str,
            process: Some(process),
            frame_index: 0,
            width,
            height,
            fps,
            total_frames,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn total_frames(&self) -> Option<usize> {
        self.total_frames
    }

    /// Ermittelt Video-Metadaten mit ffprobe (Breite, Höhe, FPS, Gesamtzahl Frames).
    fn get_video_metadata(path: &str) -> std::io::Result<(u32, u32, f64, Option<usize>)> {
        let output = Command::new("ffprobe")
            .args([
                path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
            ])
            .output()
            .map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("ffprobe Fehler: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json: Value = serde_json::from_str(&stdout).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSON Fehler: {}", e),
            )
        })?;

        let streams = json
            .get("streams")
            .and_then(|v| v.as_array())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Keine streams"))?;

        let video_stream = streams
            .iter()
            .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("video"))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Kein Video-Stream")
            })?;

        let width = video_stream
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Keine Breite"))?
            as u32;

        let height = video_stream
            .get("height")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Keine Höhe"))?
            as u32;

        let fps_str = video_stream
            .get("r_frame_rate")
            .and_then(|v| v.as_str())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Kein FPS"))?;

        let fps: f64 = if let Some((num, den)) = fps_str.split_once('/') {
            let num_val = num.parse::<f64>().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Ungültiger FPS-Zähler")
            })?;
            let den_val = den.parse::<f64>().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Ungültiger FPS-Nenner")
            })?;
            num_val / den_val
        } else {
            fps_str.parse::<f64>().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Ungültiger FPS-Wert")
            })?
        };

        let total_frames: Option<usize> = video_stream
            .get("nb_frames")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| {
                let duration = video_stream
                    .get("duration")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())?;
                Some((duration * fps).round() as usize)
            });

        Ok((width, height, fps, total_frames))
    }
}