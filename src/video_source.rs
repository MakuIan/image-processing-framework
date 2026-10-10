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

    /// Puffer für rohe Daten aus stdout
    buffer: Vec<u8>,

    /// Aktuelle Frame-Index
    frame_index: usize,

    /// Bildbreite (aus ffprobe)
    width: u32,

    /// Bildhöhe (aus ffprobe)
    height: u32,

    /// FPS Wert (aus ffprobe)
    fps: f64,
}

/// Wrapper um den FFmpeg-Prozess
struct VideoProcess {
    child: Child,
    stdout: ChildStdout,
}

impl Source for VideoSource {
    fn next_frame(&mut self) -> Option<Frame> {
        let bytes_per_frame = (self.width * self.height * 3) as usize;

        // Fülle Puffer aus stdout wenn nötig
        while self.buffer.len() < bytes_per_frame {
            match &mut self.process {
                Some(process) => {
                    match process.stdout.read_to_end(&mut self.buffer) {
                        Ok(0) => return None, // EOF erreicht
                        Ok(_) => {
                            if self.buffer.len() < bytes_per_frame {
                                return None; // Nicht genug Daten für einen Frame
                            }
                        }
                        Err(e) => {
                            eprintln!("Fehler beim Lesen von FFmpeg stdout: {}", e);
                            return None;
                        }
                    }
                }
                None => return None,
            }
        }

        // Lese genau bytes_per_frame Bytes für einen Frame
        let frame_data = self.buffer.drain(..bytes_per_frame).collect::<Vec<u8>>();

        // Erstelle RgbImage aus rohen Daten mit RgbImage::from_raw
        let img = RgbImage::from_raw(self.width, self.height, frame_data)?;

        // Berechne Timestamp: frame_index * (1/fps)
        let timestamp = (self.frame_index as f64) / self.fps;

        self.frame_index += 1;

        Some(Frame::new(img, self.frame_index - 1, timestamp))
    }

    fn release(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.child.kill().ok();
            process.child.wait().ok();
        }
        self.buffer.clear();
    }
}

impl VideoSource {
    /// Erstellt eine neue VideoSource für die gegebene Videodatei.
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref().display().to_string();

        // Zuerst metadaten mit ffprobe ermitteln (width, height, fps)
        let (width, height, fps) = VideoSource::get_video_metadata(&path)?;

        // FFmpeg-Process starten:
        // -i INPUT : Eingabedatei laden
        // -f rawvideo : rohes Videoformat ausgeben
        // -pix_fmt rgb24 : RGB24 Pixelformat (3 Bytes pro Pixel)
        // - : stdout als Output
        let mut child = Command::new("ffmpeg")
            .args(["-i", &path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()) // stderr ignorieren
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
            path,
            process: Some(process),
            buffer: Vec::new(),
            frame_index: 0,
            width,
            height,
            fps,
        })
    }

    /// Ermittelt Video-Metadaten mit ffprobe (Breite, Höhe, FPS).
    fn get_video_metadata(path: &str) -> std::io::Result<(u32, u32, f64)> {
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

        // FPS aus r_frame_rate (z.B. "30000/1001" -> 30000/1001.0)
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

        Ok((width, height, fps))
    }
}
