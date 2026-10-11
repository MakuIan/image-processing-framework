use crate::base::Sink;
use crate::frame::Frame;
use std::io::Write;
use std::process::{Child, ChildStdin, Command};

/// Unterstützte Video-Encoder
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoEncoder {
    #[default]
    Auto,
    Amf,       // AMD AMF (h264_amf)
    Nvenc,     // NVIDIA NVENC (h264_nvenc)
    Qsv,       // Intel QuickSync (h264_qsv)
    Cpu,       // CPU x264 (libx264 -preset veryfast)
}

impl VideoEncoder {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "amf" | "amd" => VideoEncoder::Amf,
            "nvenc" | "nvidia" => VideoEncoder::Nvenc,
            "qsv" | "intel" => VideoEncoder::Qsv,
            "cpu" | "x264" => VideoEncoder::Cpu,
            _ => VideoEncoder::Auto,
        }
    }
}

/// VideoSink schreibt Frames über FFmpeg in eine Videodatei.
pub struct VideoSink {
    /// Pfad zur Ausgabedatei
    output_path: String,

    /// FPS Wert für die Ausgabe
    fps: f32,

    /// Bildbreite (wird aus dem ersten Frame ermittelt oder im Konstruktor)
    width: Option<u32>,

    /// Bildhöhe (wird aus dem ersten Frame ermittelt oder im Konstruktor)
    height: Option<u32>,

    /// Handle zum FFmpeg-Prozess
    process: Option<VideoSinkProcess>,

    /// Konfigurierter Encoder
    encoder: VideoEncoder,
}

/// Wrapper um den FFmpeg-Prozess für den Sink
struct VideoSinkProcess {
    child: Child,
    stdin: ChildStdin,
}

impl VideoSink {
    /// Erstellt einen neuen VideoSink.
    pub fn new<P: AsRef<std::path::Path>>(output_path: P, fps: f32) -> Self {
        let output_path = output_path.as_ref().display().to_string();

        Self {
            output_path,
            fps,
            width: None,
            height: None,
            process: None,
            encoder: VideoEncoder::Auto,
        }
    }

    /// Setzt den Encoder
    pub fn with_encoder(mut self, encoder: VideoEncoder) -> Self {
        self.encoder = encoder;
        self
    }

    pub fn set_encoder(&mut self, encoder: VideoEncoder) {
        self.encoder = encoder;
    }

    /// Ermittelt den passenden Encoder (bei Auto prüft es AMF, NVENC, QSV und fällt auf CPU zurück)
    fn resolve_encoder(&self) -> (&'static str, Vec<String>) {
        match self.encoder {
            VideoEncoder::Amf => ("AMD AMF Hardware Encoder", vec!["-c:v".into(), "h264_amf".into()]),
            VideoEncoder::Nvenc => ("NVIDIA NVENC Hardware Encoder", vec!["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into()]),
            VideoEncoder::Qsv => ("Intel QuickSync Hardware Encoder", vec!["-c:v".into(), "h264_qsv".into()]),
            VideoEncoder::Cpu => ("CPU Software Encoder (libx264 veryfast)", vec!["-c:v".into(), "libx264".into(), "-preset".into(), "veryfast".into()]),
            VideoEncoder::Auto => {
                if Self::test_ffmpeg_codec("h264_amf") {
                    ("Auto: AMD AMF Hardware Encoder", vec!["-c:v".into(), "h264_amf".into()])
                } else if Self::test_ffmpeg_codec("h264_nvenc") {
                    ("Auto: NVIDIA NVENC Hardware Encoder", vec!["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into()])
                } else if Self::test_ffmpeg_codec("h264_qsv") {
                    ("Auto: Intel QuickSync Hardware Encoder", vec!["-c:v".into(), "h264_qsv".into()])
                } else {
                    ("Auto: CPU Software Encoder (libx264 veryfast)", vec!["-c:v".into(), "libx264".into(), "-preset".into(), "veryfast".into()])
                }
            }
        }
    }

    fn test_ffmpeg_codec(codec: &str) -> bool {
        Command::new("ffmpeg")
            .args([
                "-v", "quiet",
                "-y",
                "-f", "lavfi",
                "-i", "color=c=black:s=256x256:d=0.01",
                "-c:v", codec,
                "-pix_fmt", "yuv420p",
                "-f", "null",
                "-",
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Startet den FFmpeg-Prozess wenn die Auflösung bekannt ist.
    fn start_process(&mut self) -> std::io::Result<()> {
        let (width, height) = match (&self.width, &self.height) {
            (Some(w), Some(h)) => (*w, *h),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Auflösung unbekannt - width und height müssen gesetzt sein",
                ))
            }
        };

        let fps = self.fps;
        let (encoder_name, encoder_args) = self.resolve_encoder();
        println!("  Verwende Video-Encoder: {}", encoder_name);

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{}x{}", width, height),
            "-r", fps.to_string().as_str(),
            "-i", "-",
        ]);
        cmd.args(&encoder_args);
        cmd.args([
            "-pix_fmt", "yuv420p",
            &self.output_path,
        ]);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("FFmpeg-Prozess start fehlgeschlagen: {}", e),
            )
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "Konnte stdin von FFmpeg-Prozess nicht öffnen",
            )
        })?;

        self.process = Some(VideoSinkProcess { child, stdin });
        Ok(())
    }

    pub fn set_width(&mut self, width: u32) {
        self.width = Some(width);
    }

    pub fn set_height(&mut self, height: u32) {
        self.height = Some(height);
    }
}

impl Sink for VideoSink {
    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
        if self.width.is_none() {
            self.width = Some(frame.width());
        }
        if self.height.is_none() {
            self.height = Some(frame.height());
        }

        if self.process.is_none() {
            self.start_process()?;
        }

        let Some(process) = &mut self.process else {
            return Err("FFmpeg-Prozess nicht gestartet".into());
        };

        process.stdin.write_all(frame.data().as_raw()).map_err(
            |e| -> Box<dyn std::error::Error> {
                format!("Fehler beim Schreiben an FFmpeg stdin: {}", e).into()
            },
        )?;

        Ok(())
    }

    fn release(&mut self) {
        if let Some(mut process) = self.process.take() {
            drop(process.stdin);
            process.child.wait().ok();
        }
    }
}