use crate::base::Sink;
use crate::frame::Frame;
use std::io::Write;
use std::process::{Child, ChildStdin, Command};

/// Unterstützte Video-Codecs für die Ausgabe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoCodec {
    #[default]
    H265, // HEVC
    H264, // AVC (Abwärtskompatibilität)
}

impl VideoCodec {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "h264" | "avc" | "x264" => VideoCodec::H264,
            "h265" | "hevc" | "x265" | _ => VideoCodec::H265,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VideoCodec::H265 => "H.265 / HEVC",
            VideoCodec::H264 => "H.264 / AVC",
        }
    }
}

/// Unterstützte Video-Encoder
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoEncoder {
    #[default]
    Auto,
    Amf,   // AMD AMF (hevc_amf / h264_amf)
    Nvenc, // NVIDIA NVENC (hevc_nvenc / h264_nvenc)
    Qsv,   // Intel QuickSync (hevc_qsv / h264_qsv)
    Cpu,   // CPU x265 / x264 (libx265 / libx264)
}

impl VideoEncoder {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "amf" | "amd" => VideoEncoder::Amf,
            "nvenc" | "nvidia" => VideoEncoder::Nvenc,
            "qsv" | "intel" => VideoEncoder::Qsv,
            "cpu" | "x264" | "x265" | "software" => VideoEncoder::Cpu,
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

    /// Konfigurierter Codec (H.265 Standard)
    codec: VideoCodec,

    /// Optionaler CRF-/QP-Qualitätswert
    crf: Option<u32>,
}

/// Wrapper um den FFmpeg-Prozess für den Sink
struct VideoSinkProcess {
    child: Child,
    stdin: ChildStdin,
}

impl VideoSink {
    /// Erstellt einen neuen VideoSink mit Standard-H.265 und Auto-Encoder.
    pub fn new<P: AsRef<std::path::Path>>(output_path: P, fps: f32) -> Self {
        let output_path = output_path.as_ref().display().to_string();

        Self {
            output_path,
            fps,
            width: None,
            height: None,
            process: None,
            encoder: VideoEncoder::Auto,
            codec: VideoCodec::H265,
            crf: None,
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

    /// Setzt den Video-Codec (H.265 / H.264)
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    pub fn set_codec(&mut self, codec: VideoCodec) {
        self.codec = codec;
    }

    /// Setzt den CRF / QP Qualitätswert (z. B. 32 für H.265, 26 für H.264)
    pub fn with_crf(mut self, crf: u32) -> Self {
        self.crf = Some(crf);
        self
    }

    pub fn set_crf(&mut self, crf: Option<u32>) {
        self.crf = crf;
    }

    /// Ermittelt den passenden Encoder anhand von Codec und Konfiguration
    fn resolve_encoder(&self) -> (String, Vec<String>) {
        match self.codec {
            VideoCodec::H265 => self.resolve_hevc_encoder(),
            VideoCodec::H264 => self.resolve_h264_encoder(),
        }
    }

    fn resolve_hevc_encoder(&self) -> (String, Vec<String>) {
        let crf = self.crf.unwrap_or(32);
        let qp_p = crf.saturating_add(2);
        match self.encoder {
            VideoEncoder::Amf => (
                format!(
                    "AMD AMF Hardware Encoder (HEVC / H.265, CQP: {}/{})",
                    crf, qp_p
                ),
                vec![
                    "-c:v".into(),
                    "hevc_amf".into(),
                    "-quality".into(),
                    "balanced".into(),
                    "-rc".into(),
                    "cqp".into(),
                    "-qp_i".into(),
                    crf.to_string(),
                    "-qp_p".into(),
                    qp_p.to_string(),
                ],
            ),
            VideoEncoder::Nvenc => (
                format!("NVIDIA NVENC Hardware Encoder (HEVC / H.265, CQ: {})", crf),
                vec![
                    "-c:v".into(),
                    "hevc_nvenc".into(),
                    "-preset".into(),
                    "p4".into(),
                    "-cq".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Qsv => (
                format!(
                    "Intel QuickSync Hardware Encoder (HEVC / H.265, Global Quality: {})",
                    crf
                ),
                vec![
                    "-c:v".into(),
                    "hevc_qsv".into(),
                    "-global_quality".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Cpu => (
                format!("CPU Software Encoder (libx265 veryfast, CRF: {})", crf),
                vec![
                    "-c:v".into(),
                    "libx265".into(),
                    "-preset".into(),
                    "veryfast".into(),
                    "-crf".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Auto => {
                if Self::test_ffmpeg_codec("hevc_amf") {
                    (
                        format!(
                            "Auto: AMD AMF Hardware Encoder (HEVC / H.265, CQP: {}/{})",
                            crf, qp_p
                        ),
                        vec![
                            "-c:v".into(),
                            "hevc_amf".into(),
                            "-quality".into(),
                            "balanced".into(),
                            "-rc".into(),
                            "cqp".into(),
                            "-qp_i".into(),
                            crf.to_string(),
                            "-qp_p".into(),
                            qp_p.to_string(),
                        ],
                    )
                } else if Self::test_ffmpeg_codec("hevc_nvenc") {
                    (
                        format!(
                            "Auto: NVIDIA NVENC Hardware Encoder (HEVC / H.265, CQ: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "hevc_nvenc".into(),
                            "-preset".into(),
                            "p4".into(),
                            "-cq".into(),
                            crf.to_string(),
                        ],
                    )
                } else if Self::test_ffmpeg_codec("hevc_qsv") {
                    (
                        format!(
                            "Auto: Intel QuickSync Hardware Encoder (HEVC / H.265, Global Quality: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "hevc_qsv".into(),
                            "-global_quality".into(),
                            crf.to_string(),
                        ],
                    )
                } else if Self::test_ffmpeg_codec("libx265") {
                    (
                        format!(
                            "Auto: CPU Software Encoder (libx265 veryfast, CRF: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "libx265".into(),
                            "-preset".into(),
                            "veryfast".into(),
                            "-crf".into(),
                            crf.to_string(),
                        ],
                    )
                } else {
                    eprintln!("  Warnung: Kein HEVC/H.265-Encoder verfuegbar. Fallback auf H.264.");
                    self.resolve_h264_encoder()
                }
            }
        }
    }

    fn resolve_h264_encoder(&self) -> (String, Vec<String>) {
        let crf = self.crf.unwrap_or(26);
        let qp_p = crf.saturating_add(2);
        match self.encoder {
            VideoEncoder::Amf => (
                format!(
                    "AMD AMF Hardware Encoder (H.264 / AVC, CQP: {}/{})",
                    crf, qp_p
                ),
                vec![
                    "-c:v".into(),
                    "h264_amf".into(),
                    "-quality".into(),
                    "balanced".into(),
                    "-rc".into(),
                    "cqp".into(),
                    "-qp_i".into(),
                    crf.to_string(),
                    "-qp_p".into(),
                    qp_p.to_string(),
                ],
            ),
            VideoEncoder::Nvenc => (
                format!("NVIDIA NVENC Hardware Encoder (H.264 / AVC, CQ: {})", crf),
                vec![
                    "-c:v".into(),
                    "h264_nvenc".into(),
                    "-preset".into(),
                    "p4".into(),
                    "-cq".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Qsv => (
                format!(
                    "Intel QuickSync Hardware Encoder (H.264 / AVC, Global Quality: {})",
                    crf
                ),
                vec![
                    "-c:v".into(),
                    "h264_qsv".into(),
                    "-global_quality".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Cpu => (
                format!("CPU Software Encoder (libx264 veryfast, CRF: {})", crf),
                vec![
                    "-c:v".into(),
                    "libx264".into(),
                    "-preset".into(),
                    "veryfast".into(),
                    "-crf".into(),
                    crf.to_string(),
                ],
            ),
            VideoEncoder::Auto => {
                if Self::test_ffmpeg_codec("h264_amf") {
                    (
                        format!(
                            "Auto: AMD AMF Hardware Encoder (H.264 / AVC, CQP: {}/{})",
                            crf, qp_p
                        ),
                        vec![
                            "-c:v".into(),
                            "h264_amf".into(),
                            "-quality".into(),
                            "balanced".into(),
                            "-rc".into(),
                            "cqp".into(),
                            "-qp_i".into(),
                            crf.to_string(),
                            "-qp_p".into(),
                            qp_p.to_string(),
                        ],
                    )
                } else if Self::test_ffmpeg_codec("h264_nvenc") {
                    (
                        format!(
                            "Auto: NVIDIA NVENC Hardware Encoder (H.264 / AVC, CQ: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "h264_nvenc".into(),
                            "-preset".into(),
                            "p4".into(),
                            "-cq".into(),
                            crf.to_string(),
                        ],
                    )
                } else if Self::test_ffmpeg_codec("h264_qsv") {
                    (
                        format!(
                            "Auto: Intel QuickSync Hardware Encoder (H.264 / AVC, Global Quality: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "h264_qsv".into(),
                            "-global_quality".into(),
                            crf.to_string(),
                        ],
                    )
                } else {
                    (
                        format!(
                            "Auto: CPU Software Encoder (libx264 veryfast, CRF: {})",
                            crf
                        ),
                        vec![
                            "-c:v".into(),
                            "libx264".into(),
                            "-preset".into(),
                            "veryfast".into(),
                            "-crf".into(),
                            crf.to_string(),
                        ],
                    )
                }
            }
        }
    }

    fn test_ffmpeg_codec(codec: &str) -> bool {
        Command::new("ffmpeg")
            .args([
                "-v",
                "quiet",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.01",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
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
                ));
            }
        };

        let fps = self.fps;
        let (encoder_name, encoder_args) = self.resolve_encoder();
        println!("  Verwende Video-Encoder: {}", encoder_name);

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", width, height),
            "-r",
            fps.to_string().as_str(),
            "-i",
            "-",
        ]);
        cmd.args(&encoder_args);
        cmd.args(["-pix_fmt", "yuv420p", &self.output_path]);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            std::io::Error::other(format!("FFmpeg-Prozess start fehlgeschlagen: {}", e))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("Konnte stdin von FFmpeg-Prozess nicht öffnen"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_from_str() {
        assert_eq!(VideoCodec::from_str_loose("h265"), VideoCodec::H265);
        assert_eq!(VideoCodec::from_str_loose("hevc"), VideoCodec::H265);
        assert_eq!(VideoCodec::from_str_loose("x265"), VideoCodec::H265);
        assert_eq!(VideoCodec::from_str_loose("h264"), VideoCodec::H264);
        assert_eq!(VideoCodec::from_str_loose("avc"), VideoCodec::H264);
        assert_eq!(VideoCodec::from_str_loose("x264"), VideoCodec::H264);
        assert_eq!(VideoCodec::from_str_loose("default"), VideoCodec::H265);
    }

    #[test]
    fn test_video_encoder_from_str() {
        assert_eq!(VideoEncoder::from_str_loose("amf"), VideoEncoder::Amf);
        assert_eq!(VideoEncoder::from_str_loose("amd"), VideoEncoder::Amf);
        assert_eq!(VideoEncoder::from_str_loose("nvenc"), VideoEncoder::Nvenc);
        assert_eq!(VideoEncoder::from_str_loose("nvidia"), VideoEncoder::Nvenc);
        assert_eq!(VideoEncoder::from_str_loose("qsv"), VideoEncoder::Qsv);
        assert_eq!(VideoEncoder::from_str_loose("intel"), VideoEncoder::Qsv);
        assert_eq!(VideoEncoder::from_str_loose("cpu"), VideoEncoder::Cpu);
        assert_eq!(VideoEncoder::from_str_loose("x265"), VideoEncoder::Cpu);
        assert_eq!(VideoEncoder::from_str_loose("unknown"), VideoEncoder::Auto);
    }

    #[test]
    fn test_hevc_encoder_resolution() {
        let sink = VideoSink::new("test.mp4", 30.0)
            .with_codec(VideoCodec::H265)
            .with_encoder(VideoEncoder::Cpu)
            .with_crf(26);

        let (name, args) = sink.resolve_encoder();
        assert!(name.contains("libx265"));
        assert!(args.contains(&"-c:v".to_string()));
        assert!(args.contains(&"libx265".to_string()));
        assert!(args.contains(&"-crf".to_string()));
        assert!(args.contains(&"26".to_string()));
    }

    #[test]
    fn test_h264_encoder_resolution() {
        let sink = VideoSink::new("test.mp4", 30.0)
            .with_codec(VideoCodec::H264)
            .with_encoder(VideoEncoder::Cpu)
            .with_crf(22);

        let (name, args) = sink.resolve_encoder();
        assert!(name.contains("libx264"));
        assert!(args.contains(&"-c:v".to_string()));
        assert!(args.contains(&"libx264".to_string()));
        assert!(args.contains(&"-crf".to_string()));
        assert!(args.contains(&"22".to_string()));
    }
}
