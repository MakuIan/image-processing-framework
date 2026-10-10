use crate::base::Sink;
use crate::frame::Frame;
use image::RgbImage;
use std::io::Write;
use std::process::{Child, ChildStdin, Command};

/// VideoSink schreibt Frames ueber FFmpeg in eine Videodatei.
pub struct VideoSink {
    /// Pfad zur Ausgabetdatei
    output_path: String,

    /// FPS Wert fuer die Ausgabe
    fps: f32,

    /// Bildbreite (wird aus dem ersten Frame ermittelt oder im Konstruktor)
    width: Option<u32>,

    /// Bildhoehe (wird aus dem ersten Frame ermittelt oder im Konstruktor)
    height: Option<u32>,

    /// Handle zum FFmpeg-Prozess
    process: Option<VideoSinkProcess>,
}

/// Wrapper um den FFmpeg-Prozess fuer den Sink
struct VideoSinkProcess {
    child: Child,
    stdin: ChildStdin,
}

impl VideoSink {
    /// Erstellt einen neuen VideoSink.
    ///
    /// - `output_path`: Pfad zur auszugebenden Videodatei
    /// - `fps`: Frames pro Sekunde fuer die Ausgabe
    pub fn new<P: AsRef<std::path::Path>>(output_path: P, fps: f32) -> Self {
        let output_path = output_path.as_ref().display().to_string();

        // Der FFmpeg-Process wird bei Bedarf gestartet (wenn width/height bekannt sind)
        let process = None;

        Self {
            output_path,
            fps,
            width: None,
            height: None,
            process,
        }
    }

    /// Startet den FFmpeg-Process wenn die Aeuflsung bekannt ist.
    fn start_process(&mut self) -> std::io::Result<()> {
        let (width, height) = match (&self.width, &self.height) {
            (Some(w), Some(h)) => (*w, *h),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Aeuflsung unbekannt - width und height muessen gesetzt sein",
                ))
            }
        };

        let fps = self.fps;

        // FFmpeg-Command:
        // -y : Ueberschreiben ohne Abfrage
        // -f rawvideo : rohes Videoformat als Input
        // -pix_fmt rgb24 : RGB24 als Eingang
        // -s {width}x{height} : Aeuflsung setzen
        // -r {fps} : FPS einstellen
        // -i - : stdin als Eingabe
        // -c:v libx264 : H.264 Codec
        // -pix_fmt yuv420p : YUV420P als Ausgabe (kompatibel mit meisten Playern)
        let mut child = Command::new("ffmpeg")
            .args([
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
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                &self.output_path,
            ])
            .stdin(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("FFmpeg-Prozess start fehlgeschlagen: {}", e),
                )
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "Konnte stdin von FFmpeg-Process nicht lesen",
            )
        })?;

        self.process = Some(VideoSinkProcess { child, stdin });
        Ok(())
    }
}

impl Sink for VideoSink {
    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
        // Starte FFmpeg wenn noch nicht gestartet
        if self.process.is_none() {
            self.start_process()?;
        }

        // Konvertiere RgbImage zu rohen RGB-Bytes (vor dem borrow)
        let raw_bytes = self.rgb_image_to_raw(&frame.data());

        let Some(process) = &mut self.process else {
            return Err("FFmpeg-Process nicht gestartet".into());
        };

        // Schreibe alle Bytes an stdin
        process
            .stdin
            .write_all(&raw_bytes)
            .map_err(|e| -> Box<dyn std::error::Error> {
                format!("Fehler beim Schreiben an FFmpeg stdin: {}", e).into()
            })?;

        Ok(())
    }

    fn release(&mut self) {
        if let Some(mut process) = self.process.take() {
            // stdin automatisch geschlossen (durch drop des Process-Wrappers)
            process.child.wait().ok();

            // Exit-Status pruefen
            process.child.try_wait().ok();
        }
    }
}

impl VideoSink {
    /// Setzt die Breite fuer den FFmpeg-Prozess (wird normalerweise aus dem ersten Frame ermittelt)
    pub fn set_width(&mut self, width: u32) {
        self.width = Some(width);
    }

    /// Setzt die Hoehe fuer den FFmpeg-Prozess (wird normalerweise aus dem ersten Frame ermittelt)
    pub fn set_height(&mut self, height: u32) {
        self.height = Some(height);
    }

    /// Konvertiert ein RgbImage zu rohen RGB-Bytes im row-major Format.
    /// Die Bytes sind in der Reihenfolge: [R0, G0, B0, R1, G1, B1, R2, ...] fuer jede Reihe.
    fn rgb_image_to_raw(&self, img: &RgbImage) -> Vec<u8> {
        let width = img.width();
        let height = img.height();

        // Gesamte Anzahl Bytes berechnen (width * height * 3 fuer RGB24)
        let total_bytes = (width * height * 3) as usize;

        let mut buffer = Vec::with_capacity(total_bytes);

        // RgbImage ist bereits in row-major order, wir muessen einfach die Daten extrahieren
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                buffer.push(pixel[0]); // R
                buffer.push(pixel[1]); // G
                buffer.push(pixel[2]); // B
            }
        }

        buffer
    }
}
