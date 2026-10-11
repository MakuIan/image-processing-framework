use crate::base::Filter;
use crate::filters::cpu::{BilateralFilter, BlurFilter, GaussianNoiseFilter};
use crate::filters::gpu::{GpuBilateralFilter, GpuBlurFilter, GpuGaussianNoiseFilter};

// --- Zentrale Standardwerte (Single Source of Truth) ---
pub const DEFAULT_NOISE_MEAN: f64 = 0.0;
pub const DEFAULT_NOISE_SIGMA: f64 = 15.0;

pub const DEFAULT_BILATERAL_RADIUS: i32 = 3;
pub const DEFAULT_BILATERAL_COLOR: f64 = 50.0;
pub const DEFAULT_BILATERAL_SPACE: f64 = 50.0;

pub const DEFAULT_BLUR_SIGMA: f32 = 1.0;

/// Beschreibt die Konfiguration eines Filters unabhaengig vom Ausfuehrungsziel (CPU/GPU).
#[derive(Debug, Clone, PartialEq)]
pub enum FilterConfig {
    GaussianNoise {
        mean: f64,
        sigma: f64,
    },
    Bilateral {
        radius: i32,
        sigma_color: f64,
        sigma_space: f64,
    },
    Blur {
        sigma: f32,
    },
}

impl FilterConfig {
    // Re-Export der Konstanten auf dem Typ
    pub const DEFAULT_NOISE_MEAN: f64 = DEFAULT_NOISE_MEAN;
    pub const DEFAULT_NOISE_SIGMA: f64 = DEFAULT_NOISE_SIGMA;
    pub const DEFAULT_BILATERAL_RADIUS: i32 = DEFAULT_BILATERAL_RADIUS;
    pub const DEFAULT_BILATERAL_COLOR: f64 = DEFAULT_BILATERAL_COLOR;
    pub const DEFAULT_BILATERAL_SPACE: f64 = DEFAULT_BILATERAL_SPACE;
    pub const DEFAULT_BLUR_SIGMA: f32 = DEFAULT_BLUR_SIGMA;

    /// Erzeugt eine Standard-Konfiguration fuer Gaussian Noise.
    pub fn default_gaussian_noise() -> Self {
        Self::GaussianNoise {
            mean: DEFAULT_NOISE_MEAN,
            sigma: DEFAULT_NOISE_SIGMA,
        }
    }

    /// Erzeugt eine Standard-Konfiguration fuer den Bilateral-Filter.
    pub fn default_bilateral() -> Self {
        Self::Bilateral {
            radius: DEFAULT_BILATERAL_RADIUS,
            sigma_color: DEFAULT_BILATERAL_COLOR,
            sigma_space: DEFAULT_BILATERAL_SPACE,
        }
    }

    /// Erzeugt eine Standard-Konfiguration fuer den Blur-Filter.
    pub fn default_blur() -> Self {
        Self::Blur {
            sigma: DEFAULT_BLUR_SIGMA,
        }
    }

    /// Gibt die Standard-Filterkette (Noise -> Bilateral -> Blur) zurueck.
    pub fn default_pipeline() -> Vec<Self> {
        vec![
            Self::default_gaussian_noise(),
            Self::default_bilateral(),
            Self::default_blur(),
        ]
    }

    /// Erzeugt die konkrete Filter-Instanz als Box<dyn Filter>
    /// je nachdem, ob GPU- oder CPU-Verarbeitung gewuenscht ist.
    pub fn instantiate(&self, gpu: bool) -> Box<dyn Filter> {
        if gpu {
            match *self {
                FilterConfig::GaussianNoise { mean, sigma } => {
                    Box::new(GpuGaussianNoiseFilter::new(mean, sigma))
                }
                FilterConfig::Bilateral {
                    radius,
                    sigma_color,
                    sigma_space,
                } => Box::new(GpuBilateralFilter::new(radius, sigma_color, sigma_space)),
                FilterConfig::Blur { sigma } => Box::new(GpuBlurFilter::new(sigma)),
            }
        } else {
            match *self {
                FilterConfig::GaussianNoise { mean, sigma } => {
                    Box::new(GaussianNoiseFilter::new(mean, sigma))
                }
                FilterConfig::Bilateral {
                    radius,
                    sigma_color,
                    sigma_space,
                } => Box::new(BilateralFilter::new(radius, sigma_color, sigma_space)),
                FilterConfig::Blur { sigma } => Box::new(BlurFilter::new(sigma)),
            }
        }
    }
}
