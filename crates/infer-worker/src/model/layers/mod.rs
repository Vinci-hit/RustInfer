// src/model/layers/mod.rs
//
// Layer abstractions for different model architectures

pub mod weight_mapping;
pub mod decoder_layers;

pub use decoder_layers::DecoderLayers;
pub use weight_mapping::WeightMappingAdapter;
