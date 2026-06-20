#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Granularity {
    PerTensor,
    PerChannel,
    PerGroup,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Symmetry {
    Symmetric,
    Asymmetric,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Packing {
    AwqInt4,
    GptqInt4,
    Int8,
    Fp8,
    Mxfp4,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuantScheme {
    pub granularity: Granularity,
    pub symmetry: Symmetry,
    pub packing: Packing,
    pub group: usize,
}

impl QuantScheme {
    pub const AWQ_INT4_G128: QuantScheme = QuantScheme {
        granularity: Granularity::PerGroup,
        symmetry: Symmetry::Asymmetric,
        packing: Packing::AwqInt4,
        group: 128,
    };

    pub const fn logical_per_word(self) -> usize {
        match self.packing {
            Packing::AwqInt4 | Packing::GptqInt4 => 8,
            Packing::Int8 | Packing::Fp8 => 1,
            Packing::Mxfp4 => 8,
        }
    }
}
