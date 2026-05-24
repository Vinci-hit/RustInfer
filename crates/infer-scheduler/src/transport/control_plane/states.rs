//! Type-state markers for [`super::ControlPlane`].
//!
//! `Bootstrapping` and `Running` are uninhabited-by-construction marker types
//! used as a `PhantomData` parameter on `ControlPlane<S>`. The transition
//! `ControlPlane<Bootstrapping> -> ControlPlane<Running>` is consuming, so
//! bootstrap-only API and runtime API can never both apply to the same value.

#[derive(Debug)]
pub struct Bootstrapping {
    _priv: (),
}

#[derive(Debug)]
pub struct Running {
    _priv: (),
}

impl Bootstrapping {
    pub(super) fn new() -> Self {
        Self { _priv: () }
    }
}

impl Running {
    pub(super) fn new() -> Self {
        Self { _priv: () }
    }
}
