/// Eliminates the 7-line `as_any`/`as_any_mut` boilerplate that every
/// `Component` implementor must write.
///
/// The `Component` trait (see above) requires `as_any` and `as_any_mut`
/// for runtime type identification after `Arc<dyn Component>` erasure.
/// Every implementor writes the same stanza; this macro restores the
/// "blanket-impl feel" of the original design.
///
/// # Usage
///
/// ```text
/// // #[derive(FieldAccess, Describable)]
/// // struct MyConfig { port: u16 }
/// // impl WorkUnit for MyConfig { ... }
/// // impl_component!(MyConfig);
/// ```
///
/// Place `impl_component!(MyConfig);` after the `FieldAccess`,
/// `Describable`, and `WorkUnit` impls.
///
/// # Generic types
///
/// For generic types, use the generic form:
///
/// ```ignore
/// impl_component!(generic (U: crate::Component + 'static) for Instrumented<U>);
/// ```
#[macro_export]
macro_rules! impl_component {
    // Generic form: `generic (bounds) for Type<...>`
    (generic ( $($generics:tt)* ) for $type:ty) => {
        impl < $($generics)* > $crate::Component for $type {
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                self
            }
        }
    };
    // Concrete type (no generics)
    ($type:ty) => {
        impl $crate::Component for $type {
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                self
            }
        }
    };
}

/// Eliminates the `FieldAccess` no-op trio for components that expose no
/// configurable fields: `set_field`/`get_field` → `FieldError::NotFound`,
/// `field_names` → `&[]`.
///
/// Mirrors `impl_component!`'s two arms (concrete + generic). The `NotFound`
/// message derives from `stringify!($type)`, so it stays
/// `"<TypeName> has no configurable fields"` for concrete types.
///
/// # Usage
///
/// ```text
/// // struct MyConfig { ... }   // no `#[derive(FieldAccess)]`
/// impl_fieldless!(MyConfig);
/// ```
///
/// Do **not** use this on types with real fields (`CommandUnit`, WASM
/// components) — those must derive or hand-implement `FieldAccess`.
#[macro_export]
macro_rules! impl_fieldless {
    // Generic form: `generic (bounds) for Type<...>`
    (generic ( $($generics:tt)* ) for $type:ty) => {
        impl < $($generics)* > $crate::FieldAccess for $type {
            fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), $crate::FieldError> {
                Err($crate::FieldError::NotFound(format!(
                    "{} has no configurable fields",
                    stringify!($type)
                )))
            }
            fn get_field(&self, _name: &str) -> Result<String, $crate::FieldError> {
                Err($crate::FieldError::NotFound(format!(
                    "{} has no configurable fields",
                    stringify!($type)
                )))
            }
            fn field_names(&self) -> &'static [&'static str] {
                &[]
            }
        }
    };
    // Concrete type (no generics)
    ($type:ty) => {
        impl $crate::FieldAccess for $type {
            fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), $crate::FieldError> {
                Err($crate::FieldError::NotFound(format!(
                    "{} has no configurable fields",
                    stringify!($type)
                )))
            }
            fn get_field(&self, _name: &str) -> Result<String, $crate::FieldError> {
                Err($crate::FieldError::NotFound(format!(
                    "{} has no configurable fields",
                    stringify!($type)
                )))
            }
            fn field_names(&self) -> &'static [&'static str] {
                &[]
            }
        }
    };
}

