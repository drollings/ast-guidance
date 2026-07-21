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
