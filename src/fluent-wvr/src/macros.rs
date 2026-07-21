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
/// `Describable`, and `WorkUnit` impls. The macro accepts a single type
/// identifier (no generics — generic types should implement `Component`
/// by hand or via a per-instantiation macro call).
#[macro_export]
macro_rules! impl_component {
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
