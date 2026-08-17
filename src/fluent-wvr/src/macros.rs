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

#[cfg(test)]
mod tests {
    use crate::component_downcast_mut;
    use crate::component_downcast_ref;
    use crate::prelude::*;
    use internment::ArcIntern;
    use std::sync::Arc;

    // A concrete component for the concrete `impl_component!` arm.
    struct DemoUnit {
        name: String,
    }
    impl WorkUnit for DemoUnit {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("demo"))
        }
    }
    impl FieldAccess for DemoUnit {
        fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
            Ok(())
        }
        fn get_field(&self, _: &str) -> Result<String, FieldError> {
            Err(FieldError::NotFound("none".into()))
        }
        fn field_names(&self) -> &'static [&'static str] {
            &[]
        }
    }
    impl Describable for DemoUnit {
        fn describe(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }
    impl_component!(DemoUnit);

    #[test]
    fn impl_component_concrete_enables_downcast() {
        let comp: Arc<dyn Component> = Arc::new(DemoUnit { name: "demo".into() });
        assert!(component_downcast_ref::<DemoUnit>(&*comp).is_some());
        assert!(component_downcast_ref::<String>(&*comp).is_none());
        assert_eq!(comp.name(), "demo");
        let out = comp.execute(&WorkContext::default()).expect("execute");
        assert_eq!(out.message, "demo");
    }

    #[test]
    fn impl_component_concrete_mut_downcast() {
        let mut comp: Arc<dyn Component> = Arc::new(DemoUnit { name: "d".into() });
        let down = component_downcast_mut::<DemoUnit>(&mut comp).expect("downcast");
        down.name = "renamed".into();
        assert_eq!(
            component_downcast_ref::<DemoUnit>(&*comp).expect("ref").name,
            "renamed"
        );
    }

    // A fieldless unit for the concrete `impl_fieldless!` arm.
    struct Fieldless;
    impl WorkUnit for Fieldless {
        fn name(&self) -> &str {
            "fieldless"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("fieldless"))
        }
    }
    impl Describable for Fieldless {
        fn describe(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }
    impl_component!(Fieldless);
    impl_fieldless!(Fieldless);

    #[test]
    fn impl_fieldless_rejects_all_fields() {
        let mut f = Fieldless;
        assert!(matches!(f.set_field("x", "y"), Err(FieldError::NotFound(_))));
        assert!(matches!(f.get_field("x"), Err(FieldError::NotFound(_))));
        assert!(f.field_names().is_empty());
    }

    // A generic wrapper for the generic `impl_component!` arm.
    struct Wrap<U>(U);
    impl<U: Component> WorkUnit for Wrap<U> {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            self.0.depends()
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            self.0.provides()
        }
        fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            self.0.execute(ctx)
        }
    }
    impl<U: Component> FieldAccess for Wrap<U> {
        fn set_field(&mut self, n: &str, v: &str) -> Result<(), FieldError> {
            self.0.set_field(n, v)
        }
        fn get_field(&self, n: &str) -> Result<String, FieldError> {
            self.0.get_field(n)
        }
        fn field_names(&self) -> &'static [&'static str] {
            self.0.field_names()
        }
    }
    impl<U: Component> Describable for Wrap<U> {
        fn describe(&self) -> serde_json::Value {
            self.0.describe()
        }
    }
    impl_component!(generic (U: Component + 'static) for Wrap<U>);

    #[test]
    fn impl_component_generic_expands_for_wrapper() {
        let inner: Arc<dyn Component> = Arc::new(DemoUnit { name: "inner".into() });
        let comp: Arc<dyn Component> = Arc::new(Wrap(Arc::clone(&inner)));
        assert!(component_downcast_ref::<Wrap<Arc<dyn Component>>>(&*comp).is_some());
        assert_eq!(comp.name(), "inner");
    }
}
