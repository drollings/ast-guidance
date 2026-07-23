//! Testing utilities for the router pipeline — mock mode, fixture loaders,
//! and test doubles. All components use fixture data — no live LLM, no network.

pub mod mock;

pub use mock::{MockFixtures, MockRouter, RouterOnlyMock};
