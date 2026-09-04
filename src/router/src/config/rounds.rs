use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Fixed round count for rigor/any loop (VISION: terminate, don't loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedRounds(u8);
impl BoundedRounds {
    pub const MAX: u8 = 8;
    pub fn new(v: u8) -> Result<Self, String> {
        if v == 0 { return Err("max_passes must be >=1".into()); }
        if v > Self::MAX { return Err(format!("max_passes {} exceeds MAX={}", v, Self::MAX)); }
        Ok(Self(v))
    }
    pub fn rounds(self) -> usize { self.0 as usize }
    pub fn advance(self) -> Option<Self> {
        if self.0 > 1 { Some(Self(self.0-1)) } else { None }
    }
    pub fn get(self) -> u8 { self.0 }
}
impl TryFrom<u8> for BoundedRounds { type Error = String; fn try_from(v: u8) -> Result<Self,String>{ Self::new(v) } }
impl Default for BoundedRounds { fn default() -> Self { Self(2) } }
impl Serialize for BoundedRounds { fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error>{ s.serialize_u8(self.0) } }
impl<'de> Deserialize<'de> for BoundedRounds {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self,D::Error> {
        let v = u8::deserialize(d)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeverityThreshold(f64);
impl SeverityThreshold {
    pub fn new(v: f64) -> Result<Self,String> {
        if !(0.0..=1.0).contains(&v) { return Err(format!("SeverityThreshold {v} out of 0.0..=1.0")); }
        Ok(Self(v))
    }
    pub fn get(self) -> f64 { self.0 }
}
impl Default for SeverityThreshold { fn default() -> Self { Self(0.7) } }
impl Serialize for SeverityThreshold { fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error>{ s.serialize_f64(self.0) } }
impl<'de> Deserialize<'de> for SeverityThreshold {
    fn deserialize<D: Deserializer<'de>>(d:D)->Result<Self,D::Error>{ let v=f64::deserialize(d)?; Self::new(v).map_err(serde::de::Error::custom) }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EscalationConfidence(f64);
impl EscalationConfidence {
    pub fn new(v: f64)->Result<Self,String>{
        if !(0.0..=1.0).contains(&v){ return Err(format!("EscalationConfidence {v} out of 0.0..=1.0")); }
        Ok(Self(v))
    }
    pub fn get(self)->f64{ self.0 }
}
impl Default for EscalationConfidence{ fn default()->Self{ Self(0.4)}}
impl Serialize for EscalationConfidence{ fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok, S::Error>{ s.serialize_f64(self.0)}}
impl<'de> Deserialize<'de> for EscalationConfidence{
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{ let v=f64::deserialize(d)?; Self::new(v).map_err(serde::de::Error::custom)}
}
