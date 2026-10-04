/// Cold path function.
#[inline(always)]
#[cold]
pub const fn cold_path() {}

#[inline(always)]
pub const fn unlikely(cond: bool) -> bool {
    if cond {
        cold_path();
    }
    cond
}
