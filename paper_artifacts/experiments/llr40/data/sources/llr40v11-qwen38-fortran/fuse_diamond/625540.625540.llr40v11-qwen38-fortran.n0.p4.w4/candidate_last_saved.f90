! TSVC tsvc_2_5 fuse_diamond, fused single pass: out(i) = (t+1)*(t-1), t = a(i)*a(i)
subroutine fuse_diamond_fp64(a, out, len_1d) bind(C, name="fuse_diamond_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in)  :: len_1d
  real(c_double),          intent(in)    :: a(len_1d)
  real(c_double),          intent(inout) :: out(len_1d)
  integer(c_int64_t) :: i
  real(c_double)     :: t

  !$omp parallel do default(none) shared(a, out, len_1d) private(t, i) schedule(static)
  do i = 1, len_1d
     t = a(i) * a(i)
     out(i) = (t + 1.0d0) * (t - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
