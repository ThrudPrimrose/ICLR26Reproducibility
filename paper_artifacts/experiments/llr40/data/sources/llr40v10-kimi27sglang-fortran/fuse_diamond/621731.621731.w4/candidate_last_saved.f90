subroutine fuse_diamond_fp64(a, out, n, workspace, ws_size) bind(c, name='fuse_diamond_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t, c_ptr, c_f_pointer
  implicit none
  type(c_ptr), value, intent(in) :: a, out
  integer(c_int64_t), value, intent(in) :: n
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: ws_size

  real(c_double), pointer, contiguous :: ap(:), outp(:)
  integer(c_int64_t) :: i
  real(c_double) :: t

  call c_f_pointer(a, ap, [n])
  call c_f_pointer(out, outp, [n])

  if (n > 1024_c_int64_t) then
    !$omp parallel do simd default(none) private(i, t) shared(ap, outp, n) aligned(ap, outp:64) simdlen(8)
    do i = 1, n
      t = ap(i) * ap(i)
      outp(i) = t * t - 1.0_c_double
    end do
    !$omp end parallel do simd
  else
    do i = 1, n
      t = ap(i) * ap(i)
      outp(i) = t * t - 1.0_c_double
    end do
  end if
end subroutine fuse_diamond_fp64
