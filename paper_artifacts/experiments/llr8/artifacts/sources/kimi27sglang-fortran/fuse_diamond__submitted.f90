module fuse_diamond_mod
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
contains
  subroutine fuse_diamond_fp64(a, out, LEN_1D, workspace, workspace_size) bind(c, name='fuse_diamond_fp64')
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: out(LEN_1D)
    integer(c_int8_t), intent(in) :: workspace(*)
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i
    real(c_double) :: t

    if (LEN_1D < 16384) then
      do i = 1, LEN_1D
        t = a(i) * a(i)
        out(i) = (t + 1.0_c_double) * (t - 1.0_c_double)
      end do
    else
      !$omp parallel do simd private(i, t) schedule(static)
      do i = 1, LEN_1D
        t = a(i) * a(i)
        out(i) = (t + 1.0_c_double) * (t - 1.0_c_double)
      end do
      !$omp end parallel do simd
    end if
  end subroutine
end module
