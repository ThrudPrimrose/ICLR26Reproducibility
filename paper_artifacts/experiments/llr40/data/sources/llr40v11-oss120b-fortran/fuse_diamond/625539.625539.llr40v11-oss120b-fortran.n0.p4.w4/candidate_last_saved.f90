module fuse_diamond_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name="fuse_diamond_fp64")
    ! Arguments: a - input array, out - output array, LEN_1D - length of arrays
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: t
    !$omp parallel do default(none) shared(a, out, LEN_1D) private(i, t)
    do i = 1, LEN_1D
      t = a(i) * a(i)
      out(i) = t * t - 1.0_c_double
    end do
    !$omp end parallel do
  end subroutine fuse_diamond_fp64
end module fuse_diamond_mod
