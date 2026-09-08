module fuse_diamond_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine fuse_diamond_fp64(a, out_arr, LEN_1D) bind(C, name="fuse_diamond_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out_arr(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: t

    !$omp parallel do simd private(i, t) schedule(static)
    do i = 1, LEN_1D
      t = a(i) * a(i)
      out_arr(i) = t * t - 1.0_c_double
    end do
    !$omp end parallel do simd
  end subroutine fuse_diamond_fp64
end module fuse_diamond_mod
