module fuse_diamond_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name="fuse_diamond_fp64")
    ! Arguments
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value :: LEN_1D
    ! Locals
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
