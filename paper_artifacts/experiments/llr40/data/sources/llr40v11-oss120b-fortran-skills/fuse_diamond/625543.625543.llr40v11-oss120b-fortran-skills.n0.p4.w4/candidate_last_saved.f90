module fuse_diamond_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_diamond_fp64(a, out, len_1d) bind(C, name="fuse_diamond_fp64")
    integer(c_int64_t), value, intent(in) :: len_1d
    real(c_double), intent(in) :: a(len_1d)
    real(c_double), intent(inout) :: out(len_1d)
    integer(c_int64_t) :: i
    real(c_double) :: t
    !$omp parallel do schedule(static) default(none) shared(a, out, len_1d) private(i, t)
    do i = 1, len_1d
      t = a(i) * a(i)
      out(i) = (t + 1.0d0) * (t - 1.0d0)
    end do
    !$omp end parallel do
  end subroutine fuse_diamond_fp64
end module fuse_diamond_mod
