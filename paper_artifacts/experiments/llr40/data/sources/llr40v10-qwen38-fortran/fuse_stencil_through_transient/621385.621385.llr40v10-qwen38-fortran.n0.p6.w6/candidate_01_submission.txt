subroutine fuse_stencil_through_transient_fp64(a, out, len_1d) &
    bind(C, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(in)  :: a
  real(c_double), dimension(*), intent(out) :: out
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t) :: i

  if (len_1d > 3) then
    !$omp parallel do default(none) shared(a,out,len_1d) schedule(static)
    do i = 2, len_1d - 2
      out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
    !$omp end parallel do
  end if
end subroutine fuse_stencil_through_transient_fp64
