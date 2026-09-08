module fuse_stencil_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
    real(C_DOUBLE), intent(in) :: a(*)
    real(C_DOUBLE), intent(out) :: out(*)
    integer(C_INT64_T), value :: LEN_1D
    integer(C_INT64_T) :: i
    !
    do i = 2, LEN_1D - 2
       out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
    !
  end subroutine fuse_stencil_through_transient_fp64
end module fuse_stencil_mod
