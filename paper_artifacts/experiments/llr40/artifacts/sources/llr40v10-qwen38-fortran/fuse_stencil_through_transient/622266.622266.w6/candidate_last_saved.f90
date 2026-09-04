subroutine fuse_stencil_through_transient_fp64(a_ptr, out_ptr, len_1d) bind(c, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a_ptr, out_ptr
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), pointer, contiguous :: a(:), out(:)
  integer(c_int64_t) :: i

  call c_f_pointer(a_ptr, a, [len_1d])
  call c_f_pointer(out_ptr, out, [len_1d])

  ! Fused vertical stencil:
  !   out(i) = (a(i-1)+a(i)+a(i+1)) * (a(i)+a(i+1)+a(i+2)), i = 2 .. LEN_1D-2
  !$omp parallel do default(none) shared(a,out,len_1d) schedule(static)
  do i = 2, len_1d - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  !$omp end parallel do
end subroutine fuse_stencil_through_transient_fp64
