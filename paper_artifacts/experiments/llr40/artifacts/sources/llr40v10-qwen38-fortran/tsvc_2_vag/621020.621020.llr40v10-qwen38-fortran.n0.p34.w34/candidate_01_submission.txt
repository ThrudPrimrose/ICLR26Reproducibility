subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name='tsvc_2_vag_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a, b, ip
  integer(c_int64_t), value :: len_1d
  real(c_double), pointer, contiguous, dimension(:) :: fa
  real(c_double), pointer, contiguous, dimension(:) :: fb
  integer(c_int32_t), pointer, contiguous, dimension(:) :: fip
  integer(c_int64_t) :: i

  call c_f_pointer(a, fa, [len_1d])
  call c_f_pointer(b, fb, [len_1d])
  call c_f_pointer(ip, fip, [len_1d])

  !$omp parallel do default(none) shared(fa,fb,fip,len_1d) private(i) schedule(static)
  do i = 1, len_1d
    fa(i) = fb(fip(i))
  end do
end subroutine tsvc_2_vag_fp64
