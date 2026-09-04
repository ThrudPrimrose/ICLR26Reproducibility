subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, ws, ws_bytes) bind(c, name='tsvc_2_vtvtv_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value, intent(in) :: a, b, c
  integer(c_int64_t), value, intent(in) :: len_1d
  type(c_ptr), value, intent(in) :: ws
  integer(c_int64_t), value, intent(in) :: ws_bytes
  real(c_double), dimension(:), pointer :: ap, bp, cp
  integer(c_int64_t) :: i
  integer :: nt

  call c_f_pointer(a, ap, [len_1d])
  call c_f_pointer(b, bp, [len_1d])
  call c_f_pointer(c, cp, [len_1d])

  nt = omp_get_max_threads()
  write(*,*) "PROBE len_1d=", len_1d, " max_threads=", nt, " ws_bytes=", ws_bytes
  call flush(6)

  !$omp parallel do
  do i = 1, len_1d
    ap(i) = ap(i) * bp(i) * cp(i)
  end do
end subroutine tsvc_2_vtvtv_fp64
