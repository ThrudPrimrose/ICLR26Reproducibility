subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int32_t), intent(in) :: ip(len_1d)

  integer(c_int64_t) :: i

  ! gather: a(i) = b(ip(i)); each write lands at this iteration's own
  ! subscript, ip is 1-based as handed in -- pure PARALLEL loop
  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = b(ip(i))
  end do
end subroutine tsvc_2_vag_fp64
