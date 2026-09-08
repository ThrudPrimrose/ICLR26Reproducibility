subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, vlen
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d), cc(len_2d, len_2d)

  integer(c_int64_t) :: j, i

!$omp parallel do simd schedule(static, 1)
  do j = 1, len_2d
    do i = (j - 1) * vlen + 1, len_2d
      aa(j, i) = bb(j, i) + cc(j, i)
    end do
  end do
!$omp end parallel do simd
end subroutine tsvc_2_s1232_fp64
