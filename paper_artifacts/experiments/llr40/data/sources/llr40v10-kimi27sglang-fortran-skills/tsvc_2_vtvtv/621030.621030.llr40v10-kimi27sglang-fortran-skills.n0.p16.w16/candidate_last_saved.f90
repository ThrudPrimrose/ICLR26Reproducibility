subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  integer(c_int64_t) :: i
  integer(c_int) :: nt

  if (len_1d < 81920_c_int64_t) then
    !$omp simd
    do i = 1, len_1d
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end simd
  else
    nt = int(max(1_c_int64_t, min(int(omp_get_max_threads(), c_int64_t), len_1d / 2097152_c_int64_t)), c_int)
    !$omp parallel do simd num_threads(nt) proc_bind(close)
    do i = 1, len_1d
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do simd
  end if
end subroutine tsvc_2_vtvtv_fp64
