subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j
  integer(c_int64_t) :: n, lo0, hi0
  integer :: nt, t

  !$omp parallel private(i, j, n, lo0, hi0, nt, t)
  n = LEN_2D - 8_c_int64_t
  nt = omp_get_num_threads()
  t  = omp_get_thread_num()
  lo0 = (n * int(t, c_int64_t)) / int(nt, c_int64_t)
  hi0 = (n * int(t + 1, c_int64_t)) / int(nt, c_int64_t) - 1_c_int64_t

  do j = 9_c_int64_t, LEN_2D
    !$omp simd
    do i = lo0 + 9_c_int64_t, hi0 + 9_c_int64_t
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
    !$omp end simd
  end do

  do i = 9_c_int64_t, LEN_2D
    !$omp simd
    do j = lo0 + 9_c_int64_t, hi0 + 9_c_int64_t
      bb(j, i) = bb(j, i - 1) + cc(j, i)
    end do
    !$omp end simd
  end do
  !$omp end parallel

end subroutine tsvc_2_s2233_fp64
