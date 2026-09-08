subroutine tsvc_2_s231_fp64(aa, bb, n) bind(C, name="tsvc_2_s231_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*)
  integer(c_int64_t), value     :: n
  integer(c_int64_t) :: i, j
  print '(A,I0,A,I0,A,I0)', ' MAX_THREADS=', omp_get_max_threads(), ' NUM_THREADS=', omp_get_num_threads(), ' N=', n
  do i = 0, n - 1
    do j = 1, n - 1
      aa(j*n + i + 1) = aa((j-1)*n + i + 1) + bb(j*n + i + 1)
    end do
  end do
end subroutine tsvc_2_s231_fp64
